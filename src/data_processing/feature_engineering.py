import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def run_feature_engineering():
    # Define absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "Processed data")
    
    print("[Features] Loading preprocessed intermediate datasets...")
    customers_path = os.path.join(PROCESSED_DIR, "preprocessed_customers.csv")
    articles_path = os.path.join(PROCESSED_DIR, "preprocessed_articles.csv")
    transactions_path = os.path.join(PROCESSED_DIR, "preprocessed_transactions.csv")
    
    if not (os.path.exists(customers_path) and os.path.exists(articles_path) and os.path.exists(transactions_path)):
        print(f"Error: Missing intermediate preprocessed files in {PROCESSED_DIR}")
        sys.exit(1)
        
    customers = pd.read_csv(customers_path)
    articles = pd.read_csv(articles_path)
    transactions = pd.read_csv(transactions_path)
    
    # 1. Parse date features from transactions
    print("[Features] Engineering temporal/date features...")
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["year"] = transactions["t_dat"].dt.year
    transactions["month"] = transactions["t_dat"].dt.month
    transactions["day"] = transactions["t_dat"].dt.day
    transactions["day_of_week"] = transactions["t_dat"].dt.dayofweek
    transactions["week"] = transactions["t_dat"].dt.isocalendar().week.astype(int)
    transactions["interaction"] = 1
    
    # 2. Categorical columns label encoding
    print("[Features] Performing categorical Label Encoding...")
    categorical_article_cols = [
        "prod_name", "product_type_name", "product_group_name",
        "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
        "perceived_colour_master_name", "department_name", "index_name",
        "index_group_name", "section_name", "garment_group_name"
    ]
    
    categorical_customer_cols = ["club_member_status", "fashion_news_frequency"]
    
    for col in categorical_article_cols:
        if col in articles.columns:
            le = LabelEncoder()
            articles[col] = le.fit_transform(articles[col].astype(str))
            
    for col in categorical_customer_cols:
        if col in customers.columns:
            le = LabelEncoder()
            customers[col] = le.fit_transform(customers[col].astype(str))
            
    # 3. Scaling continuous values
    print("[Features] Scaling age and price features using MinMaxScaler...")
    scaler_age = MinMaxScaler()
    scaler_price = MinMaxScaler()
    
    customers[["age"]] = scaler_age.fit_transform(customers[["age"]])
    transactions[["price"]] = scaler_price.fit_transform(transactions[["price"]])
    
    # Ensure ID columns are string for reliable joining
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)
    articles["article_id"] = articles["article_id"].astype(str)
    
    # 4. Merge datasets
    print("[Features] Merging transactions, customers, and articles tables...")
    merged_data = transactions.merge(customers, on="customer_id", how="left")
    merged_data = merged_data.merge(articles, on="article_id", how="left")
    
    # Drop rows missing indices or timestamps
    merged_data = merged_data.dropna(subset=["user_idx", "item_idx", "t_dat"])
    
    # 5. Engagement features: user interaction counts
    print("[Features] Computing customer interaction counts...")
    interaction_counts = merged_data.groupby("customer_id")["article_id"].count().reset_index()
    interaction_counts.columns = ["customer_id", "interaction_count"]
    merged_data = merged_data.merge(interaction_counts, on="customer_id", how="left")
    
    # 6. Popularity features: article purchase counts
    print("[Features] Computing article popularity scores...")
    article_popularity = merged_data.groupby("article_id")["customer_id"].count().reset_index()
    article_popularity.columns = ["article_id", "popularity_score"]
    merged_data = merged_data.merge(article_popularity, on="article_id", how="left")
    
    # Sort chronologically
    print("[Features] Sorting features chronologically...")
    final_data = merged_data.sort_values(by="t_dat").reset_index(drop=True)
    
    # 7. Save outputs
    print("[Features] Saving final engineered feature sets...")
    final_data.to_csv(os.path.join(PROCESSED_DIR, "processed_data.csv"), index=False)
    final_data.to_csv(os.path.join(PROCESSED_DIR, "final_features.csv"), index=False)
    
    print(f"[Features] Done. Final dataset shape: {final_data.shape}")

if __name__ == "__main__":
    run_feature_engineering()
