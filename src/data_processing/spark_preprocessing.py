import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data():
    # Define absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DIR = os.path.join(BASE_DIR, "Data", "Raw data")
    PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "Processed data")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("[Preprocessing] Loading raw datasets...")
    articles_path = os.path.join(RAW_DIR, "articles_hm.csv")
    customers_path = os.path.join(RAW_DIR, "customer_hm.csv")
    transactions_path = os.path.join(RAW_DIR, "transactions_hm.csv")
    
    if not (os.path.exists(articles_path) and os.path.exists(customers_path) and os.path.exists(transactions_path)):
        print(f"Error: Missing raw data files in {RAW_DIR}")
        sys.exit(1)
        
    articles = pd.read_csv(articles_path)
    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path)
    
    print(f"[Preprocessing] Initial shapes: Articles={articles.shape}, Customers={customers.shape}, Transactions={transactions.shape}")
    
    # 1. Drop duplicates
    print("[Preprocessing] Removing duplicates...")
    articles.drop_duplicates(inplace=True)
    customers.drop_duplicates(inplace=True)
    transactions.drop_duplicates(inplace=True)
    
    # 2. Standardise column names
    print("[Preprocessing] Standardising column names...")
    articles.columns = articles.columns.str.lower().str.strip()
    customers.columns = customers.columns.str.lower().str.strip()
    transactions.columns = transactions.columns.str.lower().str.strip()
    
    # 3. Validate critical columns
    required_transactions = ["t_dat", "customer_id", "article_id", "price"]
    missing_transactions = [col for col in required_transactions if col not in transactions.columns]
    if missing_transactions:
        raise ValueError(f"Missing columns in transactions: {missing_transactions}")
        
    if "customer_id" not in customers.columns:
        raise ValueError("Missing customer_id column in customers dataset")
        
    if "article_id" not in articles.columns:
        raise ValueError("Missing article_id column in articles dataset")
        
    # 4. Fill missing values
    print("[Preprocessing] Fulfilling missing values...")
    if "detail_desc" in articles.columns:
        articles["detail_desc"] = articles["detail_desc"].fillna("No Description")
        
    for col in articles.columns:
        if articles[col].dtype == "object":
            articles[col] = articles[col].fillna("UNKNOWN")
            
    for col in ["club_member_status", "fashion_news_frequency"]:
        if col not in customers.columns:
            customers[col] = "UNKNOWN"
        customers[col] = customers[col].fillna("UNKNOWN")
        
    if "age" not in customers.columns:
        customers["age"] = 0
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    age_imputer = SimpleImputer(strategy="median")
    customers["age"] = age_imputer.fit_transform(customers[["age"]]).ravel()
    
    transactions["price"] = pd.to_numeric(transactions["price"], errors="coerce")
    transactions["price"] = transactions["price"].fillna(0.0)
    
    # 5. Type conversions and dropping nulls
    print("[Preprocessing] Performing type conversions...")
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"], errors="coerce")
    transactions = transactions.dropna(subset=["t_dat", "customer_id", "article_id"])
    
    articles["article_id"] = articles["article_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    
    # 6. Generate indices mapping
    print("[Preprocessing] Generating user/item index mapping tables...")
    user_map = pd.DataFrame({"customer_id": transactions["customer_id"].unique()})
    user_map["user_idx"] = np.arange(len(user_map))
    
    item_map = pd.DataFrame({"article_id": transactions["article_id"].unique()})
    item_map["item_idx"] = np.arange(len(item_map))
    
    # Merge indices into transactions
    transactions = transactions.merge(user_map, on="customer_id", how="left")
    transactions = transactions.merge(item_map, on="article_id", how="left")
    
    # 7. Save intermediate outputs
    print(f"[Preprocessing] Saving preprocessed outputs under {PROCESSED_DIR}...")
    user_map.to_csv(os.path.join(PROCESSED_DIR, "user_id_map.csv"), index=False)
    item_map.to_csv(os.path.join(PROCESSED_DIR, "item_id_map.csv"), index=False)
    
    customers.to_csv(os.path.join(PROCESSED_DIR, "preprocessed_customers.csv"), index=False)
    articles.to_csv(os.path.join(PROCESSED_DIR, "preprocessed_articles.csv"), index=False)
    transactions.to_csv(os.path.join(PROCESSED_DIR, "preprocessed_transactions.csv"), index=False)
    
    print(f"[Preprocessing] Done. Cleaned shapes: Articles={articles.shape}, Customers={customers.shape}, Transactions={transactions.shape}")

if __name__ == "__main__":
    preprocess_data()
