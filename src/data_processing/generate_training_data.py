import os
import sys
import json
import pandas as pd

def generate_splits():
    # Define absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(BASE_DIR, "Data", "Processed data")
    
    print("[Splits] Loading final engineered feature set...")
    features_path = os.path.join(PROCESSED_DIR, "final_features.csv")
    
    if not os.path.exists(features_path):
        print(f"Error: Missing final features file: {features_path}")
        sys.exit(1)
        
    final_data = pd.read_csv(features_path)
    final_data["t_dat"] = pd.to_datetime(final_data["t_dat"])
    
    # 1. Chronological splitting
    print("[Splits] Performing chronological split around 2019-11-01...")
    split_date = pd.Timestamp("2019-11-01")
    
    train_data = final_data[final_data["t_dat"] < split_date].copy()
    valid_data = final_data[final_data["t_dat"] >= split_date].copy()
    
    if train_data.empty:
        raise ValueError("Train data split is empty. Check interaction date range.")
    if valid_data.empty:
        raise ValueError("Validation data split is empty. Check interaction date range.")
        
    # 2. Extract model-ready columns
    model_ready_columns = [
        "user_idx", "item_idx", "interaction", "price", "year", "month", "week",
        "day_of_week", "age", "interaction_count", "popularity_score",
        "club_member_status", "fashion_news_frequency", "product_type_name",
        "product_group_name", "colour_group_name", "section_name", "garment_group_name"
    ]
    
    available_cols = [col for col in model_ready_columns if col in final_data.columns]
    
    print("[Splits] Generating target model-ready dataset...")
    model_ready_dataset = final_data[available_cols].copy()
    model_ready_dataset = model_ready_dataset.fillna(0)
    model_ready_dataset = model_ready_dataset.drop_duplicates()
    
    # Subsample to target size (100k) as in raw preprocessing notebook
    target_samples = min(100000, len(model_ready_dataset))
    print(f"[Splits] Sampling {target_samples} records for model-ready training...")
    model_ready_dataset = model_ready_dataset.sample(n=target_samples, random_state=42)
    
    # 3. Save datasets
    print("[Splits] Exporting splits and model-ready CSVs...")
    train_data.to_csv(os.path.join(PROCESSED_DIR, "train_context_aware_recommendation.csv"), index=False)
    valid_data.to_csv(os.path.join(PROCESSED_DIR, "valid_context_aware_recommendation.csv"), index=False)
    model_ready_dataset.to_csv(os.path.join(PROCESSED_DIR, "model_ready_recommendation_dataset.csv"), index=False)
    
    # 4. Export analytics summary
    print("[Splits] Creating JSON execution summary metadata...")
    summary = {
        "final_rows": int(len(final_data)),
        "train_rows": int(len(train_data)),
        "valid_rows": int(len(valid_data)),
        "unique_users": int(final_data["customer_id"].nunique() if "customer_id" in final_data.columns else 0),
        "unique_items": int(final_data["article_id"].nunique() if "article_id" in final_data.columns else 0),
        "date_min": str(final_data["t_dat"].min()),
        "date_max": str(final_data["t_dat"].max())
    }
    
    with open(os.path.join(PROCESSED_DIR, "preprocessing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[Splits] Done. Preprocessing metrics: {summary}")

if __name__ == "__main__":
    generate_splits()
