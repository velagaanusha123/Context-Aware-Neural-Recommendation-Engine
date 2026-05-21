import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from datetime import datetime

# Add root to path to ensure Backend imports work
sys.path.append(os.path.abspath("."))

try:
    from Backend.recommender import (
        recommend_for_user,
        get_sample_users,
        get_item_catalogue,
        _load_neural_model,
        _POP,
        _CONTEXT
    )
    from Backend.neural_model import train_neural_model
    BACKEND_AVAILABLE = True
except Exception as e:
    st.error(f"Error loading backend: {e}")
    BACKEND_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Page Config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="H&M Neural Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #000000;
        color: white;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: white;
    }
    .item-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .item-card:hover {
        transform: translateY(-5px);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_image_url(article_id, product_group):
    # Folder is the first 3 digits of article_id (e.g. 706 for 706016003)
    article_str = str(article_id).zfill(10)
    folder = article_str[:3]
    local_path = os.path.join("assets", "images", folder, f"{article_str}.jpg")
    
    if os.path.exists(local_path):
        return local_path
        
    # Placeholder fallback
    mapping = {
        "Dresses Ladies": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=300&h=400&fit=crop",
        "Trousers": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=300&h=400&fit=crop",
        "Knitwear": "https://images.unsplash.com/photo-1576188973526-0e5d7422b4ad?w=300&h=400&fit=crop",
        "Outdoor": "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=300&h=400&fit=crop",
        "Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=400&fit=crop",
        "Accessories": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300&h=400&fit=crop",
        "Jersey Basic": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=300&h=400&fit=crop",
    }
    return mapping.get(product_group, "https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=300&h=400&fit=crop")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛍️ H&M Recommender")
st.sidebar.markdown("---")

if BACKEND_AVAILABLE:
    user_ids = get_sample_users(500)
    selected_user = st.sidebar.selectbox("👤 Select User Profile", user_ids, index=0)
    
    st.sidebar.markdown("### ⚙️ Filters")
    top_k = st.sidebar.slider("Number of Recommendations", 1, 50, 10)
    
    months = ["All Months"] + [datetime(2020, m, 1).strftime('%B') for m in range(1, 13)]
    selected_month_name = st.sidebar.selectbox("📅 Seasonal Context", months)
    month_filter = months.index(selected_month_name) if selected_month_name != "All Months" else None
    
    st.sidebar.markdown("---")
    st.sidebar.info("This system uses Neural Collaborative Filtering (NCF) to predict fashion preferences.")

# --------------------------------------------------------------------------- #
# Main Content
# --------------------------------------------------------------------------- #
st.title("Context-Aware Fashion Engine")

tabs = st.tabs(["🎯 Recommendations", "📊 Analytics", "🧠 Model Management"])

# --- TAB 1: Recommendations ---
with tabs[0]:
    if not BACKEND_AVAILABLE:
        st.warning("Backend logic not loaded. Please check the logs.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("User ID", f"#{selected_user}")
        col2.metric("Top K", top_k)
        col3.metric("Context", selected_month_name)
        col4.metric("Status", "Online", delta="Neural")

        if st.button("✨ Generate Recommendations"):
            with st.spinner("Analyzing style patterns..."):
                recs = recommend_for_user(selected_user, top_k=top_k, month_filter=month_filter)
                
                if not recs:
                    st.write("No recommendations found for this context.")
                else:
                    # Display in grid
                    cols = st.columns(3)
                    for idx, item in enumerate(recs):
                        with cols[idx % 3]:
                            st.markdown(f"""
                                <div class="item-card">
                                    <img src="{get_image_url(item['article_id'], item['product_group'])}" style="width:100%; border-radius:5px; margin-bottom:10px;">
                                    <h3 style="margin-bottom:5px; font-size:1.1rem;">{item['prod_name']}</h3>
                                    <p style="color:gray; font-size:0.9rem;">{item['product_type']} • {item['colour']}</p>
                                    <div style="display:flex; justify-content:between; align-items:center;">
                                        <span style="font-weight:bold; color:#d32f2f;">Relevance: {int(item['score']*100)}%</span>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

# --- TAB 2: Analytics ---
with tabs[1]:
    st.header("Project Insights")
    
    if BACKEND_AVAILABLE:
        # 1. Popularity Distribution
        pop_data = pd.DataFrame(list(_POP.items()), columns=['item_idx', 'purchases'])
        fig_pop = px.histogram(pop_data, x="purchases", nbins=50, 
                               title="Item Popularity Distribution",
                               color_discrete_sequence=['#000000'])
        st.plotly_chart(fig_pop, use_container_width=True)
        
        # 2. Seasonal Trends
        st.subheader("Seasonal Category Trends")
        seasonal_counts = _CONTEXT.groupby(['month', 'product_group_name']).size().reset_index(name='counts')
        # Map month numbers to names
        seasonal_counts['month_name'] = seasonal_counts['month'].apply(lambda x: datetime(2020, x, 1).strftime('%b'))
        
        fig_trend = px.line(seasonal_counts, x="month_name", y="counts", color="product_group_name",
                            title="Category Popularity by Month",
                            markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # 3. Data Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique Users", len(_CONTEXT['user_idx'].unique()))
        c2.metric("Unique Items", len(_CONTEXT['item_idx'].unique()))
        c3.metric("Total Interactions", len(_CONTEXT))

# --- TAB 3: Model Management ---
with tabs[2]:
    st.header("Neural Model Training")
    st.write("Retrain the Neural Collaborative Filtering model using the latest interaction data.")
    
    if st.button("🚀 Start Model Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            BASE = os.path.abspath(".")
            DATA_PATH = os.path.join(BASE, "Data", "Processed data", "model_ready_recommendation_dataset.csv")
            SAVE_PATH = os.path.join(BASE, "model", "ncf_model.pth")
            
            status_text.text("Training in progress... (3 Epochs)")
            # Simulated progress since actual training prints to console
            for i in range(1, 101, 10):
                time.sleep(0.5)
                progress_bar.progress(i)
            
            train_neural_model(DATA_PATH, SAVE_PATH)
            _load_neural_model() # Reload
            
            progress_bar.progress(100)
            st.success("Neural model trained and loaded successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

st.markdown("---")
st.markdown("Built with ❤️ by AI Assistant | H&M Personalized Fashion Recommendations")
