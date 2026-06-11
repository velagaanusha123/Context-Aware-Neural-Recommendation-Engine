import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import subprocess
from datetime import datetime


# =========================================================
# PATH
# =========================================================

sys.path.append(os.path.abspath("."))


# =========================================================
# BACKEND IMPORT
# =========================================================

BACKEND_AVAILABLE = False

try:

    from Backend.recommender import (
        recommend_for_user,
        get_sample_users,
        get_item_catalogue,
        _CONTEXT
    )


    CONTEXT = _CONTEXT

    BACKEND_AVAILABLE = True


except Exception as e:

    st.error(f"Backend loading error: {e}")



# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="H&M Neural Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# =========================================================
# CSS
# =========================================================

st.markdown(
"""
<style>

.main{
background:#f8f9fa;
}

.stButton button{
width:100%;
background:black;
color:white;
border-radius:10px;
height:3em;
}

.item-card{

background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 4px 12px #ddd;
margin-bottom:20px;

}

</style>

""",
unsafe_allow_html=True
)



# =========================================================
# IMAGE
# =========================================================

def get_image_url(article_id):

    article = str(article_id).zfill(10)

    folder = article[:3]


    path = os.path.join(
        "assets",
        "images",
        folder,
        f"{article}.jpg"
    )


    if os.path.exists(path):
        return path


    return (
        "https://images.unsplash.com/"
        "photo-1558769132-cb1aea458c5e"
        "?w=300&h=400&fit=crop"
    )



# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛍️ H&M Recommender")


if BACKEND_AVAILABLE:


    users = get_sample_users(500)


    selected_user = st.sidebar.selectbox(
        "👤 Select User",
        users
    )


    top_k = st.sidebar.slider(
        "Recommendations",
        1,
        50,
        10
    )


    months = (
        ["All Months"] +
        [
            datetime(2020,m,1).strftime("%B")
            for m in range(1,13)
        ]
    )


    selected_month = st.sidebar.selectbox(
        "📅 Season",
        months
    )


    month_filter = (
        months.index(selected_month)
        if selected_month != "All Months"
        else None
    )



# =========================================================
# MAIN
# =========================================================

st.title(
    "🛍️ Context-Aware H&M Recommendation Engine"
)


tabs = st.tabs(
[
"🎯 Recommendations",
"📊 Analytics",
"🧠 Model"
]
)



# =========================================================
# RECOMMENDATIONS
# =========================================================


with tabs[0]:


    if not BACKEND_AVAILABLE:

        st.warning(
            "Backend not loaded"
        )


    else:


        c1,c2,c3 = st.columns(3)


        c1.metric(
            "User",
            selected_user
        )


        c2.metric(
            "Top K",
            top_k
        )


        c3.metric(
            "Status",
            "Online"
        )



        if st.button(
            "✨ Generate Recommendations"
        ):


            with st.spinner(
                "Searching recommendations..."
            ):


                recs = recommend_for_user(
                    selected_user,
                    top_k=top_k,
                    month_filter=month_filter
                )



                if not recs:

                    st.warning(
                        "No recommendations found"
                    )


                else:


                    cols = st.columns(3)


                    for i,item in enumerate(recs):

                        with cols[i%3]:


                            st.markdown(
                            f"""

                            <div class="item-card">


                            <img src="{get_image_url(
                            item.get('article_id')
                            )}"
                            width="100%">


                            <h3>
                            {item.get('prod_name','Product')}
                            </h3>


                            <p>
                            {item.get('product_type','')}
                            |
                            {item.get('colour','')}
                            </p>


                            <b>
                            Score:
                            {round(
                            item.get('score',0)*100,
                            2
                            )}%
                            </b>


                            </div>

                            """,
                            unsafe_allow_html=True
                            )





# =========================================================
# ANALYTICS
# =========================================================


with tabs[1]:


    st.header("📊 Analytics")


    if BACKEND_AVAILABLE:


        if not CONTEXT.empty:


            df = (
                CONTEXT
                .groupby("product_group_name")
                .size()
                .reset_index(
                    name="count"
                )
            )


            fig = px.bar(
                df,
                x="product_group_name",
                y="count",
                title="Product Category Popularity"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )



        else:

            st.info(
                "No analytics data available"
            )





# =========================================================
# MODEL
# =========================================================


with tabs[2]:


    st.header(
        "🧠 Model Management"
    )


    st.write(
        "Neural Recommendation Model"
    )


    st.code(
"""
python Backend/neural_model.py
"""
    )


    if st.button(
        "🚀 Start Training"
    ):


        try:


            subprocess.run(
                [
                    "python",
                    "Backend/neural_model.py"
                ]
            )


            st.success(
                "Training completed"
            )


        except Exception as e:


            st.error(
                str(e)
            )




st.markdown("---")

st.markdown(
"Built with ❤️ | Context-Aware Neural Recommendation Engine"
)