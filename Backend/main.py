import os
import sys

# Add the project root to sys.path so 'Backend' can be imported when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# Import real recommender (loads data on startup)
from Backend.recommender import (
    recommend_for_user,
    get_sample_users,
    get_item_catalogue,
    _load_neural_model,
)
from Backend.neural_model import train_neural_model

# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Context-Aware H&M Recommendation Engine",
    description=(
        "Neural collaborative-filtering recommender built on the H&M "
        "Personalized Fashion Recommendations dataset."
    ),
    version="2.0.0",
)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(BASE, "assets")),
    name="assets"
)


# Allow frontend (any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response schemas
# --------------------------------------------------------------------------- #

class RecommendRequest(BaseModel):
    user_idx: int
    top_k: int = 10
    month_filter: Optional[int] = None   # 1-12 for seasonal context


class RecommendationItem(BaseModel):
    item_idx: int
    article_id: Optional[int] = None
    prod_name: str
    product_type: Optional[str] = None
    product_group: Optional[str] = None
    colour: Optional[str] = None
    section: Optional[str] = None
    garment_group: Optional[str] = None
    score: Optional[float] = None
    popularity: Optional[int] = None


class RecommendResponse(BaseModel):
    user_idx: int
    top_k: int
    month_filter: Optional[int]
    recommendations: List[RecommendationItem]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/", summary="Health check", tags=["System"])
def health_check():
    return {"status": "online", "version": "2.0.0", "dataset": "H&M Fashion"}


@app.get("/users/sample", summary="Get sample user IDs", tags=["Users"])
def sample_users(n: int = Query(default=100, ge=1, le=500)):
    """Return up to *n* user IDs that exist in the dataset (for demo use)."""
    users = get_sample_users(n)
    return {"count": len(users), "user_ids": users}


@app.get("/items", summary="Browse item catalogue", tags=["Items"])
def browse_items(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return a paginated slice of the H&M article catalogue."""
    items = get_item_catalogue(limit=limit, offset=offset)
    return {"count": len(items), "offset": offset, "items": items}


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get personalised recommendations",
    tags=["Recommendations"],
)
def recommend(req: RecommendRequest):
    """
    Return top-K H&M product recommendations for the given *user_idx*.

    Optionally pass `month_filter` (1–12) to restrict recommendations to items
    popular in that month (seasonal context awareness).
    """
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 50")
    if req.month_filter is not None and not (1 <= req.month_filter <= 12):
        raise HTTPException(status_code=422, detail="month_filter must be 1–12")

    try:
        recs = recommend_for_user(
            user_idx=req.user_idx,
            top_k=req.top_k,
            month_filter=req.month_filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendResponse(
        user_idx=req.user_idx,
        top_k=req.top_k,
        month_filter=req.month_filter,
        recommendations=[RecommendationItem(**r) for r in recs],
    )


@app.post("/train", summary="Train the Neural Model", tags=["System"])
def train():
    """
    Trigger training of the Neural Collaborative Filtering model.
    This uses the preprocessed H&M dataset.
    """
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE, "Data", "Processed data", "model_ready_recommendation_dataset.csv")
    SAVE_PATH = os.path.join(BASE, "model", "ncf_model.pth")
    
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
        
    try:
        train_neural_model(DATA_PATH, SAVE_PATH)
        # Reload the model into the recommender
        _load_neural_model()
        return {"status": "success", "message": "Neural model trained and loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/analytics/stats", tags=["Analytics"])
def get_stats():
    from Backend.recommender import _CONTEXT
    return {
        "users": int(len(_CONTEXT['user_idx'].unique())),
        "items": int(len(_CONTEXT['item_idx'].unique())),
        "interactions": int(len(_CONTEXT))
    }
    
@app.get("/analytics/seasonal", tags=["Analytics"])
def get_seasonal_trends():
    try:
        from Backend.recommender import _CONTEXT, _ITEM_META
        import pandas as pd
        from datetime import datetime
        
        # Ensure we have data
        if _CONTEXT is None or _CONTEXT.empty:
            return []

        # Map item_idx to product_group to get string names instead of integer encoded values
        trends_df = _CONTEXT.copy()
        trends_df['product_group_str'] = trends_df['item_idx'].map(lambda x: _ITEM_META.get(x, {}).get('product_group', 'Unknown'))

        # Get top categories per month
        # Using .size() and reset_index to get counts
        trends = trends_df.groupby(['month', 'product_group_str']).size().reset_index(name='counts')
        
        data = []
        # Month names mapping
        month_map = {i: datetime(2020, i, 1).strftime('%b') for i in range(1, 13)}
        
        for month_val in range(1, 13):
            # Filter for this month
            m_data = trends[trends['month'] == month_val]
            if m_data.empty:
                continue
                
            # Get top 5 categories
            m_data = m_data.sort_values('counts', ascending=False).head(5)
            
            for _, row in m_data.iterrows():
                data.append({
                    "month": month_map.get(month_val, str(month_val)),
                    "category": str(row['product_group_str']),
                    "count": int(row['counts'])
                })
        return data
    except Exception as e:
        print(f"Analytics Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/user/{user_idx}", tags=["Analytics"])
def get_user_activity(user_idx: int):
    try:
        from Backend.recommender import _CONTEXT
        import pandas as pd
        from datetime import datetime
        
        # Filter for this user
        u_data = _CONTEXT[_CONTEXT['user_idx'] == user_idx]
        if u_data.empty:
            return []
            
        # Group by month and get counts
        activity = u_data.groupby('month').size().reset_index(name='count')
        
        data = []
        month_map = {i: datetime(2020, i, 1).strftime('%b') for i in range(1, 13)}
        
        for _, row in activity.iterrows():
            m_val = int(row['month'])
            data.append({
                "month": month_map.get(m_val, str(m_val)),
                "count": int(row['count'])
            })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
