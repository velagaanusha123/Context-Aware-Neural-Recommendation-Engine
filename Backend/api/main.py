from fastapi import FastAPI

from Backend.api.recommend import recommend

app = FastAPI(
    title="Fashion Recommendation API"
)

@app.get("/")
def home(): 

    return {
        "message": "Fashion Recommender Running"
    }


@app.get("/recommend")
def get_recommendations(
    user_id: int,
    k: int = 10
):

    recommendations = recommend(
        user_id=user_id,
        k=k
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }