from locust import HttpUser, task, between

class RecommendationUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def recommend(self):
        self.client.post(
            "/recommend",
            json={
                "user_idx": 5,
                "top_k": 10,
                "month_filter": 2
            }
        )