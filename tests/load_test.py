from locust import HttpUser, task, between

class RecommendationUser(HttpUser):

    wait_time = between(1, 3)

    @task
    def get_recommendations(self):

        self.client.get(
            "/recommend?user_id=222851&k=10"
        )