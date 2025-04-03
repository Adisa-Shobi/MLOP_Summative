from locust import HttpUser, task, between
import json
import random


class GymUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Initialize any data needed for the test"""
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "locust/2.24.1"
        }

    @task(3)
    def predict_experience_level(self):
        """Test the experience level prediction endpoint"""
        # Sample data for prediction matching the API structure
        data = {
            "age": random.randint(18, 70),
            "gender": random.choice(["Male", "Female"]),
            "weight": random.uniform(45.0, 120.0),
            "height": random.uniform(1.5, 2.0),
            "max_bpm": random.randint(120, 200),
            "avg_bpm": random.randint(100, 180),
            "resting_bpm": random.randint(50, 100),
            "session_duration": random.uniform(0.5, 3.0),
            "calories_burned": random.uniform(200.0, 1000.0),
            "workout_type": random.choice(["Cardio", "HIIT", "Yoga", "Strength"]),
            "fat_percentage": random.uniform(10.0, 40.0),
            "water_intake": random.uniform(1.0, 5.0),
            "workout_frequency": random.randint(1, 7),
            "bmi": random.uniform(18.0, 35.0)
        }

        self.client.post("/v1/predict", json=data, headers=self.headers)

    @task(2)
    def evaluate_model(self):
        """Test the model evaluation endpoint"""
        self.client.get("/v1/evaluate", headers=self.headers)

    @task(2)
    def get_training_data(self):
        """Test the training data endpoint"""
        self.client.get("/v1/training-data", headers=self.headers)

    @task(2)
    def get_visualization_data(self):
        """Test the visualization data endpoint"""
        self.client.get("/v1/visualization-data", headers=self.headers)

    @task(1)
    def home(self):
        """Test the home endpoint"""
        self.client.get("/v1/", headers=self.headers)
