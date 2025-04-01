from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class XDatapoint(BaseModel):
    """
    Pydantic model representing a fitness and health datapoint in a FastAPI application.

    Attributes:
    - age: Age of the individual
    - gender: Gender of the individual
    - weight: Weight in kilograms
    - height: Height in meters
    - max_bpm: Maximum heart rate
    - avg_bpm: Average heart rate
    - resting_bpm: Resting heart rate
    - session_duration: Workout session duration in hours
    - calories_burned: Calories burned during workout
    - workout_type: Type of workout performed
    - fat_percentage: Body fat percentage
    - water_intake: Water intake in liters
    - workout_frequency: Workout frequency in days per week
    - bmi: Body Mass Index
    """
    # Personal Details
    age: int = Field(..., gt=0, le=120, description="Age of the individual")
    gender: Literal['Male', 'Female',
                    ] = Field(..., description="Gender of the individual")

    # Physical Measurements
    weight: float = Field(..., gt=0, description="Weight in kilograms")
    height: float = Field(..., gt=0, description="Height in meters")

    # Heart Rate Metrics
    max_bpm: int = Field(..., gt=0, description="Maximum heart rate")
    avg_bpm: int = Field(..., gt=0, description="Average heart rate")
    resting_bpm: int = Field(..., gt=0, description="Resting heart rate")

    # Workout Details
    session_duration: float = Field(..., ge=0,
                                    description="Workout session duration in hours")
    calories_burned: float = Field(..., ge=0,
                                   description="Calories burned during workout")
    workout_type: Literal['cardio', 'strength', 'hiit', 'yoga', 'Cardio', 'Strength', 'HIIT', 'Yoga'] = Field(
        ...,
        description="Type of workout performed",

    )

    # Body Composition
    fat_percentage: float = Field(..., ge=0, le=100,
                                  description="Body fat percentage")

    # Hydration and Frequency
    water_intake: float = Field(..., ge=0,
                                description="Water intake in liters")
    workout_frequency: int = Field(..., ge=0, le=7,
                                   description="Workout frequency in days per week")

    # Health Metric
    bmi: float = Field(..., gt=0, description="Body Mass Index")

    class Config:
        # Enable JSON Schema extra validation
        extra = 'forbid'  # Prevents additional fields not defined in the model
        # Example of how to customize JSON encoding
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Example usage and validation


def validate_datapoint(data: dict) -> XDatapoint:
    """
    Helper function to validate a dictionary against the XDatapoint model

    Args:
        data (dict): Dictionary of datapoint information

    Returns:
        XDatapoint: Validated Pydantic model instance
    """
    return XDatapoint(**data)


# Example of a valid datapoint
example_datapoint = {
    "age": 30,
    "gender": "Male",
    "weight": 75.5,
    "height": 1.75,
    "max_bpm": 185,
    "avg_bpm": 145,
    "resting_bpm": 65,
    "session_duration": 1.5,
    "calories_burned": 450,
    "workout_type": "HIIT",
    "fat_percentage": 15.5,
    "water_intake": 2.5,
    "workout_frequency": 4,
    "bmi": 24.6
}

# Demonstrate validation
validated_datapoint = validate_datapoint(example_datapoint)
