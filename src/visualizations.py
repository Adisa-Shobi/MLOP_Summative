import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from src.preprocessing import standardize_df


def visualization_data(
        csv_file_path: str = 'data/gym_members_exercise_tracking.csv'
) -> Dict[str, Any]:
    """
    Process fitness data CSV and generate analytics summary.

    Args:
        csv_file_path (str): Path to the CSV file containing fitness data

    Returns:
        dict: Structured analytics data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file has missing required columns
        RuntimeError: For general processing errors
    """
    try:
        # Validate file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(
                f"CSV file not found at path: {csv_file_path}")

        # Read the CSV file and standardize column names
        try:
            df = pd.read_csv(csv_file_path)
            # Standardize column names to lowercase first part
            df = standardize_df(df)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

        # Check for empty dataframe
        if df.empty:
            return create_empty_result()

        # Map numeric experience levels to text labels
        experience_map = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}

        # Validate experience levels - using standardized column name
        invalid_levels = set(df['experience_level'].dropna(
        ).unique()) - set(experience_map.keys())
        if invalid_levels:
            # Filter to only valid levels
            df = df[df['experience_level'].isin(experience_map.keys())]

        df['experience_text'] = df['experience_level'].map(experience_map)

        # Handle potential errors in BMI calculation
        if 'bmi' not in df.columns:
            try:
                # Check for zero or negative heights - using standardized column names
                if (df['height'] <= 0).any():
                    df = df[df['height'] > 0]

                # Calculate BMI (Weight in kg / (Height in m)²) - using standardized column names
                df['bmi'] = df['weight'] / (df['height'] ** 2)
            except Exception:
                df['bmi'] = np.nan

        # Total number of members
        total_members = len(df)

        # Calculate averages with error handling - using standardized column names
        try:
            avg_age = round(df['age'].mean(), 1)
        except Exception:
            avg_age = 0

        try:
            avg_bmi = round(df['bmi'].mean(), 1)
        except Exception:
            avg_bmi = 0

        # Get distribution by experience level with error handling
        try:
            exp_counts = df['experience_text'].value_counts()
            beginner_count = int(exp_counts.get('Beginner', 0))
            intermediate_count = int(exp_counts.get('Intermediate', 0))
            advanced_count = int(exp_counts.get('Advanced', 0))
        except Exception:
            beginner_count = intermediate_count = advanced_count = 0

        # Calculate age by experience level with error handling
        try:
            age_by_exp = df.groupby('experience_text')[
                'age'].mean().round().astype(int).to_dict()
        except Exception:
            age_by_exp = {'Beginner': 0, 'Intermediate': 0, 'Advanced': 0}

        # Calculate duration and calories by experience level with error handling
        try:
            duration_calories_by_exp = df.groupby('experience_text').agg({
                'session_duration': 'mean',
                'calories_burned': 'mean'
            }).round(2)
        except Exception:
            duration_calories_by_exp = pd.DataFrame({
                'session_duration': [0, 0, 0],
                'calories_burned': [0, 0, 0]
            }, index=['Beginner', 'Intermediate', 'Advanced'])

        # Workout preferences by experience level with error handling
        workout_prefs = calculate_workout_preferences(df, experience_map)

        # Body composition by experience level with error handling
        try:
            body_comp = df.groupby('experience_text').agg({
                'bmi': 'mean',
                'fat_percentage': 'mean'
            }).round(1)
        except Exception:
            body_comp = pd.DataFrame({
                'bmi': [0, 0, 0],
                'fat_percentage': [0, 0, 0]
            }, index=['Beginner', 'Intermediate', 'Advanced'])

        # Workout frequency data with error handling
        try:
            freq_data = df.groupby('workout_frequency').agg({
                'fat_percentage': 'mean',
                'experience_level': 'count'
            })
            freq_data = freq_data.rename(columns={'experience_level': 'count'})
            freq_data['fat_percentage'] = freq_data['fat_percentage'].round(1)
        except Exception:
            freq_data = pd.DataFrame({
                'fat_percentage': [0],
                'count': [0]
            }, index=[0])

        # Construct the response structure
        result = {
            "summary": {
                "totalMembers": total_members,
                "averageAge": avg_age,
                "averageBMI": avg_bmi,
                "fitnessLevelDistribution": {
                    "beginner": beginner_count,
                    "intermediate": intermediate_count,
                    "advanced": advanced_count
                }
            },
            "experienceLevelData": {
                "ageByExperience": [
                    {"level": "Beginner", "avgAge": age_by_exp.get(
                        'Beginner', 0), "count": beginner_count},
                    {"level": "Intermediate", "avgAge": age_by_exp.get(
                        'Intermediate', 0), "count": intermediate_count},
                    {"level": "Advanced", "avgAge": age_by_exp.get(
                        'Advanced', 0), "count": advanced_count}
                ],
                "durationByExperience": get_duration_data(duration_calories_by_exp, beginner_count, intermediate_count, advanced_count),
                "workoutPreferencesByExperience": workout_prefs,
                "bodyCompositionByExperience": get_body_comp_data(body_comp),
                "workoutFrequencyData": get_frequency_data(freq_data, df)
            }
        }

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to generate visualization data: {str(e)}")


def get_duration_data(duration_data, beginner_count, intermediate_count, advanced_count):
    """Helper function to safely extract duration data"""
    try:
        result = []
        for level, count in [("Beginner", beginner_count), ("Intermediate", intermediate_count), ("Advanced", advanced_count)]:
            avg_duration = 0.0
            avg_calories = 0

            # Safe access to DataFrame values
            if level in duration_data.index:
                if 'session_duration' in duration_data.columns:
                    avg_duration = float(
                        duration_data.loc[level, 'session_duration'])
                if 'calories_burned' in duration_data.columns:
                    avg_calories = int(
                        duration_data.loc[level, 'calories_burned'])

            result.append({
                "level": level,
                "avgDuration": avg_duration,
                "avgCalories": avg_calories,
                "count": count
            })
        return result
    except Exception as e:
        print(f"Error in get_duration_data: {str(e)}")
        return [
            {"level": "Beginner", "avgDuration": 0.0,
                "avgCalories": 0, "count": beginner_count},
            {"level": "Intermediate", "avgDuration": 0.0,
                "avgCalories": 0, "count": intermediate_count},
            {"level": "Advanced", "avgDuration": 0.0,
                "avgCalories": 0, "count": advanced_count}
        ]


def calculate_workout_preferences(df, experience_map):
    """
    Calculate workout preferences by experience level with improved error handling.
    Returns a dictionary of workout preferences percentages.
    """
    workout_prefs = {
        'beginner': {'cardio': 0, 'strength': 0, 'yoga': 0, 'hiit': 0},
        'intermediate': {'cardio': 0, 'strength': 0, 'yoga': 0, 'hiit': 0},
        'advanced': {'cardio': 0, 'strength': 0, 'yoga': 0, 'hiit': 0}
    }

    # Debug: Check if workout_type column exists
    if 'workout_type' not in df.columns:
        print("Error: 'workout_type' column is missing from the DataFrame")
        return workout_prefs

    # Debug: Display unique workout types
    print("Unique workout types:", df['workout_type'].unique())

    try:
        # Normalize workout types to lowercase for consistent comparison
        df['workout_type_lower'] = df['workout_type'].str.lower()

        # Expected workout types (lowercase)
        workout_types = ['cardio', 'strength', 'yoga', 'hiit']

        for level, level_text in experience_map.items():
            # Filter to just this experience level
            level_df = df[df['experience_level'] == level]

            # Debug: Show how many records for this level
            print(f"Level {level_text} ({level}): {len(level_df)} records")

            if not level_df.empty:
                total = len(level_df)
                # Get counts for each workout type (case insensitive)
                workout_counts = level_df['workout_type_lower'].value_counts()

                # Debug: Show the workout counts
                print(f"Workout counts for {level_text}:",
                      workout_counts.to_dict())

                # Calculate percentages with safety checks
                level_prefs = {}
                for workout in workout_types:
                    count = workout_counts.get(workout, 0)
                    # Use max(total, 1) to prevent division by zero
                    percentage = int(round(count / max(total, 1) * 100))
                    level_prefs[workout] = percentage

                # Save the calculated preferences
                workout_prefs[level_text.lower()] = level_prefs

    except Exception as e:
        print(f"Error calculating workout preferences: {str(e)}")
        # The default empty workout_prefs will be returned

    return workout_prefs


def get_body_comp_data(body_comp):
    """Helper function to safely extract body composition data"""
    try:
        result = []
        for level in ["Beginner", "Intermediate", "Advanced"]:
            avg_bmi = 0.0
            avg_body_fat = 0.0

            # Safe access to DataFrame values
            if level in body_comp.index:
                if 'bmi' in body_comp.columns:
                    avg_bmi = float(body_comp.loc[level, 'bmi'])
                if 'fat_percentage' in body_comp.columns:
                    avg_body_fat = float(
                        body_comp.loc[level, 'fat_percentage'])

            result.append({
                "level": level,
                "avgBMI": avg_bmi,
                "avgBodyFat": avg_body_fat
            })
        return result
    except Exception as e:
        print(f"Error in get_body_comp_data: {str(e)}")
        return [
            {"level": "Beginner", "avgBMI": 0.0, "avgBodyFat": 0.0},
            {"level": "Intermediate", "avgBMI": 0.0, "avgBodyFat": 0.0},
            {"level": "Advanced", "avgBMI": 0.0, "avgBodyFat": 0.0}
        ]


def get_frequency_data(freq_data, df):
    """Helper function to safely extract frequency data"""
    try:
        result = []
        for freq in sorted(df['workout_frequency'].unique()):
            avg_fat_percentage = 0.0
            count = 0

            # Safe access to DataFrame values
            if freq in freq_data.index:
                if 'fat_percentage' in freq_data.columns:
                    avg_fat_percentage = float(
                        freq_data.loc[freq, 'fat_percentage'])
                if 'count' in freq_data.columns:
                    count = int(freq_data.loc[freq, 'count'])

            result.append({
                "frequency": int(freq),
                "avgFatPercentage": avg_fat_percentage,
                "count": count
            })
        return result
    except Exception as e:
        print(f"Error in get_frequency_data: {str(e)}")
        return [{"frequency": 0, "avgFatPercentage": 0.0, "count": 0}]


def create_empty_result():
    """Create an empty result structure when data is missing"""
    return {
        "summary": {
            "totalMembers": 0,
            "averageAge": 0.0,
            "averageBMI": 0.0,
            "fitnessLevelDistribution": {
                "beginner": 0,
                "intermediate": 0,
                "advanced": 0
            }
        },
        "experienceLevelData": {
            "ageByExperience": [
                {"level": "Beginner", "avgAge": 0, "count": 0},
                {"level": "Intermediate", "avgAge": 0, "count": 0},
                {"level": "Advanced", "avgAge": 0, "count": 0}
            ],
            "durationByExperience": [
                {"level": "Beginner", "avgDuration": 0.0,
                    "avgCalories": 0, "count": 0},
                {"level": "Intermediate", "avgDuration": 0.0,
                    "avgCalories": 0, "count": 0},
                {"level": "Advanced", "avgDuration": 0.0,
                    "avgCalories": 0, "count": 0}
            ],
            "workoutPreferencesByExperience": {
                "beginner": {"cardio": 0, "strength": 0, "yoga": 0, "hiit": 0},
                "intermediate": {"cardio": 0, "strength": 0, "yoga": 0, "hiit": 0},
                "advanced": {"cardio": 0, "strength": 0, "yoga": 0, "hiit": 0}
            },
            "bodyCompositionByExperience": [
                {"level": "Beginner", "avgBMI": 0.0, "avgBodyFat": 0.0},
                {"level": "Intermediate", "avgBMI": 0.0, "avgBodyFat": 0.0},
                {"level": "Advanced", "avgBMI": 0.0, "avgBodyFat": 0.0}
            ],
            "workoutFrequencyData": []
        }
    }
