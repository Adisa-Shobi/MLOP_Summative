#!/usr/bin/env python3

import pandas as pd
import os
import sys


def split_data():
    """Split a CSV dataset into training and testing sets based on user input."""
    from sklearn.model_selection import train_test_split
    try:
        # Prompt user for input
        input_file = input("Enter the path to your CSV file: ")
        train_dir = input("Enter the directory for training data: ")
        test_dir = input("Enter the directory for testing data: ")

        # Prompt for split ratio with validation
        while True:
            try:
                split_ratio = float(input("Enter the split ratio (0-1): "))
                if 0 < split_ratio < 1:
                    break
                else:
                    print("Split ratio must be between 0 and 1. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load CSV data
        df = pd.read_csv(input_file)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Create directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Split the data
        train_df, test_df = train_test_split(
            df, train_size=split_ratio, random_state=42)

        # Generate output filenames
        base_filename = os.path.basename(input_file)
        name = os.path.splitext(base_filename)[0]

        # Save the files
        train_path = os.path.join(train_dir, f"{name}_train.csv")
        test_path = os.path.join(test_dir, f"{name}_test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train set: {train_df.shape[0]} rows saved to {train_path}")
        print(f"Test set: {test_df.shape[0]} rows saved to {test_path}")

        return True

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return False


if __name__ == "__main__":
    split_data()
