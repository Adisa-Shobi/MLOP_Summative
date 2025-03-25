#!/usr/bin/env python3

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
import os


def preprocess_data(df: pd.DataFrame, target: str) -> pd.DataFrame:

    # Drop missing values
    df_no_missing = handle_missing_values(df)

    # One hot encoding for categorical variables
    df_encoded = one_hot_encode(df_no_missing)

    # Handle class imbalance
    df_balanced = handle_imbalance_on_target(df_encoded, target)

    # Feature scaling
    df_scaled = handle_scaling(df_balanced, target)

    # Return data
    return df_scaled


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Args
    ----
    df : pd.DataFrame
        The input dataframe

    Returns
    -------
    pd.DataFrame
        The dataframe with missing values handled
    """
    from sklearn.impute import KNNImputer
    import numpy as np

    # Assert there are missing values in the dataframe
    # Check if there are any missing values
    if not df.isna().any().any():
        print("No missing values found, returning original dataframe")
        return df

    # Create a copy of the original dataframe
    df_imputed = df.copy()

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Handle numeric columns with KNN imputation
    if numeric_cols:
        # Initialize the KNN imputer
        imputer = KNNImputer(n_neighbors=5, weights='uniform')

        # Fit and transform the numeric columns
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Handle categorical columns (using most frequent value)
    for col in categorical_cols:
        if df[col].isna().any():
            most_frequent = df[col].mode()[0]
            df_imputed[col].fillna(most_frequent, inplace=True)

    return df_imputed


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One hot encode the categorical variables in the dataframe

    Args
    ----
    df : pd.DataFrame
        The input dataframe

    Returns
    -------
    pd.DataFrame
        The dataframe with categorical variables one hot encoded
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()

    # Identify categorical columns (object and category dtypes)
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")

    if not categorical_cols:
        print("No categorical columns found for one-hot encoding")
        return df_encoded

    # Apply one-hot encoding using pandas get_dummies
    for col in categorical_cols:
        # Create dummies and add prefix to avoid column name conflicts
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)

        # Join the encoded variables to the dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)

        # Drop the original categorical column
        df_encoded = df_encoded.drop(col, axis=1)

    return df_encoded


def handle_imbalance_on_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Handle class imbalance in the dataset

    Args
    ----
    df : pd.DataFrame
        The input dataframe

    Returns
    -------
    pd.DataFrame
        The dataframe with class imbalance handled
    """
    # Calculate class distribution
    class_counts = df[target].value_counts()
    sorted_counts = class_counts.sort_values(ascending=False)

    # Determine maximum imbalance
    if len(sorted_counts) > 1:
        max_class = sorted_counts.index[0]
        count_max = sorted_counts.iloc[0]
        count_second = sorted_counts.iloc[1]
        imbalance_percentage = (
            (count_max - count_second) / count_second) * 100
        print(
            f"The max imbalance occurs on class {max_class} at {imbalance_percentage:.2f}% greater than the next greatest class")
    else:
        print("Only one class present, no imbalance to handle")

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine resampled features and target back into a dataframe
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.Series(y_resampled, name=target)], axis=1)

    return df_resampled


def handle_scaling(
        df: pd.DataFrame,
        target: str
) -> pd.DataFrame:
    """
    Handle feature scaling in the dataset, excluding the target variable

    Args
    ----
    df : pd.DataFrame
        The input dataframe
    target : str
        The name of the target column to exclude from scaling

    Returns
    -------
    pd.DataFrame
        The dataframe with feature scaling applied to all numeric features except the target
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Extract target column before scaling
    if target in df.columns:
        target_series = df[target].copy()
        df_without_target = df.drop(columns=[target])
    else:
        # If target is not in df, continue without removing
        target_series = None
        df_without_target = df.copy()

    # Identify numerical columns (float64 and int64) to scale
    numerical_cols = df_without_target.select_dtypes(
        include=['float64', 'int64']).columns

    # Proceed only if there are numerical columns to scale
    if not numerical_cols.empty:
        # Check if the pickled scaler file exists
        if os.path.exists('../models/scaler.pkl'):
            # Load the existing scaler from the pickle file
            with open('../models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            # Use the loaded scaler to transform the numerical columns
            scaled_data = scaler.transform(df_without_target[numerical_cols])

            # Clip scaled data
            # Implementing clipping between -5 and 5
            scaled_data = np.clip(scaled_data, -5, 5)
        else:
            # Create a new StandardScaler instance
            scaler = StandardScaler()
            # Fit the scaler to the numerical columns and transform them
            scaled_data = scaler.fit_transform(
                df_without_target[numerical_cols])

            # Clip scaled data
            # Implementing clipping between -5 and 5
            scaled_data = np.clip(scaled_data, -5, 5)

            # Save the new scaler to a pickle file for future use
            with open('../models/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        # Convert the scaled data back to a DataFrame with the original index and column names
        scaled_df = pd.DataFrame(
            scaled_data, index=df_without_target.index, columns=numerical_cols)

        # Update the dataframe with the scaled values
        df_without_target[numerical_cols] = scaled_df

    # Recombine the result
    if target_series is not None:
        # Add the target column back to the dataframe
        result_df = df_without_target.copy()
        result_df[target] = target_series
    else:
        result_df = df_without_target

    # Return the DataFrame with scaling applied (excluding target)
    return result_df


def test_val_train_split(df: pd.DataFrame, target: str) -> tuple:
    """
    Split the data into training and validation sets

    Args
    ----
    df : pd.DataFrame
        The input dataframe
    target : str
        The target variable

    Returns
    -------
    tuple
        A tuple containing the training and validation sets
    """
    X = df.drop(columns=[target])
    y = df[target]
    y = offset_target_column(y)

    from sklearn.model_selection import train_test_split

    X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain, test_size=0.15)

    return X_train, y_train, X_test, y_test, X_val, y_val


def offset_target_column(y: pd.Series) -> pd.Series:
    """
    Offset target column values to start from 0 instead of 1 for use with 
    sparse categorical crossentropy.
    
    Parameters
    ----------
    y : pd.Series
        Target column with values starting from 1 (e.g., 1, 2, 3)
        
    Returns
    -------
    pd.Series
        Target column with values offset to start from 0 (e.g., 0, 1, 2)
    """
    # Create a copy to avoid modifying the original
    y_offset = y.copy()
    
    # Get the minimum value
    min_val = y.min()
    
    # Offset by the minimum value to make it zero-indexed
    y_offset = y - min_val
    
    # Log the transformation
    print(f"Target values transformed from {y.min()}-{y.max()} to {y_offset.min()}-{y_offset.max()}")
    
    # Create a mapping dictionary for reference
    value_mapping = {orig: new for orig, new in zip(sorted(y.unique()), sorted(y_offset.unique()))}
    print(f"Value mapping: {value_mapping}")
    
    return y_offset
