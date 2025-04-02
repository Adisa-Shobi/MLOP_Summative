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
   
    # Standardizes dataframe
    df_standardized = standardize_df(df_encoded)

    # Handle class imbalance
    df_balanced = handle_imbalance_on_target(df_standardized, target)

    # Feature scaling
    df_scaled = handle_scaling(df_balanced, target)

    # Return data
    return df_scaled


def standardize_df(df):
    """
    Splits column names and retains the first name in lowercase

    Args
    ----
    df: Dataframe
        The data to be standardized

    Return
    ------
        Returns standardized dataframe
    """
    # Create a copy of the dataframe to avoid modifying the original
    standardized_df = df.copy()

    # Get the column names
    columns = standardized_df.columns

    # Create a dictionary to map old column names to new ones
    column_mapping = {}

    # Process each column name
    for col in columns:
        # Split by space only
        split_col = col.split(' ')

        # Take the first part and convert to lowercase
        new_col = split_col[0].lower()

        # Add to mapping dictionary
        column_mapping[col] = new_col

    # Rename the columns
    standardized_df = standardized_df.rename(columns=column_mapping)

    return standardized_df


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
        if os.path.exists('./models/scaler.pkl'):
            # Load the existing scaler from the pickle file
            with open('./models/scaler.pkl', 'rb') as f:
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
    print(
        f"Target values transformed from {y.min()}-{y.max()} to {y_offset.min()}-{y_offset.max()}")

    # Create a mapping dictionary for reference
    value_mapping = {orig: new for orig, new in zip(
        sorted(y.unique()), sorted(y_offset.unique()))}
    print(f"Value mapping: {value_mapping}")

    return y_offset

def build_categorical_mapping(df):
    """
    Builds a dictionary mapping of categorical (string) columns to their unique values.
    
    Args:
        df: pandas DataFrame
            The dataset to analyze for categorical columns
    
    Returns:
        dict: A mapping of categorical column names to lists of their unique values
    """
    # Initialize an empty dictionary to store our mappings
    categorical_mapping = {}
    
    # Iterate through each column in the dataframe
    for column in df.columns:
        # Check if this column contains string data
        # We'll consider a column categorical if it has string (object) dtype
        # or if it's a categorical dtype
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            # Get all unique values for this column and convert to a list
            unique_values = df[column].unique().tolist()
            
            # Filter out any non-string values (like NaN) just to be safe
            unique_values = [value for value in unique_values 
                             if isinstance(value, str)]
            
            # Add this column and its unique values to our mapping
            if unique_values:  # Only add if there are actual string values
                categorical_mapping[column] = unique_values
    
    return categorical_mapping


def add_one_hot_encoded_columns(
    X,
    categorical_columns={'gender': ['male', 'female'],
                         'workout_type': ['cardio', 'strength', 'hiit', 'yoga']}
):
    """
    Adds missing one-hot encoded columns with zero values to a single data point.
    Only adds columns that don't already exist.

    Args:
        X: dict or pandas Series
            The single data point to be processed
        categorical_columns: dict
            Dictionary mapping column names to their possible values
            Example: {'gender': ['male', 'female'], 'workout_type': ['cardio', 'strength', 'hiit']}

    Returns:
        dict or pandas Series with any missing one-hot encoded columns added and filled with zeros
    """
    import pandas as pd

    # Convert to pandas Series if it's a dictionary
    is_dict = isinstance(X, dict)
    if is_dict:
        X = pd.Series(X)

    # Make a copy to avoid modifying the original
    X_processed = X.copy()

    # For each categorical column and its possible values
    for column, values in categorical_columns.items():
        for value in values:
            # Create the column name using the pattern column_value (in lowercase)
            column_name = f"{column.lower()}_{value.lower()}"
            
            # If the column doesn't exist, add it with a zero value
            if column_name not in X_processed:
                X_processed[column_name] = False

    # Return in the same format as the input
    if is_dict:
        return X_processed.to_dict()
    return X_processed
