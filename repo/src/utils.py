import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Type, List, Union
from pandas.api.types import is_string_dtype
from fastapi import UploadFile

TRAINING_DATA_DIR = "training_data"


def ensure_training_dir_exists():
    """Ensure the training data directory exists."""
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)


def write_training_data(df: pd.DataFrame, filename: str) -> str:
    """
    Write DataFrame to a timestamped CSV file.

    Args:
        df (pd.DataFrame): Training data
        filename (str): Base filename

    Returns:
        str: Full path of saved file
    """
    ensure_training_dir_exists()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.csv"
    filepath = os.path.join(TRAINING_DATA_DIR, full_filename)
    df.to_csv(filepath, index=False)
    return filepath


def write_uploaded_file(file: UploadFile, base_filename: str) -> str:
    """
    Write an uploaded file to the training directory.

    Args:
        file (UploadFile): Uploaded file
        base_filename (str): Base filename

    Returns:
        str: Full path of saved file
    """
    ensure_training_dir_exists()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = os.path.splitext(file.filename)[1] or '.csv'
    full_filename = f"{base_filename}_{timestamp}{file_ext}"
    filepath = os.path.join(TRAINING_DATA_DIR, full_filename)

    with open(filepath, 'wb') as buffer:
        buffer.write(file.file.read())

    return full_filename


def load_training_data(
        filename: str
        ) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded training data

    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        ValueError: For other data loading issues
    """
    try:
        # Read the CSV file
        ensure_training_dir_exists()
        filepath = os.path.join(TRAINING_DATA_DIR, filename)
        
        df = pd.read_csv(filepath)

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("The loaded DataFrame is empty")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file is empty: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading training data: {str(e)}")


def delete_uploaded_file(filename: str) -> bool:
    """
    Delete a file from the specified path.

    This function attempts to delete a file located at the given filepath. 
    It provides robust error handling and informative feedback about the deletion process.

    Args:
        filepath (str): The full path to the file that should be deleted.

    Returns:
        bool: 
            - True if the file was successfully deleted
            - False if the file does not exist at the specified path
    """
    try:
        # Check if file exists
        ensure_training_dir_exists()
        filepath = os.path.join(TRAINING_DATA_DIR, filename)
        if not os.path.exists(filepath):
            return False

        # Delete the file
        os.remove(filepath)
        return True

    except PermissionError:
        raise PermissionError(
            f"Permission denied: Unable to delete file {filepath}")
    except OSError as e:
        raise OSError(f"Error deleting file {filepath}: {str(e)}")


def derive_schema_from_file(filepath: str) -> Dict[str, Type]:
    """
    Automatically derive a schema from a CSV file.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        Dict[str, Type]: Mapping of column names to their types
    """
    df = pd.read_csv(filepath)

    def map_dtype_to_type(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return int
        elif pd.api.types.is_float_dtype(dtype):
            return float
        elif pd.api.types.is_bool_dtype(dtype):
            return bool
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return str
        else:
            return str

    return {
        column: map_dtype_to_type(df[column].dtype)
        for column in df.columns
    }


def validate_training_data(
    training_data: pd.DataFrame,
    reference_schema: Dict[str, type]
) -> Dict[str, Any]:
    """
    Validate training data against a reference schema.

    Args:
        training_data (pd.DataFrame): DataFrame to validate
        reference_schema (Dict[str, type]): Expected columns and types

    Returns:
        Dict[str, Any]: Validation report
    """
    validation_report = {
        'is_valid': True,
        'errors': [],
        'column_checks': {}
    }

    # Check for missing columns
    missing_columns = set(reference_schema.keys()) - set(training_data.columns)
    if missing_columns:
        validation_report['is_valid'] = False
        validation_report['errors'].append(
            f"Missing columns: {missing_columns}")

    # Check for extra columns
    extra_columns = set(training_data.columns) - set(reference_schema.keys())
    if extra_columns:
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"Extra columns: {extra_columns}")

    # Validate each column
    for column, expected_type in reference_schema.items():
        if column not in training_data.columns:
            continue

        column_check = {
            'name': column,
            'expected_type': expected_type.__name__,
            'type_match': True,
            'null_check': True,
            'null_percentage': 0.0
        }

        try:
            # Handle special case for numeric types
            if expected_type in [int, float]:
                converted_column = pd.to_numeric(
                    training_data[column], errors='coerce')

                type_match = (
                    (expected_type == int and converted_column.dtype == np.int64) or
                    (expected_type == float and converted_column.dtype in [
                     np.float64, np.float32])
                )
                column_check['type_match'] = type_match

                if not type_match:
                    validation_report['is_valid'] = False
                    validation_report['errors'].append(
                        f"Type mismatch in column {column}")
            else:
                # For other types, use direct type checking
                type_match = (expected_type == str and is_string_dtype(
                    training_data[column])) or training_data[column].dtype == expected_type
                column_check['type_match'] = type_match

                if not type_match:
                    validation_report['is_valid'] = False
                    validation_report['errors'].append(
                        f"Type mismatch in column {column}")

            # Null percentage check
            null_percentage = training_data[column].isnull().mean() * 100
            column_check['null_percentage'] = null_percentage

            # Null threshold check
            if null_percentage > 20:
                column_check['null_check'] = False
                validation_report['is_valid'] = False
                validation_report['errors'].append(
                    f"High null percentage in column {column}: {null_percentage:.2f}%")

        except Exception as e:
            column_check['type_match'] = False
            validation_report['is_valid'] = False
            validation_report['errors'].append(
                f"Error processing column {column}: {str(e)}")

        # Store column check results
        validation_report['column_checks'][column] = column_check

    return validation_report


def list_training_data() -> List[Dict[str, Union[str, float, datetime]]]:
    """
    List all training data files with their storage times.

    Returns:
        List of dictionaries with filename and timestamp
    """
    ensure_training_dir_exists()

    training_files = []

    for filename in os.listdir(TRAINING_DATA_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(TRAINING_DATA_DIR, filename)
            creation_time = os.path.getctime(filepath)

            training_files.append({
                'filename': filename,
                'timestamp': creation_time,
                'datetime': datetime.fromtimestamp(creation_time)
            })

    training_files.sort(key=lambda x: x['timestamp'], reverse=True)

    return training_files


def clean_training_directory(keep_recent: int = 1):
    """
    Clean the training directory, keeping only the most recent files.

    Args:
        keep_recent (int, optional): Number of most recent files to keep
    """
    ensure_training_dir_exists()

    files = list_training_data()

    if len(files) > keep_recent:
        for file in files[keep_recent:]:
            filepath = os.path.join(TRAINING_DATA_DIR, file['filename'])
            os.remove(filepath)
