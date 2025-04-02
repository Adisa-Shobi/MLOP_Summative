import sys
import os
sys.path.append(os.path.abspath('.'))
from src.model import predict, retrain_model, evaluate_model
from src.api.model import XDatapoint
from typing import Dict
from fastapi import APIRouter, Request, File, UploadFile, HTTPException
from src.utils import *


v1 = APIRouter(
    prefix="/v1",
    tags=["gym"]
)


# Home Route
@v1.get('/', response_model=dict)
def home(request: Request):
    """
    Root endpoint for v1 router with detailed logging

    Args
    ----
    request: Request
        The current request object
    """
    return {
        "message": "Welcome to Workout Prediction App",
        "endpoints": [
            "/v1/upload-training-data",
            "/v1/predict",
            "/v1/model-evaluation"
        ],
        "debug_info": {
            "full_path": str(request.url),
        }
    }

# Endpoint to upload training csv and retrain


@v1.post('/upload-training-data')
async def upload_training_data(
    file: UploadFile = File(...),
    base_filename: str = "training_data"
) -> Dict:
    """
    Endpoint to upload training CSV and retrain the model.

    Args:
        file: CSV file containing training data
        base_filename: Base filename for storing the uploaded file

    Returns:
        Dict with success message
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise ValueError("Only CSV files are allowed")

        # Save the file and get its path
        filename = write_uploaded_file(file, base_filename)

        # Derive schema from reference file
        schema = derive_schema_from_file(
            "data/gym_members_exercise_tracking.csv")

        # Load the uploaded training data
        data = load_training_data(filename)

        # Validate the uploaded data against the reference schema
        report = validate_training_data(
            training_data=data,
            reference_schema=schema
        )

        # If data is not valid, raise an exception with detailed errors
        if not report['is_valid']:
            # Construct a detailed error message
            error_details = "\n".join(report['errors'])
            raise ValueError(f"Data validation failed:\n{error_details}")

        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "rows": len(data),
            "columns": list(data.columns)
        }
    except ValueError as ve:
        # Validation errors (file type, schema issues)
        delete_uploaded_file(filename)
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnf:
        # Reference file not found
        delete_uploaded_file(filename)
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to make prediction based on one data point
@v1.post("/predict")
def predict_endpoint(data: XDatapoint):
    """
    Makes prediction based on datapoint

    Args:
        data: Input data point matching the XDatapoint schema

    Returns:
        JSON response with prediction result
    """
    try:
        # Call the predict function from your model
        prediction_result = predict(X=data)

        # Return the prediction result as JSON
        return {
            "status": "success",
            "prediction": prediction_result,
            "message": "Prediction completed successfully"
        }
    except Exception as e:
        # Handle errors gracefully
        return {
            "status": "error",
            "prediction": None,
            "message": f"Error during prediction: {str(e)}"
        }


@v1.post("/retrain")
async def retrain_endpoint(request: Request):
    """
    Endpoint to retrain the model with new data.

    Args:
        request: Request object containing a JSON body with 'filename' field

    Returns:
        JSON response with retraining result
    """
    try:
        # Parse the JSON body from the request
        try:
            body = await request.json()
            filename = body.get('filename', '')
        except Exception as e:
            raise HTTPException(
                status_code=400, detail="Please provide a valid filename")

        if not filename:
            raise HTTPException(
                status_code=400, detail="Filename is required")

        # Load the training data
        df = load_training_data(filename)

        # Retrain the model
        result = retrain_model(df)

        return {
            "status": "success",
            "message": "Model successfully retrained",
            "details": result
        }
        pass
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during retraining: {str(e)}"
        }

# Endpoint for fetching model evaluation
@v1.get("/evaluate")
async def evaluate_endpoint():
    """
    Endpoint for fetching model evaluation
    """
    try:
        evaluation = evaluate_model()

        return {
                "status": "success",
                "message": "Model evaluation completed successfully",
                "details": evaluation
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during evaluation: {str(e)}"
        }


# Get all training files
@v1.get('/training-data')
async def get_training_data():
    """
    Gets all the training files stored locally

    Returns
    -------
        A list of all training files
    """
    try:
        tr_files = list_training_data()

        return tr_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete specific trining files
@v1.delete("/training-data/{filename}")
async def delete_training_file(filename: str):
    """
    Deletes a specific training file
    """
    try:
        delete_uploaded_file(filename)
        return {
            "status": "success",
            "message": "Training file deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Delete all training Files
@v1.delete("/training-data")
async def delete_all_training_files():
    """
    Deletes all training files
    """
    try:
        clean_training_directory()
        return {
            "status": "success",
            "message": "All training files deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
