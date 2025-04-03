#!/usr/bin/env python3
from src.api.model import XDatapoint
from src.preprocessing import *
from src.model import load_model
from pathlib import Path

# Make prediction
def predict(
        X: XDatapoint,
        model_path: str = "./models/gym_model.keras"
) -> float:
    """
    Makes a predication for one datapoint

    Args
    ----
    model_path: str
        The path to keras model that need to be loaded
    X: XDatapoint
        Data object to make prediction based on
    """
    # # Validate input type
    # if not isinstance(X, XDatapoint):
    #     raise TypeError(f"Expected XDatapoint, got {type(X).__name__}")

    try:
        # Convert to DataFrame
        X_df = pd.DataFrame([X.model_dump()])
        df = pd.read_csv('./data/test/gym_members_exercise_tracking_test.csv')
        categorical_map = build_categorical_mapping(df)

        # Data preparation steps
        try:
            X_df = handle_missing_values(X_df)
            X_df = one_hot_encode(X_df)
            X_df = standardize_df(X_df)
            X_df = add_one_hot_encoded_columns(X_df, categorical_map)
            X_df = handle_scaling(X_df, '')
        except Exception as prep_error:
            raise ValueError(f"Data preparation error: {str(prep_error)}")

        print("\nProcessed DataFrame:")
        print(X_df)

        # Load model with additional error handling
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {model_path}")

            model = load_model(model_path)
        except (FileNotFoundError, ImportError) as model_load_error:
            raise FileNotFoundError(
                f"Model loading error: {str(model_load_error)}")

        # Make prediction with error handling
        # try:
        # Ensure input is in the correct format for Keras
        y = model.predict(X_df)

        return get_predicted_class(y)

        # except Exception as pred_error:
        #     raise ValueError(f"Prediction error: {str(pred_error)}")

    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()

        # Reraise with a more informative message
        raise RuntimeError(f"Prediction failed: {str(e)}") from e


# Get Predicted class
def get_predicted_class(y):
    if isinstance(y, np.ndarray):
        # Get the predicted class (index with highest probability)
        predicted_class = int(np.argmax(y[0] if y.ndim > 1 else y))

        # Convert probabilities to Python list of floats
        probs = [float(p) for p in (y[0] if y.ndim > 1 else y)]

        # Return a simple dictionary
        return {
            "class": predicted_class,
            "probabilities": probs
        }
    return y  # For non-numpy inputs
