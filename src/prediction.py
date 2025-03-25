#!/usr/bin/env python3

from src.model import load_model, evaluate_model

# Define predict function
def predict(data):
    """
    Predict the class label for the input data

    Parameters
    ----------
    data : dict
        The input data

    Returns
    -------
    dict
        The prediction
    """
    # Load the model
    model = load_model()

    # Make prediction
    prediction = model.predict(data)    

    return prediction