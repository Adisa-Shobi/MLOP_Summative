#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os

# Define Model


def define_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimizer: str = "adam",
    regularizer: str = "l2",
    learning_rate: float = 0.001,
) -> tf.keras.models.Sequential:
    """
    Define a neural network model for classification and compile it.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features used to determine the input shape of the model.
    y_train : pd.Series
        Training labels used to determine the number of output classes.
    optimizer : str, optional
        Optimizer to use for training, either "adam" or "sgd". Defaults to "adam".
    regularizer : str, optional
        Regularizer to apply to the dense layers, either "l1" or "l2". Defaults to "l2".
    learning_rate : float, optional
        Learning rate for the optimizer. Defaults to 0.001.

    Returns
    -------
    tf.keras.models.Sequential
        The compiled neural network model.
    """
    # Determine input shape from training features and number of classes from labels
    input_shape = X_train.shape[1]
    num_classes = y_train.nunique()

    # Configure the regularizer based on the provided parameter
    if regularizer == "l2":
        reg = regularizers.l2(0.01)
    elif regularizer == "l1":
        reg = regularizers.l1(0.01)
    else:
        reg = None  # No regularization if an invalid option is provided

    # Define the model architecture
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(
            input_shape,), kernel_regularizer=reg),
        layers.Dense(32, activation='relu', kernel_regularizer=reg),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Configure the optimizer based on the provided parameter
    if optimizer == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate)
    else:
        # Default to Adam if invalid
        opt = optimizers.Adam(learning_rate=learning_rate)

    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Train Model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int = 100,
    stopping_patience: int = 10,
) -> tf.keras.Model:
    """
    Train the model

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation labels.
    epochs : int, optional
        Number of epochs to train. Defaults to 100.
    stopping_patience : int, optional
        Patience for early stopping. Defaults to 10.

    Returns
    -------
    tf.keras.Model
        The trained model
    """
    # Define and compile the model with default hyperparameters
    model = define_model(X_train, y_train, optimizer='adam',
                         regularizer='l2', learning_rate=0.001)

    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=stopping_patience, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    return model

# Retrain Model


def retrain_model(
        model: tf.keras.Model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        epochs: int = 100,
        stopping_patience: int = 10,
) -> tf.keras.Model:
    """
    Retrain the model using transfer learning

    Parameters
    ----------
    model : tf.keras.Model
        The pre-trained model to be fine-tuned
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target values
    epochs : int, optional
        Maximum number of training epochs, default is 100
    stopping_patience : int, optional
        Number of epochs with no improvement after which training will stop, default is 10

    Returns
    -------
    tf.keras.Model
        The retrained model
    """
    import tensorflow as tf
    import numpy as np

    # Create early stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    # Create model checkpoint to save the best model during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_retrained_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Compile the model
    # Note: We're recompiling but keeping all weights/parameters from the original model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # Adjust based on your task (regression/classification)
        loss='mean_squared_error',
        metrics=['mae']  # Add appropriate metrics for your task
    )

    # Train the model with explicit validation data
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),  # Using provided validation data
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        batch_size=32  # Adjust batch size as needed
    )

    print(f"Model retrained for {len(history.history['loss'])} epochs")

    return model

# Evaluate Model


def evaluate_model():
    """
    Evaluate the model

    Returns
    -------
    dict
        The evaluation metrics
    """
    # TODO: Implement model evaluation
    pass

# Save model


def save_model(
        model: tf.keras.Model,
        model_dir: str = "../models",
):
    """
    Save the model

    Parameters
    ----------
    model : tf.keras.Model
        The model to save
    model_dir : str, optional
        The directory to save the model in, by default "../models"

    Returns
    -------
    str
        The path to the saved model
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the model file name
    model_name = "gym_model.keras"

    # Construct the full path to save the model
    model_path = os.path.join(model_dir, model_name)

    # Save the model to the specified path
    model.save(model_path)

    return model_path

# Load Model


def load_model(
    model_path: str = "../models/gym_model.h5",
) -> tf.keras.Model:
    """
    Load the model

    Parameters
    ----------
    model_path : str
        Path to the saved model file, default is "../models/gym_model.h5"

    Returns
    -------
    tf.keras.Model
        The loaded model
    """
    import tensorflow as tf

    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
