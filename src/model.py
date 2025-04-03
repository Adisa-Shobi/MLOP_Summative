#!/usr/bin/env python3

import tensorflow as tf
import keras
from tensorflow.keras import models, layers, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from .preprocessing import *
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

    save_model(model=model)

    return model


# Retrain Model
def retrain_model(
        data: pd.DataFrame,
        epochs: int = 100,
        stopping_patience: int = 10,
        target: str = 'experience_level'
) -> tf.keras.Model:
    """
    Retrain the model using transfer learning

    Parameters
    ----------
    df: DataFrame
        Training data
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
    from sklearn.model_selection import train_test_split

    df = preprocess_data(data, target)

    X = df.drop(columns=[target])
    y = df[target]
    y = offset_target_column(y)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.15)

    model = load_model()

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
        X,
        y,
        epochs=epochs,  # Using provided validation data
        callbacks=[early_stopping, checkpoint],
        verbose=1,
        batch_size=32  # Adjust batch size as needed
    )

    print(f"Model retrained for {len(history.history['loss'])} epochs")

    save_model(model=model)

    return evaluate_model(model, X_test, y_test)


def evaluate_model(
        model=None,
        X_test=None,
        y_test=None
):
    """
    Evaluate the model

    Parameters
    ----------
    model : keras.Model, optional
        The model to evaluate (will load from disk if not provided)
    X_test : pd.DataFrame
        The test features
    y_test : pd.Series or np.ndarray
        The test labels

    Returns
    -------
    dict
        The evaluation metrics including confusion matrix
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
    try:
        if model is None:
            model = load_model()

        if X_test is None or y_test is None:
            test_data = pd.read_csv(
                './data/test/gym_members_exercise_tracking_test.csv')
            df = preprocess_data(test_data, 'experience_level')
            X_test = df.drop(columns=['experience_level'])
            y_test = offset_target_column(df['experience_level'])

        # Make predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Ensure y_test is in the right format
        if isinstance(y_test, pd.DataFrame) or (isinstance(y_test, np.ndarray) and y_test.ndim > 1):
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test

        # Get metrics from model.evaluate()
        evaluation = model.evaluate(X_test, y_test, verbose=0)
        # Handle different numbers of returned metrics
        if isinstance(evaluation, list) and len(evaluation) >= 4:
            test_loss, test_accuracy, test_precision, test_recall = evaluation
        elif isinstance(evaluation, list) and len(evaluation) == 2:
            test_loss, test_accuracy = evaluation
            # Calculate precision and recall manually
            y_true_int = y_true.astype(int)
            y_pred_int = y_pred.astype(int)
            test_precision = precision_score(
                y_true_int, y_pred_int, average='weighted')
            test_recall = recall_score(
                y_true_int, y_pred_int, average='weighted')
        else:
            # Handle case with single value or other unusual return
            test_loss = evaluation if not isinstance(
                evaluation, list) else evaluation[0]
            test_accuracy = None
            test_precision = None
            test_recall = None

        # Calculate F1 score and classification report
        y_true_int = y_true.astype(int)
        y_pred_int = y_pred.astype(int)
        f1 = f1_score(y_true_int, y_pred_int, average='weighted')

        # Dynamically determine class labels based on unique values
        unique_classes = np.unique(y_true_int)
        class_labels = [f"Class {i}" for i in unique_classes]

        # Generate confusion matrix
        cm = confusion_matrix(y_true_int, y_pred_int)

        report = classification_report(
            y_true_int, y_pred_int, labels=unique_classes, output_dict=True)

        return {
            'loss': float(test_loss) if test_loss is not None else None,
            'accuracy': float(test_accuracy) if test_accuracy is not None else None,
            'precision': float(test_precision) if test_precision is not None else None,
            'recall': float(test_recall) if test_recall is not None else None,
            'f1_score': float(f1),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}")
    except Exception as e:
        raise Exception(f"Error during evaluation: {str(e)}")


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
    MODEL_PATH = "./models/gym_model.keras"

    # Construct the full path to save the model
    model_path = os.path.join(model_dir, model_name)

    # Save the model to the specified path
    model.save(model_path)

    return model_path


# Load Model
def load_model(
    model_path: str = "./models/gym_model.keras",
) -> keras.Model:
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
    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

