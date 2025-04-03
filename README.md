# Gym Member Experience Level Classification

A machine learning project that predicts gym members' experience levels based on their workout data and physical characteristics.

## Web

[Web Repo Link](https://github.com/Adisa-Shobi/MLOP_web)

[Web Live Link](https://predict-visualize-train.onrender.com/)

## Project Description

This project uses machine learning to classify gym members into different experience levels based on their workout data and physical characteristics. The model is trained on a dataset of gym members and can predict whether a member is a beginner, intermediate, or advanced based on various metrics such as age, gender, weight, height, heart rate data, workout type, and more.

The project includes:
- Data preprocessing and feature engineering
- Model training and evaluation
- REST API for predictions
- Load testing with Locust

## Dataset

The dataset used for this project is the [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data) from Kaggle. It contains information about gym members including:

- Age
- Gender
- Weight (kg)
- Height (m)
- Heart rate data (Max, Average, Resting)
- Session duration
- Calories burned
- Workout type
- Fat percentage
- Water intake
- Workout frequency
- Experience level (target variable)

## Load Testing with Locust

1. Start the Locust service:
   ```bash
   docker-compose up locust
   ```

2. Access the Locust web interface at http://localhost:8089

3. Configure the test:
   - Number of users
   - Spawn rate
   - Host (default: https://mlop-summative-wh2a.onrender.com)

4. Click "Start swarming" to begin the load test

### Locust Test Results
![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfG9UduOHshI1QWsiFBCmE68CajijCfI8cFQRH5OI5Dq_24a4FWE513BN-KoDK540ASw-2tjRi-9ste7pmJDHcdNq6b9-IgJTta2xYm5RjGPPrgYtFz_0-kF97OhGFZN2gAhjK7?key=HxPGg6zACOJOE9PdY9ELAw)

[Results](https://adisa-shobi.github.io/MLOP_Summative/)


## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API Locally

1. Start the API using Docker Compose:
   ```bash
   docker-compose up api
   ```

2. The API will be available at http://localhost:8000

### API Endpoints

The API provides the following endpoints:

#### Home
- **URL**: `/v1/`
- **Method**: `GET`
- **Description**: Root endpoint that provides information about available endpoints
- **Response**: JSON with welcome message and list of endpoints

#### Upload Training Data
- **URL**: `/v1/upload-training-data`
- **Method**: `POST`
- **Description**: Upload a CSV file containing training data
- **Request**: Form data with CSV file and optional base filename
- **Response**: JSON with upload status and file information

#### Predict Experience Level
- **URL**: `/v1/predict`
- **Method**: `POST`
- **Description**: Predict the experience level of a gym member based on their data
- **Request Body**: JSON with member data (see example below)
- **Response**: JSON with predicted experience level and confidence scores

#### Retrain Model
- **URL**: `/v1/retrain`
- **Method**: `POST`
- **Description**: Retrain the model with new data
- **Request Body**: JSON with filename of training data
- **Response**: JSON with retraining status and details

#### Model Evaluation
- **URL**: `/v1/evaluate`
- **Method**: `GET`
- **Description**: Get model evaluation metrics including accuracy, precision, recall, and F1 score
- **Response**: JSON with evaluation metrics

#### Training Data Statistics
- **URL**: `/v1/training-data`
- **Method**: `GET`
- **Description**: Get list of all training files
- **Response**: JSON with list of training files

#### Delete Training File
- **URL**: `/v1/training-data/{filename}`
- **Method**: `DELETE`
- **Description**: Delete a specific training file
- **Response**: JSON with deletion status

#### Delete All Training Files
- **URL**: `/v1/training-data`
- **Method**: `DELETE`
- **Description**: Delete all training files leaving only 1
- **Response**: JSON with deletion status

#### Visualization Data
- **URL**: `/v1/visualization-data`
- **Method**: `GET`
- **Description**: Get data for visualizations
- **Query Parameters**: 
  - `filename` (optional): Specific CSV file to visualize. If not provided, uses default dataset
- **Response**: JSON with visualization data and source file information

Example prediction request:
```bash
curl --request POST \
  --url http://localhost:8000/v1/predict \
  --header 'Content-Type: application/json' \
  --data '{
    "age": 30,
    "gender": "Male",
    "weight": 75.5,
    "height": 1.75,
    "max_bpm": 185,
    "avg_bpm": 145,
    "resting_bpm": 65,
    "session_duration": 1.5,
    "calories_burned": 450,
    "workout_type": "HIIT",
    "fat_percentage": 15.5,
    "water_intake": 2.5,
    "workout_frequency": 4,
    "bmi": 24.6
}'
```

Example response:
```json
{
	"status": "success",
	"prediction": {
		"class": 1,
		"probabilities": [
			0.3332487642765045,
			0.33359742164611816,
			0.33315378427505493
		]
	},
	"message": "Prediction completed successfully"
}
```

## Project Structure

```
.
├── data/                  # Dataset files
├── models/                # Trained model files
├── notebook/              # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing functions
│   ├── model.py           # Model training and evaluation
│   └── api.py             # FastAPI application
│   ├── visualizations.py  # Derives trends fron selected dataset
├── locustfile.py          # Locust load testing configuration
├── docker-compose.yml     # Docker Compose configuration
├── locust_result.html     # Results fro Locust test
├── Dockerfile             # Docker configuration
└── requirements.txt       # Python dependencies
```

## Model Performance

The model is trained to classify gym members into three experience levels:
- Level 0: Beginner
- Level 1: Intermediate
- Level 2: Advanced

The model achieves high accuracy on the test set and is deployed as a REST API for easy integration with other applications.

## Acknowledgments

- [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset/data) from Kaggle
- TensorFlow and scikit-learn for machine learning capabilities
- FastAPI for the REST API
- Locust for load testing 