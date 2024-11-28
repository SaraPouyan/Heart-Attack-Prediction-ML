
# Heart Attack Prediction ML

<div align="center">
    <img src="./static/assets/background.png" alt="Logo" width="1200" height="600">
</div>

Heart-Attack-Prediction-ML is a machine learning application for predicting the likelihood of a heart attack based on various health parameters. The project includes data preprocessing, model training, and a Flask web application to interact with the prediction system.

## Features

- **Exploratory Data Analysis (EDA):** Analyze the dataset with Jupyter notebooks.
- **Data Pipeline:** Automates data ingestion, transformation, and model training.
- **Web Application:** Flask-based interface to input patient data and get predictions.

## Prerequisites

- Python 3.7 or above
- pip (Python package manager)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SaraPouyan/Heart-Attack-Prediction-ML.git
   cd Heart-Attack-Prediction-ML
   ```

2. **Create and activate a virtual environment** (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask server**:

   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://127.0.0.1:5000/predict-data`.

3. **Input the required patient data** in the web interface to receive a prediction.

## Demo

Here’s a quick demonstration of the application:

  <div align="center">
      <img src="./static/assets/demo.gif" width="500" height="400" alt="Demo GIF" align="center">
  </div>


## Notebooks

Explore the data and model training process using the notebooks in the `notebook/` directory:

 - `heart_attack_analysis_eda.ipynb`: Exploratory Data Analysis
 - `model_training.ipynb`: Model training and evaluation


## License

This project is licensed under the MIT License.
