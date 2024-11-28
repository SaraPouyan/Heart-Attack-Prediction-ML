from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                age=request.form.get('age', type=int),
                sex=request.form.get('sex', type=int),
                cp=request.form.get('cp', type=int),
                trtbps=request.form.get('trtbps', type=int),
                chol=request.form.get('chol', type=int),
                fbs=request.form.get('fbs', type=int),
                restecg=request.form.get('restecg', type=int),
                thalachh=request.form.get('thalachh', type=int),
                exng=request.form.get('exng', type=int),
                oldpeak=request.form.get('oldpeak', type=float),
                slp=request.form.get('slp', type=int),
                caa=request.form.get('caa', type=int),
                thall=request.form.get('thall', type=int)
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("after Prediction")
            return render_template(
                'home.html',
                results=int(results[0])  
            )
        except Exception as e:
            return render_template(
                'home.html',
                results="Error occurred: " + str(e)
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0")   