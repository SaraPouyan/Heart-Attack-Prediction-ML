import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
        self.age = int(age)
        self.sex = int(sex)
        self.cp = int(cp)
        self.trtbps = int(trtbps)
        self.chol = int(chol)
        self.fbs = int(fbs)
        self.restecg = int(restecg)
        self.thalachh = int(thalachh)
        self.exng = int(exng)
        self.oldpeak = float(oldpeak)
        self.slp = int(slp)
        self.caa = int(caa)
        self.thall = int(thall)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'sex': [self.sex],
                'cp': [self.cp],
                'trtbps': [self.trtbps],
                'chol': [self.chol],
                'fbs': [self.fbs],
                'restecg': [self.restecg],
                'thalachh': [self.thalachh],
                'exng': [self.exng],
                'oldpeak': [self.oldpeak],
                'slp': [self.slp],
                'caa': [self.caa],
                'thall': [self.thall]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)