import sys
import os

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    try:
        best_estimators = {}
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            
            search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy') if param_grid else None
            
            if search:
                search.fit(X_train, y_train)  
                best_estimators[model_name] = search.best_estimator_
                
            else:
                model.fit(X_train, y_train)
                best_estimators[model_name] = model

        results = {}
        for model_name, model in best_estimators.items():

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy_score = accuracy_score(y_train, y_train_pred)

            test_accuracy_score = accuracy_score(y_test, y_test_pred)

            results[model_name] = test_accuracy_score

        return results, best_estimators 

    except Exception as e:
        raise CustomException(e, sys)