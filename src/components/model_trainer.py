import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data!")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=5000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42),
                "Support Vector Machine (SVM)": SVC(),
                "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),  
                "Neural Network (MLP)": MLPClassifier(tol=1e-3, max_iter=500, random_state=42),
                "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42),
                "Extra Trees": ExtraTreesClassifier(random_state=42)
            }

            param_grids = {
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                },
                "XGBoost": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "Support Vector Machine (SVM)": {
                    'C': [0.1, 1, 3, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                },
                "K-Nearest Neighbors (KNN)": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "Naive Bayes": {},  
                "Neural Network (MLP)": {
                    'hidden_layer_sizes': [(50,), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['adam', 'sgd'],
                    'learning_rate': ['constant', 'adaptive']
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 1, 10]
                },
                "Extra Trees": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }

            results, fitted_models = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param_grids=param_grids
            )
        
            best_model_score = max(results.values())
            best_model_name = max(results, key=results.get)
            best_model = fitted_models[best_model_name] 

            if best_model_score < 0.6:
                raise CustomException("No best model found!")

            logging.info(f"Best model: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)