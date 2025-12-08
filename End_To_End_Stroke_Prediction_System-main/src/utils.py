import os
import sys

import numpy as np 
import pandas as pd
# import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import re
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


class CurrencyPercentCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].apply(
                    lambda x: float(re.sub(r"[^\d.]", "", str(x))) if pd.notnull(x) else x
                )
        return X


class TargetGuidedEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoding_dict = {}

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y, name="stroke")
        
        combined = pd.concat([X, y], axis=1)
        

        for col in self.columns:
        
            self.encoding_dict[col] = combined.groupby(col)[y.name].mean().to_dict()
            
        
        return self

    def transform(self, combined):
        combined = combined.copy()
        for col in self.columns:
            if col in combined.columns:
                combined[col] = combined[col].map(self.encoding_dict[col])
        return combined


    
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, scoring='accuracy', error_score='raise')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            model_score = round(model.score(X_test, y_test), 3)

            print(f"model: {model}, length: {len(y_test_pred)}, length of predicted 1s: {len(y_test_pred[y_test_pred==1])}, score: {model_score}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"Confusion Matrix for {list(models.keys())[i]}:\n{cm}\n")

            # Optional: classification report
            cr = classification_report(y_test, y_test_pred)
            print(f"Classification Report for {list(models.keys())[i]}:\n{cr}\n")

            report[list(models.keys())[i]] = model_score
            logging.info(f"my report:{report}")

        return report
    except Exception as e:
        raise CustomException(e, sys)


    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)