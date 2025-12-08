import os
import sys

import numpy as np 
import pandas as pd
# import dill
import pickle
from sklearn.metrics import r2_score
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
                # print("Cleaning:", col)
                # print("Before cleaning:", X[col].unique()[:10])
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
        y = pd.Series(y, name="price_target")
        # print(f"{X}is data")

        # Combine X and y
        combined = pd.concat([X, y], axis=1)
        # print(combined)

        for col in self.columns:
        
            self.encoding_dict[col] = combined.groupby(col)[y.name].mean().to_dict()
            # print(self.encoding_dict)
        
        return self

    def transform(self, combined):
        combined = combined.copy()
        for col in self.columns:
            if col in combined.columns:
                combined[col] = combined[col].map(self.encoding_dict[col])
        return combined


    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info(f"my report:{report}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
        
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)