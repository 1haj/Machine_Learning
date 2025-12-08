import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

################### Sklearn ####################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(),
                "SVC": SVC(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }
            params={
                
                "Random Forest":{

                'n_estimators' : [50, 100, 250, 500],
                'criterion' : ['gini', 'entropy', 'log_loss'],
                'max_features' : ['sqrt', 'log2']
            
                },
                # "Linear Regression":{},
                "LogisticRegression":{
                'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                'class_weight' : ['balanced'],
                'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'max_iter':[1000]
                },
                "SVC":{
                    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                    'class_weight':['balanced']
                },
                "DecisionTreeClassifier":{
                'criterion' : ['gini', 'entropy', 'log_loss'],
                'splitter' : ['best', 'random'],
                'max_depth' : list(np.arange(4, 30, 1))
                },
                "KNeighborsClassifier":{
                'n_neighbors' : list(np.arange(3, 20, 2)),
                'p' : [1, 2, 3, 4]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            

            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            # print(best_model)

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            print(best_model_name,accuracy)
            return accuracy
            
        except Exception as e:
            raise CustomException(e,sys)