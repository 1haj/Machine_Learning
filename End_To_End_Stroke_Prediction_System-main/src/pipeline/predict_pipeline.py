import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        age, bmi, avg_glucose_level,gender, ever_married, Residence_type,work_type, smoking_status
        ):

        self.age = age

        self.bmi = bmi

        self.avg_glucose_level = avg_glucose_level

        self.gender = gender

        self.ever_married = ever_married

        self.Residence_type = Residence_type

        self.work_type = work_type

        self.smoking_status = smoking_status


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "bmi": [self.bmi],
                "avg_glucose_level": [self.avg_glucose_level],
                "gender": [self.gender],
                "ever_married": [self.ever_married],
                "Residence_type": [self.Residence_type],
                "work_type": [self.work_type],
                "smoking_status": [self.smoking_status]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

