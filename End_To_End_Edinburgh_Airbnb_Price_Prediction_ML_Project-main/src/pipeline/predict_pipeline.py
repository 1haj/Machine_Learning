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
        bathrooms,
        bedrooms,
        beds,
        accommodates,
        guests_included,
        minimum_nights_avg_ntm,
        maximum_nights_avg_ntm,
        cleaning_fee,
        extra_people,
        security_deposit,
        host_response_rate,
        room_type,
        property_type,
        neighbourhood_cleansed,
        host_is_superhost,
        bed_type,
        cancellation_policy
        ):

        self.bathrooms = bathrooms

        self.bedrooms = bedrooms

        self.beds = beds

        self.accommodates = accommodates

        self.guests_included = guests_included

        self.minimum_nights_avg_ntm = minimum_nights_avg_ntm

        self.maximum_nights_avg_ntm = maximum_nights_avg_ntm

        self.cleaning_fee = cleaning_fee

        self.extra_people = extra_people

        self.security_deposit = security_deposit
        
        self.host_response_rate = host_response_rate
        
        self.room_type=room_type
        self.property_type=property_type

        self.neighbourhood_cleansed = neighbourhood_cleansed


        self.host_is_superhost =host_is_superhost
        
        self.bed_type = bed_type
        
        self.cancellation_policy=cancellation_policy

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "bathrooms": [self.bathrooms],
                "bedrooms": [self.bedrooms],
                "beds": [self.beds],
                "accommodates": [self.accommodates],
                "guests_included": [self.guests_included],
                "minimum_nights_avg_ntm": [self.minimum_nights_avg_ntm],
                "maximum_nights_avg_ntm": [self.maximum_nights_avg_ntm],
                "cleaning_fee": [self.cleaning_fee],
                "extra_people": [self.extra_people],
                "security_deposit": [self.security_deposit],
                "host_response_rate": [self.host_response_rate],
                "room_type": [self.room_type],
                "property_type": [self.property_type],
                "neighbourhood_cleansed": [self.neighbourhood_cleansed],
                "host_is_superhost": [self.host_is_superhost],
                "bed_type": [self.bed_type],
                "cancellation_policy": [self.cancellation_policy]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

