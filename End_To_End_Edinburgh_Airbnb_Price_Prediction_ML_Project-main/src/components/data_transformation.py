import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object,CurrencyPercentCleaner,TargetGuidedEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            # Numeric columns
            
            numerical_columns = [
                'bathrooms', 'bedrooms', 'beds', 'accommodates', 'guests_included',
                'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm','cleaning_fee', 'extra_people','security_deposit', 'host_response_rate']

            columns_dollar_percents = ['security_deposit', 'cleaning_fee', 'extra_people', 'host_response_rate']
            # columns_dollar_percents = ['security_deposit', 'cleaning_fee']
            

            num_pipeline = Pipeline(steps=[
            ("cleaner", CurrencyPercentCleaner(columns=columns_dollar_percents)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ])

            # One-hot categorical columns
            # one_hot_columns = ['room_type','host_response_time','bed_type','instant_bookable','cancellation_policy','host_is_superhost']
            one_hot_columns = ['room_type','bed_type','cancellation_policy','host_is_superhost']
            one_hot_pipeline = Pipeline(steps=[
                ("one_hot", OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler())
            ])
            
            

            # Target-guided categorical columns
            target_encod_columns = ['property_type','neighbourhood_cleansed']
            # target_encod_columns = ['property_type']
            target_pipeline = Pipeline(steps=[
                ("target_enc", TargetGuidedEncoder(columns=target_encod_columns)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler())
            ])
        
        
            
            # Combine into ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("one_hot_cat", one_hot_pipeline, one_hot_columns),
                ("target_enc_cat", target_pipeline, target_encod_columns)
            ])
            logging.info(f"Numerical columns: {numerical_columns}")
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price_target"
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            # print(f"{type(input_feature_train_df)} is my data")
            upper_limit_train = target_feature_train_df.quantile(0.95)
            target_feature_train_df_filtered = target_feature_train_df[target_feature_train_df<upper_limit_train]
            input_feature_train_df_filtered=input_feature_train_df.loc[target_feature_train_df_filtered.index,:]
            
            # print(input_feature_train_df_filtered.shape,target_feature_test_df_filtered.shape)
            upper_limit_test= target_feature_test_df.quantile(0.95)
            target_feature_test_df_filtered = target_feature_test_df[target_feature_test_df<upper_limit_test]
            input_feature_test_df_filtered=input_feature_test_df.loc[target_feature_test_df_filtered.index,:]
            
            # print(upper_limit)

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df_filtered,target_feature_train_df_filtered)
            # print(input_feature_train_arr)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df_filtered)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df_filtered)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df_filtered)]
            # print(pd.DataFrame(input_feature_train_arr).isna().sum())

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
