import sys
from dataclasses import dataclass



#standard files
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler



#from the crreated files
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#this is a type of decorator
@dataclass #simple class to store data  
# and automatically adds
# 1 construcor (__init__)
# 2 string representation (__repr__)
# 3 comparison method (__eq__) 


class DataTransformationConfig:
    pre_processor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for the data transformation
        """
        
        try:
            numerical_column=["writing_score","reading_score"]
            categorical_column=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"numerical columns:{numerical_column}")
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical columns{categorical_column}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_dataframe=pd.read_csv(train_path)
            test_dataframe=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_object=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_column=["writing_score","reading_score"]

            input_feature_train_dataframe=train_dataframe.drop(columns=[target_column_name],axis=1)
            target_feature_train_dataframe=train_dataframe[target_column_name]


            input_feature_test_dataframe=test_dataframe.drop(columns=[target_column_name],axis=1)
            target_feature_test_dataframe=test_dataframe[target_column_name]



            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_array=preprocessing_object.fit_transform(input_feature_train_dataframe)
            input_feature_test_array=preprocessing_object.transform(input_feature_test_dataframe)

            #difference between tranform , fit and fit_transform:
            #  fit():
            #        gives mean and standard deviation of each feature
            #  tranform():
            #        uses the formula x=(x-mean)/standard_deviation
            #  fit_transform():
            #        uses both fit and transform 


            train_array=np.c_[
                input_feature_train_array,np.array(target_feature_train_dataframe)
            ]
            test_array=np.c_[
                input_feature_test_array,np.array(target_feature_test_dataframe)
            ]
            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.pre_processor_obj_file_path,
                object=preprocessing_object
            )

            return(
                train_array,
                test_array,
                self.data_transformation_config.pre_processor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)