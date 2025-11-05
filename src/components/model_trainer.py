import os
import sys
from dataclasses import dataclass



from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object, 
    evaluate_models,
    )



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "k-Neighbours":KNeighborsRegressor(),
                "XGBregression":XGBRegressor(),
                "catboosting":CatBoostRegressor(verbose=False),
                "Adaboost":AdaBoostRegressor(),
                
            }

            params={
                "Decision Tree":{
                    "criterion":["squared_error","friedman_mse","absolute_error","poisson"],
                    #"splitter":["best","random"],
                    # "max_features":["sqrt","log2"]
                },
                "Random forest":{
                    #"criterion":["squared_error","friedman_mse","absolute_error","poisson"],
                    #"max_features":["sqrt","log2","None"],
                    'n_estimators':[8,16,32,46,128,256]
                },
                "Gradient boosting": {   
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression":{},
                "k-Neighbours": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'euclidean', 'manhattan']
                },
                "XGBregression":{
                    "learning_rate":[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,46,128,256]
                },
                "catboosting":{
                    "depth":[6,8,10],
                    "learning_rate":[.1,.01,.05,.001],
                    "iterations":[30,50,100]
                },
                "Adaboost":{
                    "learning_rate":[.1,.01,.05,.001],
                    #"loss":["linear","square","exponential"],
                    'n_estimators':[8,16,32,46,128,256]
                }

            }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            #to get the best model
            best_model_score=max(sorted(model_report.values()))

            # to get the best model name from the dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<.6:
                raise CustomException("no best model found")
            logging.info(f"Best model found here")

            preprocessing_obj=models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model,
            )


            predicted=best_model.predict(x_test)

            r2_score_of_model=r2_score(y_test,predicted)

            return r2_score_of_model

        except Exception as e:
            raise CustomException(e,sys)
    
