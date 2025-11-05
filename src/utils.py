

########## this utils will have all the commmon things trying to use in the entire project


import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


from sklearn.metrics import r2_score


def save_object(file_path,object):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_object:
            dill.dump(object,file_object)
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for model_name in models:

            model=models[model_name]
            param=params[model_name]

            gs=GridSearchCV(model,param,cv=3)
            gs.fit(x_train,y_train)


            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

           # model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score=r2_score(y_test,y_test_pred)
            report[model_name]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
