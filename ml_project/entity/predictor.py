import os
import sys

from ml_project.exception import ProjectException
from ml_project.util.util import load_object

import pandas as pd


class ml_projectData:

    def __init__(self,
                age:str,
                workclass:int,
                fnlwgt:str,
                education:int,
                education_num:str,
                marital_status:int,
                occupation:int,
                relationship:int,
                race:int,
                sex:int,
                capital_gain:str,
                capital_loss:str,
                hours_per_week:str,
                country:int,
                salary:str = None
                 ):
        try:
            self.age = age
            self.workclass = workclass
            self.fnlwgt = fnlwgt
            self.education = education
            self.education_num = education_num
            self.marital_status = marital_status
            self.occupation = occupation
            self.relationship = relationship
            self.race = race
            self.sex = sex
            self.capital_gain = capital_gain
            self.capital_loss = capital_loss
            self.hours_per_week = hours_per_week
            self.country = country
            self.salary = salary
        except Exception as e:
            raise ProjectException(e, sys) from e

    def get_ml_project_input_data_frame(self):

        try:
            ml_project_input_dict = self.get_ml_project_data_as_dict()
            return pd.DataFrame(ml_project_input_dict)
        except Exception as e:
            raise ProjectException(e, sys) from e

    def get_ml_project_data_as_dict(self):
        try:
            input_data = {
                "age" : [self.age],
                "workclass" : [self.workclass],
                "fnlwgt" : [self.fnlwgt],
                "education" : [self.education],
                "education_num" : [self.education_num],
                "marital_status" : [self.marital_status],
                "occupation" : [self.occupation],
                "relationship" : [self.relationship],
                "race" : [self.race],
                "sex" : [self.sex],
                "capital_gain" : [self.capital_gain],
                "capital_loss" : [self.capital_loss],
                "hours_per_week" : [self.hours_per_week],
                "country" : [self.country]
                }
            return input_data
        except Exception as e:
            raise ProjectException(e, sys)


class ml_projectPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise ProjectException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise ProjectException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            salary = model.predict(X)
            if salary == 1:
                return 'salary will be >50k'
            elif salary == 0:
                return 'salary will be <50k'
        except Exception as e:
            raise ProjectException(e, sys) from e