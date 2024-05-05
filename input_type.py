import pandas as pd
import joblib
import os
script_dir = os.path.dirname(__file__)

from config import BUILDS_FOLDER_NAME, FEATURES_ORDER_ON_FIT_FILE
from typing import Dict

# def class_init_fn(input_dict: dict):
#     print(input_dict)
#     return Input(phys_act=, age_cat=, alcohol_drink=,asthma=, bmi=, diabetic=, diff_walk=, gen_health=, kid_dis=, ment_health=,phys_health=, race=, sex=, skin_canc=, sleep_time=, smoking=, stroke=)        

class Input:    
    def __init__(self, input_dict: dict):
        self.FULL_INPUT = input_dict
        self.phys_health: float = float(input_dict['PhysicalHealth'])
        self.ment_health: float = float(input_dict['MentalHealth'])
        self.sleep_time: float = float(input_dict['SleepTime'])
        self.bmi: float = float(input_dict['BMI'])
        self.smoking = input_dict['Smoking']
        self.alcohol_drink = input_dict['AlcoholDrinking']
        self.stroke = input_dict['Stroke']
        self.diff_walk = input_dict['DiffWalking']
        self.sex = input_dict['Sex']
        self.age_cat = input_dict['AgeCategory']
        self.race = input_dict['Race']
        self.diabetic = input_dict['Diabetic']
        self.phys_act = input_dict['PhysicalActivity']
        self.gen_health = input_dict['GeneralHealth']
        self.asthma = input_dict['Asthma']
        self.kid_dis = input_dict['KidneyDisease']
        self.skin_canc = input_dict['SkinCancer']
        
    def arange_features(self, features: Dict[str,list]) -> dict:
        features_order_during_fit: list = joblib.load(os.path.join(script_dir,BUILDS_FOLDER_NAME,FEATURES_ORDER_ON_FIT_FILE))
        # features_order_during_fit: list = joblib.load(f"{script_dir}/{BUILDS_FOLDER_NAME}/{FEATURES_ORDER_ON_FIT_FILE})
        old_dataset = features
        new_dataset_dict: Dict[str,list] = {}
        for col in features_order_during_fit:
            new_dataset_dict[col] = old_dataset[col]
        return new_dataset_dict
    
    def get_input_item(self) -> pd.DataFrame:
        features = {
                "PhysicalHealth": [self.phys_health],
                "MentalHealth": [self.ment_health],
                "SleepTime": [self.sleep_time],
                "BMI": [self.bmi],
                "Smoking": [self.smoking],
                "AlcoholDrinking": [self.alcohol_drink],
                "Stroke": [self.stroke],
                "DiffWalking": [self.diff_walk],
                "Sex": [self.sex],
                "AgeCategory": [self.age_cat],
                "Race": [self.race],
                "Diabetic": [self.diabetic],
                "PhysicalActivity": [self.phys_act],
                "GenHealth": [self.gen_health],
                "Asthma": [self.asthma],
                "KidneyDisease": [self.kid_dis],
                "SkinCancer": [self.skin_canc]
            }
        ordered_features = self.arange_features(features=features)
        return pd.DataFrame(ordered_features)