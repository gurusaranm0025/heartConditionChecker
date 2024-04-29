import pandas as pd
import joblib
import os
script_dir = os.path.dirname(__file__)

from config import BUILDS_FOLDER_NAME, FEATURES_ORDER_ON_FIT_FILE
from typing import Dict

class Input:    
    def __init__(self, phys_health: float, ment_health: float, sleep_time: float, bmi: float, smoking, alcohol_drink, stroke, diff_walk, sex, age_cat, race, diabetic, phys_act, gen_health, asthma, kid_dis, skin_canc):
        self.phys_health: float = phys_health
        self.ment_health: float = ment_health
        self.sleep_time: float = sleep_time
        self.bmi: float = bmi
        self.smoking = smoking
        self.alcohol_drink = alcohol_drink
        self.stroke = stroke
        self.diff_walk = diff_walk
        self.sex = sex
        self.age_cat = age_cat
        self.race = race
        self.diabetic = diabetic
        self.phys_act = phys_act
        self.gen_health = gen_health
        self.asthma = asthma
        self.kid_dis = kid_dis
        self.skin_canc = skin_canc
        
    def arange_features(self, features: Dict[str,list]) -> dict:
        features_order_during_fit: list = joblib.load(f"{script_dir}/{BUILDS_FOLDER_NAME}/{FEATURES_ORDER_ON_FIT_FILE}")
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