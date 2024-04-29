import pandas as pd
import numpy as np
import os
script_dir = os.path.dirname(__file__)
import joblib

from config import BUILDS_FOLDER_NAME, ENCODERS_FILE_NAME, SCALERS_FILE_NAME, SKLEARN_MODELS_FILE_NAMES, KERAS_MODELS_FILE_NAMES, DATASET_PATH_FOR_LINUX, CAT_COLS, NUM_COLS, TARGET_CLASS
from input_type import Input
from Dataset import TrainTestDataset
from typing import Dict, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

dataset = TrainTestDataset(dataset_path=DATASET_PATH_FOR_LINUX, test_size=0.2)

class Model:
    def __init__(self, file_name: str, dataset: TrainTestDataset):
        self.filename: str = file_name
        self.name: str = file_name.removesuffix('.pkl')
        print(f"Loading {self.name} mdoel")
        self.path: str = f"{script_dir}/{BUILDS_FOLDER_NAME}/{file_name}"
        self.model= joblib.load(self.path)
        self.encoders: Dict[str,LabelEncoder] = None
        self.scalers: Dict[str,StandardScaler] = None
        self.sampleX: pd.DataFrame = self.__load_sampleX()
        self._prediction_inp: pd.DataFrame = None
        self.__load_encoders()
        self.__load_scalers()
        
    def __load_sampleX(self) -> None:
        df = dataset.get_X()
        df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)
        self.sampleX = df
    
    def __load_encoders(self) -> None:
        path = f"{script_dir}/{BUILDS_FOLDER_NAME}/{ENCODERS_FILE_NAME}"
        self.encoders = joblib.load(path)
    
    def __load_scalers(self) -> None:
        path = f"{script_dir}/{BUILDS_FOLDER_NAME}/{SCALERS_FILE_NAME}"
        self.scalers = joblib.load(path)
        
    def __encode_cat_features(self, cat_cols: list = CAT_COLS) -> None:
        for col in cat_cols:
            if col != TARGET_CLASS:
                self._prediction_inp[col] = self.encoders[col].transform(self._prediction_inp[col])
    
    def __scale_num_features(self, num_cols: list = NUM_COLS) -> None:
        for col in num_cols:
            feature_to_scale: pd.DataFrame = self._prediction_inp[col]
            self._prediction_inp[col] = self.scalers[col].transform(feature_to_scale.values.reshape(-1,1)).squeeze()
        
    def preprocess_input(self, input: Input) -> None:
        self._prediction_inp = pd.concat([input.get_input_item(), self.sampleX], axis=0)
        self._prediction_inp = self._prediction_inp[:1]
        self.__encode_cat_features()
        self.__scale_num_features()        
            
    def predict(self, input: Input) -> np.ndarray:
        # self._prediction_inp = input.get_input_item()
        # self._prediction_inp = pd.concat([input.get_input_item(), self.sampleX], axis=0)
        # self._prediction_inp = self._prediction_inp[:1]
        # self.__encode_cat_features()
        # self.__scale_num_features()
        self.preprocess_input(input=input)
        prediction = self.model.predict(self._prediction_inp)
        return prediction
    
    def predict_proba(self, input: Input):
        self.preprocess_input(input=input)
        try:
            proba = self.model.predict_proba(self._prediction_inp)
            return proba
        except:
            return None

class EnsembleModelCollection:
    def __init__(self, models: dict):
        self.models_dict: Dict[str,Model] = {}
        self.__load_models(models=models)
        encoder_path = f"{script_dir}/{BUILDS_FOLDER_NAME}/{ENCODERS_FILE_NAME}"
        self.encoder: Dict[str,LabelEncoder] = joblib.load(encoder_path)
    
    def __load_models(self, models: dict) -> None:
        for name, filename in models.items():
            self.models_dict[name] = Model(dataset=dataset, file_name=filename)
    
    def __OneDArrayHandler(self, dict: dict) -> dict:
        local_dict = dict
        local_dict['ANN'] = np.array([int(local_dict['ANN'][0][0])])
        return local_dict
    
    def most_freq_class(self, classes: dict) -> np.ndarray:
        all_classes = self.__OneDArrayHandler(dict=classes)
        all_classes = [class_pred[0] for name, class_pred in classes.items()]
        class_counts = Counter(all_classes)
        most_freq, _ = class_counts.most_common()[0]
        return np.array([most_freq])
    
    def avg_proba(self, probabilities: dict) -> float:
        sum: int = 0
        n: int = 0
        for name, probability in probabilities.items():
            # if type(probability) != None:
            try:
                sum += round(probability[0][1]*100, 2)
                n += 1
            except:
                continue
        return round((float(sum)/float(n)), 2)
        
    def predict(self, input: Input) -> dict:
        all_predictions: Dict[str,np.ndarray] = {}
        all_predictions_proba :Dict[str] = {}
        for name, model in self.models_dict.items():
            all_predictions[name] = model.predict(input=input)
            # if name != 'ANN' or name != 'LinearSVC' or name != 'DecisionTree':
            all_predictions_proba[name] = model.predict_proba(input=input)
        final_prediction = self.most_freq_class(all_predictions)
        final_prediction = self.encoder[TARGET_CLASS].inverse_transform(final_prediction)
        return {'final_prediction': final_prediction, 'prediction_proba': self.avg_proba(probabilities=all_predictions_proba)}

if __name__ == "__main__":
    all_models: dict = {**KERAS_MODELS_FILE_NAMES, **SKLEARN_MODELS_FILE_NAMES}
    EnsembleModel = EnsembleModelCollection(models=all_models)
    input = Input(bmi=44.56,ment_health=5,phys_health=8,sleep_time=6,age_cat='55-59',alcohol_drink='Yes',asthma='No',diabetic='No',diff_walk='No',gen_health='Good',kid_dis='No',phys_act='No',race='Other',sex='Male',skin_canc='No',smoking='Yes',stroke='No')
    print(EnsembleModel.predict(input=input))
    
