import pandas as pd
import joblib
import os
script_dir = os.path.dirname(__file__)
from config import CAT_COLS, NUM_COLS, DATASET_PATH_FOR_LINUX, DATASET_PATH_FOR_WINDOWS, TARGET_CLASS, BUILDS_FOLDER_NAME, FEATURES_ORDER_ON_FIT_FILE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict

class Preprocessing:
    def __init__(self, dataset_path: str, save_encoders: bool = False) -> None:
        assert len(dataset_path) > 0, "Need the dataset path to get things going."
        self.DATASET_PATH: str = f"{script_dir}/{dataset_path}"
        self.loaded_dataset: pd.DataFrame = pd.read_csv(self.DATASET_PATH)
        self.cat_cols: list = CAT_COLS
        self.num_cols: list = NUM_COLS
        self.categorical_features: pd.DataFrame = self.loaded_dataset[self.cat_cols].copy()
        self.numerical_features: pd.DataFrame = self.loaded_dataset[self.num_cols].copy()
        self.label_encoders: Dict[str,LabelEncoder] = {}
        self.standard_scalers: Dict[str,StandardScaler] = {}
        self.encoded_cat_features: pd.DataFrame = self.categorical_features.copy()
        self.encoded_num_features: pd.DataFrame = self.numerical_features.copy()
        self.label_encode()
        self.standard_scale()
        self.encoded_scaled_dataset = self.get_encoded_scaled_dataset()
        if save_encoders:
            self.save_encoders()
            self.save_scalers()
        
    def label_encode(self) -> None:
        for col in self.cat_cols:
            self.label_encoders[col] = LabelEncoder()
            self.encoded_cat_features[col] = self.label_encoders[col].fit_transform(self.encoded_cat_features[col])
            
    def standard_scale(self) -> None:
        for col in self.num_cols:
            self.standard_scalers[col] = StandardScaler()
            feature_to_scale = self.encoded_num_features[col]
            self.encoded_num_features[col] = self.standard_scalers[col].fit_transform(feature_to_scale.values.reshape(-1,1)).squeeze()
    
    def get_encoded_scaled_dataset(self) -> pd.DataFrame:
        return pd.concat([self.encoded_cat_features, self.encoded_num_features], axis=1)
    
    def save_encoders(self) -> None:
        file_path = script_dir + "/builds/LabelEncoders.pkl"
        joblib.dump(self.label_encoders, file_path)
    
    def save_scalers(self) -> None:
        file_path = script_dir + "/builds/StandardScalers.pkl"
        joblib.dump(self.standard_scalers, file_path)

class TrainTestDataset(Preprocessing):
        def __init__(self, dataset_path: str, test_size: float,save_encoders: bool = False, random_state: int = 0) -> None:
             super().__init__(dataset_path, save_encoders=save_encoders)
             self.X: pd.DataFrame = self.encoded_scaled_dataset.drop(TARGET_CLASS, axis=1)
             self.Y: pd.DataFrame = self.encoded_scaled_dataset[TARGET_CLASS]
             self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state)
             if save_encoders:
                 self.features_order_on_fit()

        def features_order_on_fit(self) -> None:
            order_list: list = [col for col in self.X.columns]
            path = f"{script_dir}/{BUILDS_FOLDER_NAME}/{FEATURES_ORDER_ON_FIT_FILE}"
            joblib.dump(order_list, path)
        
        def get_train_test_shape(self) -> None:
            print("TRAINING SET SHAPE")
            print("X_TRAIN, X_TEST ==> {},{}".format(self.x_train.shape, self.x_test.shape))
            print("TESTING SET SHAPE")
            print("Y_TRAIN, Y_TEST ==> {},{}".format(self.y_train.shape, self.y_test.shape))
            
        def get_train_test_set(self) -> pd.DataFrame:
            return self.x_train, self.x_test, self.y_train, self.y_test
        
        def get_X_Y(self) -> pd.DataFrame:
            return self.X, self.Y

        def get_Y(self) -> pd.DataFrame:
            return self.Y

        def get_X(self) -> pd.DataFrame:
            return self.X

        def get_datset(self) -> pd.DataFrame:
            return pd.concat([self.X, self.Y], axis=1)
                        
             
if __name__ == "__main__":
    sample = Preprocessing(dataset_path=DATASET_PATH_FOR_LINUX)
    print(sample.loaded_dataset)
    print(sample.cat_cols)