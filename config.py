CAT_COLS: list = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer", "HeartDisease"]

NUM_COLS: list = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']

TARGET_CLASS: str = "HeartDisease"

DATASET_NAME = "heart_2020_cleaned.csv"

DATASET_MAIN_FOLDER = "dataset"

DATASET_SUB_FOLDER = "2020"

# DATASET_PATH_FOR_LINUX: str = "./dataset/2020/heart_2020_cleaned.csv"

# DATASET_PATH_FOR_WINDOWS: str = ".\\dataset\\2020\\heart_2020_cleaned.csv"

CLASS_WEIGHTS: dict = {0:1, 1:5}

ENCODERS_FILE_NAME = "LabelEncoders.pkl"

SCALERS_FILE_NAME = "StandardScalers.pkl"

FEATURES_ORDER_ON_FIT_FILE = "FeatureFitOrder.pkl"

SKLEARN_MODELS_FILE_NAMES = {'DecisionTree': 'DecisionTree.pkl', 'LinearSVC': 'LinearSVC.pkl', 'LogisticRegression': 'LogisticRegression.pkl'}

KERAS_MODELS_FILE_NAMES = {'ANN': 'ANN.pkl'}

BUILDS_FOLDER_NAME = "builds"