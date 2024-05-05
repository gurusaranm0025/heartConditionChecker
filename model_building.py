from config import CLASS_WEIGHTS, DATASET_PATH_FOR_LINUX, DATASET_NAME, DATASET_MAIN_FOLDER, DATASET_SUB_FOLDER, BUILDS_FOLDER_NAME
from Dataset import TrainTestDataset

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os 
script_dir = os.path.dirname(__file__)

Stratified_KFold = StratifiedKFold(n_splits=10)

ModelDataset = TrainTestDataset(dataset_path=os.path.join(script_dir,DATASET_MAIN_FOLDER,DATASET_SUB_FOLDER,DATASET_NAME),save_encoders=True, test_size=0.2, )

x_train, x_test, y_train, y_test = ModelDataset.get_train_test_set()

print("MODEL BUILDING ------------------------------------------------------------------------------------------------------------------------")
final_models: dict = {}

print("LOG MODEL BUILDING ------------------------------------------------------------------------------------------------------------------------")

log_model = LogisticRegression(max_iter=500, fit_intercept=True, intercept_scaling=1, dual=False, multi_class='auto', n_jobs=None, random_state=None, tol=0.0001, l1_ratio=None, warm_start=False)

C_values = np.logspace(-2, 2, 5)
params_grid = {
    'solver': ['saga', 'newton_cg'],
    'penalty': ['l2'],
    'C': [1.0],
    'class_weight': [None, CLASS_WEIGHTS],    
}

best_params_lr_ar = {
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'C': [1.0],
    'class_weight': [None]
}

grid_search_lr = GridSearchCV(estimator=log_model, param_grid=best_params_lr_ar, cv=Stratified_KFold, scoring='accuracy', n_jobs=7, verbose=1, return_train_score=True)
grid_search_lr.fit(x_train, y_train)
best_params_lr = grid_search_lr.best_params_
best_score_lr = grid_search_lr.best_score_
best_model_lr = grid_search_lr.best_estimator_
final_models['LogisticRegression'] = {'estimator': best_model_lr, 'params': best_params_lr, 'score': best_score_lr}
print('BEST PARAMETER CONFIG IS ==> ', best_params_lr)
print('BEST SCORE IS ==> ', best_score_lr)
predictions_lr = best_model_lr.predict(x_test)
cm_lr = confusion_matrix(y_pred=predictions_lr, y_true=y_test)
plt.figure(figsize=(12,8))
sns.heatmap(cm_lr, annot=True)
plt.title("Confusion Matrix for Logistic Regression on Test Data")
plt.show()
# print("DECISION TREE MODEL BUILDING ----------------------------------------------------------------------------------------------------")
# decision_tree_model = DecisionTreeClassifier()

# params_grid_dt = {
#     'criterion': ['log_loss', 'entropy'],
#     'max_depth': [5, 15, 25],
#     'random_state': [0, None],
#     'class_weight': [None, CLASS_WEIGHTS]
# }

# best_params_dt_ar = {
#     'criterion': ['log_loss'],
#     'max_depth': [5],
#     'random_state': [0],
#     'class_weight': [None]
# }

# grid_search_dt = GridSearchCV(estimator=decision_tree_model, param_grid=best_params_dt_ar, n_jobs=7, cv=Stratified_KFold, scoring='accuracy', verbose=1, return_train_score=True)

# grid_search_dt.fit(x_train, y_train)
# best_params_dt = grid_search_dt.best_params_
# best_score_dt = grid_search_dt.best_score_
# best_model_dt = grid_search_dt.best_estimator_
# final_models['DecisionTree'] = {'estimator': best_model_dt, 'params': best_params_dt, 'score': best_score_dt}
# print('BEST PARAMETER CONFIG IS ==> ', best_params_dt)
# print('BEST SCORE IS ==> ', best_score_dt)


# print("RANDOM FOREST MODEL BUILDING ----------------------------------------------------------------------------------------------------")
# random_forest_model = RandomForestClassifier()

# params_grid_rf = {
#     'max_depth': [7],
#     'n_estimators': [25],
#     'random_state': [0],
#     'class_weight': [None, CLASS_WEIGHTS]
# }

# best_params_rf_ar = {
#     'max_depth': [10],
#     'n_estimators': [25],
#     'random_state':[0],
#     'class_weight': [None]
# }

# grid_search_rf = GridSearchCV(estimator=random_forest_model, param_grid=best_params_rf_ar, n_jobs=7, cv=Stratified_KFold, scoring='accuracy', verbose=1, return_train_score=True)

# grid_search_rf.fit(x_train, y_train)
# best_params_rf = grid_search_rf.best_params_
# best_score_rf = grid_search_rf.best_score_
# best_model_rf = grid_search_rf.best_estimator_
# final_models['RandomForest'] = {'estimator': best_model_rf, 'params': best_params_rf, 'score': best_score_rf}
# print('BEST PARAMETER CONFIG IS ==> ', best_params_rf)
# print('BEST SCORE IS ==> ', best_score_rf)

# print("LINEAR SUPPORT VECTOR CLASSIFIER MODEL BUILDING ----------------------------------------------------------------------------------------------------")
# linear_svc_model = LinearSVC(max_iter=800)

# C_values = np.logspace(-2, 2, 5)
# params_grid_lsvc = {
#     'random_state': [0],
#     'penalty': ['l2'],
#     'C': [100.0, 1.0],
#     'class_weight': [None, CLASS_WEIGHTS]
# }

# best_params_lsvc_ar = {
#     'random_state': [0],
#     'penalty': ['l2'],
#     'C': [1.0],
#     'class_weight': [None]
# }

# grid_search_lsvc = GridSearchCV(estimator=linear_svc_model, param_grid=best_params_lsvc_ar, n_jobs=11, cv=Stratified_KFold, scoring='accuracy', verbose=1)

# grid_search_lsvc.fit(x_train, y_train)
# best_params_lsvc = grid_search_lsvc.best_params_
# best_score_lsvc = grid_search_lsvc.best_score_
# best_model_lsvc = grid_search_lsvc.best_estimator_
# final_models['LinearSVC'] = {'estimator': best_model_lsvc, 'params': best_params_lsvc, 'score': best_score_lsvc}
# print('BEST PARAMETER CONFIG IS ==> ', best_params_lsvc)
# print('BEST SCORE IS ==> ', best_score_lsvc)


# print("ARTIFICIAL NEURAL NETWORK MODEL BUILDING ----------------------------------------------------------------------------------------------------")

# from keras import models, layers

# ANN_model = models.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)),
#     layers.BatchNormalization(),
#     layers.Dense(units=32, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=16, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=8, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=4,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=2,activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=1, activation='sigmoid')
# ])

# ANN_model = models.Sequential([
#     layers.InputLayer(shape=(x_train.shape[1],)),
#     layers.Dense(units=17, activation='relu'),
#     layers.Dense(units=9, activation='relu'),
#     layers.Dense(units=1, activation='sigmoid')
# ])

# ANN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','precision'])
# ANN_model.fit(x_train, y_train, epochs=13, batch_size=17, validation_split=0.2)
# loss, accuracy, precison = ANN_model.evaluate(x_test, y_test)
# print("TEST LOSS ==> ", loss)
# print("TEST ACCURACY ==> ", accuracy)
# print("TEST PRECISION ==> ", precison)
# final_models['ANN'] = {'estimator': ANN_model, 'params': ANN_model.get_config(), 'score': {'loss': loss, 'accuracy': accuracy, 'precision': precison}}

# def generate_names(models: dict = final_models):
#     name_dict = {}
#     for model_name, _ in models.items():
#         if model_name != 'ANN':
#             name_dict[model_name] = f"{model_name}.pkl"
#         else:
#             name_dict[model_name] = f"{model_name}.pkl"
#     return name_dict

def save_models(models: dict = final_models):
    # names = generate_names()
    for model_name, model_dict in models.items():
        print(f"{model_name}: {model_dict}")
        file_path = os.path.join(script_dir,BUILDS_FOLDER_NAME,f"{model_name}.pkl")
        # file_path = f"{script_dir}/builds/{model_name}.pkl"
        joblib.dump(model_dict['estimator'], file_path)

# if __name__ == "__main__":
    # save_models()