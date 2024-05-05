from flask import Flask, request, jsonify
from input_type import Input
from config import KERAS_MODELS_FILE_NAMES, SKLEARN_MODELS_FILE_NAMES
from model import EnsembleModelCollection
import os
app = Flask(__name__)


all_models: dict = {**KERAS_MODELS_FILE_NAMES, **SKLEARN_MODELS_FILE_NAMES}
EnsembleModel = EnsembleModelCollection(models=all_models)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        inputClass: Input = Input(input_dict=data)
        prediction = EnsembleModel.predict(input=inputClass)
        final_prediction = {}
        for name, value in prediction.items():
            if type(value) == float:
                final_prediction[name] = value
            else:
                final_prediction[name] = value[0]
        print("prediction successfull ==>", final_prediction)
        return jsonify(final_prediction)

if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__) 
    os.chdir(curr_dir)
    os.system("python -m gunicorn --config gunicorn_config.py app:app")
    