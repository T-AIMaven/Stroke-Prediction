# | filename: app.py
# | code-line-numbers: true

import tarfile
import tempfile
import numpy as np
import json
import joblib
import logging


from flask import Flask, request, jsonify
from pathlib import Path
import xgboost as xgb


MODEL_PATH = Path(__file__).parent
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the optimal threshold from the evaluation report
def load_optimal_threshold():
    with open(MODEL_PATH / "evaluation.json", "r") as f:
        report = json.load(f)
        return report["metrics"]["best_threshold"]["value"]

OPTIMAL_THRESHOLD = load_optimal_threshold()


class Model:
    model = None

    def load(self):
        """
        Extracts the model package and loads the model in memory
        if it hasn't been loaded yet.
        """
        # We want to load the model only if it is not loaded yet.
        if not Model.model:

            # Before we load the model, we need to extract it in
            # a temporal directory.

            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)
                
                # model_filepath = Path(directory) / "xgbclass"
                model_filepath = Path(directory) / "xgbclass"

          
                logger.info(f"Loading model from {model_filepath}")                  
                Model.model = xgb.Booster()
                Model.model.load_model(model_filepath)
                logger.info("Model loaded successfully")

                

    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        
        # Make predictions
        dtest = xgb.DMatrix(data)

        return Model.model.predict(dtest)


app = Flask(__name__)
model = Model()


@app.route("/predict/", methods=["POST"])
def predict():
    try:
        data = request.data.decode("utf-8").strip().split('\n')
        features = np.array([row.split(",") for row in data]).astype(np.float32)


        # Generate probability predictions
        predictions_proba = model.predict(features)
        
        results = []
        for prediction_proba in predictions_proba:
            # Apply the threshold to get binary predictions
            pred_value = int(prediction_proba >= OPTIMAL_THRESHOLD)
            confidence = float(prediction_proba)
            results.append({"prediction": pred_value, "confidence": confidence})

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

