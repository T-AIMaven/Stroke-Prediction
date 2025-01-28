# | filename: script.py
# | code-line-numbers: true

import argparse
import json
import os
import tarfile

from pathlib import Path
from comet_ml import Experiment

import pandas as pd
import xgboost as xgb
from packaging import version
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def train(
    model_directory,
    train_path,
    validation_path,
    pipeline_path,
    experiment,
    eta=0.3):
    """
    Train the model, generate metrics, log in comet and save the model
    """

    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)
        
    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation = X_validation.drop(X_validation.columns[-1], axis=1)
    
    
    model = xgb.XGBClassifier(objective='binary:logistic', 
                              eval_metric='auc', 
                              eta=eta,
                              nthread=1)
                               
    model.fit(X_train, 
              y_train,
              eval_set=[(X_validation, y_validation)],
              early_stopping_rounds=10
              )
    
    # Predictions
    predictions = model.predict(X_validation)
    y_pred_proba = model.predict_proba(X_validation)[:, 1]

    # Evaluate the model 
    roc_auc= roc_auc_score(y_validation, y_pred_proba)
    precision = precision_score(y_validation, predictions)
    recall = recall_score(y_validation, predictions)
    f1 = f1_score(y_validation, predictions)

    print(f'auc: {roc_auc}')  
    print(f'precision: {precision}')  
    print(f'recall: {recall}')  
    print(f'f1: {f1}')  

    # SAVE MODEL AS BOOSTER
    booster = model.get_booster()
    model_filepath = Path(model_directory) / "xgbclass"
    booster.save_model(model_filepath)
    # booster.save_model(model_filepath.as_posix())
    
    # # # SAVE MODEL AS TXT
    # model_filepath = Path(model_directory) / "model.txt"
    # model.save_model(model_filepath)

    # # # # SAVE MODEL AS PICKLE
    # import pickle
    # model_filepath = Path(model_directory) / "model.pkl"
    # with open(model_filepath, 'wb') as f:
    #     # pickle.dump(model.get_booster(), f)        
    #     pickle.dump(model, f)


  
    # Bundle transformation pipelines with model
    with tarfile.open(Path(pipeline_path) / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_directory)

    if experiment:
        experiment.log_parameters(
            {
                "eta": eta,
            })

        experiment.log_dataset_hash(X_train)
        
        experiment.log_confusion_matrix(
                                y_validation.astype(int), predictions.astype(int))
        
        experiment.log_model("stroke", model_filepath.as_posix())
        experiment.log_metric("roc_auc", roc_auc)
        experiment.log_metric("precision", precision)
        experiment.log_metric("recall", recall)
        experiment.log_metric("f1", f1)

        
if __name__ == "__main__":
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    # Create a Comet experiment to log the metrics and parameters
    comet_api_key = os.environ.get("COMET_API_KEY", None)
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", None)

    experiment = (
        Experiment(
            project_name=comet_project_name,
            api_key=comet_api_key,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        if comet_api_key and comet_project_name
        else None)

    training_env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))
    job_name = training_env.get("job_name", None) if training_env else None

    
    # SageMaker's training job name = experiment name
    if job_name and experiment:
        experiment.set_name(job_name)

    # SageMaker will create a model.tar.gz file with anything
    # inside this directory when the training script finishes.
    # SageMaker creates one channel for each one of the inputs to the Training Step.
    train(model_directory=os.environ["SM_MODEL_DIR"], 
          train_path=os.environ["SM_CHANNEL_TRAIN"],
          validation_path=os.environ["SM_CHANNEL_VALIDATION"],
          pipeline_path=os.environ["SM_CHANNEL_PIPELINE"],
          experiment=experiment,
          eta=args.eta)
