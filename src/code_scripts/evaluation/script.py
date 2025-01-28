# | filename: script.py
# | code-line-numbers: true

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score, roc_curve 
import xgboost as xgb



def evaluate(model_path, test_path, output_path):
    """
    Model loadin, evaluation and generation of metrics report
    """
        
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    # Extract the model and load it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))
        
    # # LOAD AS TXT
    # model = xgb.XGBClassifier()
    # model.load_model(Path(model_path) / "model.txt") 
    # # predictions = np.argmax(model.predict(X_test), axis=-1)
    # predictions = model.predict(X_test)
    # # Make predictions for model_logloss
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    
    # LOAD AS BOOSTER
    model_filepath = Path(model_path) / "xgbclass"
    model = xgb.Booster()
    model.load_model(model_filepath)
    # booster.set_param("nthread", 1)
    # model.load_model(model_filepath.as_posix())
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    
    # # LOAD AS PICKLE BOOSTER
    # import pickle
       
    # model_filepath = Path(model_path) / "model.pkl"
    # with open(model_filepath, 'rb') as f:
    #     model = xgb.Booster()
    #     model = pickle.load(f)
        
    # dtest = xgb.DMatrix(X_test)
    # y_pred_proba = model.predict(dtest)    
 

    # # LOAD AS PICKLE 
    # import pickle
    # model_filepath = Path(model_path) / "model.pkl"

    # with open(model_filepath, 'rb') as f:
    #     model = pickle.load(f)
        
    # predictions = model.predict(X_test)
    # # Make predictions for model_logloss
    # y_pred_proba = model.predict_proba(X_test)[:, 1]  
    
    
    # Evaluate the model with the custom threshold
    roc_auc = roc_auc_score(y_test, y_pred_proba)     
    print(f'auc: {roc_auc}') 
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Find the threshold that gives the best trade-off
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = float(thresholds[optimal_idx])
    
    # Evaluate the model with the custom threshold
    y_pred_custom_threshold = (y_pred_proba >= best_threshold).astype(int)
    precision_custom = precision_score(y_test, y_pred_custom_threshold)
    recall_custom = recall_score(y_test, y_pred_custom_threshold)
    f1_custom = f1_score(y_test, y_pred_custom_threshold)
    
    # Log in the expected format
    print(f'best_threshold: {best_threshold}')  
    print(f'auc: {roc_auc}')  
    print(f'precision (custom threshold): {precision_custom}')  
    print(f'recall (custom threshold): {recall_custom}')  
    print(f'f1 (custom threshold): {f1_custom}')  


    # Eevaluation report
    evaluation_report = {
        "metrics": {
            "auc": {"value": roc_auc},
            "best_threshold": {"value": best_threshold},
            "precision_custom": {"value": precision_custom},
            "recall_custom": {"value": recall_custom},
            "f1_custom": {"value": f1_custom},
        },
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path="/opt/ml/processing/model/",
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/",
    )
