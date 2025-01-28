# | filename: script.py
# | code-line-numbers: true

import os
import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess(base_directory):
    """
    Load, split, and transform the data.
    """
    
    df = _read_data_from_input_csv_files(base_directory)
        
    X_cat = df.drop(columns = ["stroke"]).select_dtypes(include=['object'])
    cat_features = X_cat.columns.to_list()

    categorical_transformer = make_pipeline(
                                    OneHotEncoder())
    
    features_transformer = ColumnTransformer(
                                    transformers=[
                                        ("categorical", 
                                        categorical_transformer, 
                                        cat_features),
                                    ],
                                    remainder='passthrough')
    

    
    df_train, df_validation, df_test = _split_data(df)
    
    _save_train_baseline(base_directory, df_train)
    _save_test_baseline(base_directory, df_test)

    y_test = df_test.pop("stroke").values
    y_train = df_train.pop("stroke").values
    y_validation = df_validation.pop("stroke").values

    X_train = features_transformer.fit_transform(df_train)  
    X_validation = features_transformer.transform(df_validation) 
    X_test = features_transformer.transform(df_test)  

    feature_names = features_transformer.get_feature_names_out().tolist()

    _save_splits(
            base_directory,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            feature_names)
    
    _save_model(base_directory, features_transformer)
    

def _read_data_from_input_csv_files(base_directory):
    """
    Read the data from the input CSV files.

    This function reads every CSV file available and
    concatenates them into a single dataframe.
    """
    
    input_directory = Path(base_directory) / "input"
    files = list(input_directory.glob("*.csv"))

    if len(files) == 0:
        message = f"The are no CSV files in {input_directory.as_posix()}/"
        raise ValueError(message)

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    return df.sample(frac=1, random_state=42)   


def _split_data(df):
    """
    Split the data into train, validation, and test.
    """
    
    stratify_column = df['stroke']
    df_train, temp = train_test_split(df, test_size=0.3, stratify=stratify_column, random_state=42)
    df_validation, df_test = train_test_split(temp, test_size=0.5, stratify=temp['stroke'], random_state=42)

    return df_train, df_validation, df_test


def _save_train_baseline(base_directory, df_train):
    """
    Save the untransformed training data to disk.
    Determines the baseline for the model    
    """
    
    baseline_path = Path(base_directory) / "train-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_train.copy()

    df = df.drop("stroke", axis=1)

    df.to_csv(baseline_path / "train-baseline.csv", header=True, index=False)

    
def _save_test_baseline(base_directory, df_test):
    """Save the untransformed test data to disk.

    We will need the test data to compute a baseline to
    determine the quality of the model predictions when deployed.
    """
    
    baseline_path = Path(base_directory) / "test-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_test.copy()

    df.to_csv(baseline_path / "test-baseline.csv", header=False, index=False)


def _save_splits(base_directory,
                 X_train,
                 y_train,
                 X_validation,  
                 y_validation,
                 X_test,
                 y_test,
                 feature_names):
    
    """Save data splits to disk.

    This function concatenates the transformed features
    and the target variable, and saves each one of the split
    sets to disk.
    """
    
    train = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    validation = np.concatenate((X_validation, y_validation.reshape(-1, 1)), axis=1)
    test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    
    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
      
    pd.DataFrame(train, columns=feature_names + ['stroke']).to_csv(train_path / "train.csv", header=True, index=False)
    pd.DataFrame(validation, columns=feature_names + ['stroke']).to_csv(validation_path / "validation.csv", header=True, index=False)
    pd.DataFrame(test, columns=feature_names + ['stroke']).to_csv(test_path / "test.csv", header=True, index=False)



def _save_model(base_directory, features_transformer):
    """Save the Scikit-Learn transformation pipelines.

    This function creates a model.tar.gz file that
    contains the two transformation pipelines we built
    to transform the data.
    """
    
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(features_transformer, Path(directory) / "features.joblib")

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{(model_path / 'model.tar.gz').as_posix()}", "w:gz") as tar:
            tar.add(Path(directory) / "features.joblib", arcname="features.joblib",)


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
