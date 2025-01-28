# MLOPS STROKE PREDICTION ⚱️

<p align="center">
  <img width="976" alt="aws" src="https://github.com/benitomartin/peft-gemma-2b/assets/116911431/ad464ac5-e4d0-4ed0-bb36-a1dbe4b9c613">
</p>

This is a personal MLOps project based on a [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) dataset for stroke prediction. 

Feel free to ⭐ and clone this repo 😉

## Tech Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Structure

The project has been structured with the following folders and files:

- `data:` raw and clean data
- `src:` source code. It is divided into:
    - Notebooks with EDA, Baseline Model and AWS Pipelines incl. unit testing
    - `code_scripts`: processing, training, evaluation, docker container, serving and lambda
- `requirements.txt:` project requirements

## Project Description

The dataset was obtained from Kaggle and contains 5110  rows and 10 columns to detect stroke predictions. To prepare the data for modelling, an **Exploratory Data Analysis** was conducted where it was detected that the dataset is very imbalance (95% no stroke, 5% stroke). For modeling, the categorical features where encoded, XGBoost was use das model and the best roc-auc threshold was selected for the predictions using aditionally threshold-moving for the predictions due to the imbalance. The learning rate was tuned in order to find the best one on the deployed model.

<p align="center">
    <img src="https://github.com/benitomartin/peft-gemma-2b/assets/116911431/f306a317-c7d7-470d-8adc-c36c7951d539">
</p>

## Pipeline Deployment

All pipelines where deployed on AWS SageMaker, as well as the Model Registry and Endpoints. The following pipelines where created:

- ✅ Preprocessing
- ✅ Training
- ✅ Tuning
- ✅ Evaluation
- ✅ Model Registry
- ✅ Model Conditional Registry
- ✅ Deployment

Additionally the experiments were tracked on Comel ML.

<p align="center">
    <img src="https://github.com/benitomartin/mlops-aws-stroke/assets/116911431/892ead03-7c8f-4e61-ad75-7abeee35e11b">
    </p>
