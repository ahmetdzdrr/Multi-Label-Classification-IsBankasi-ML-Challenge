# Isbankasi Machine Learning Project

## Table of Contents

- [Introduction](#introduction)
- [What is Multi-Label Classification?](#what-is-multi-label-classification)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage Code](#usage)
- [How to Use Optuna?](#how-to-use-optuna)


## Introduction

Welcome to the Isbankasi Machine Learning project repository. This project is designed for machine learning practitioners and data scientists who are interested in working with Isbankasi data. The repository includes code for various data preprocessing tasks, data visualization, model optimization, and multi-label classification.

In this README, we provide an overview of the code and its functionalities.

## What is Multi-Label Classification?

### Classifier Chain (CC)

- **Description**: Classifier Chain is a multi-label classification technique where a set of binary classifiers is created, one for each label in the dataset.
  
- **Methodology**: Each binary classifier predicts the presence or absence of its corresponding label. The order of the labels matters, as the prediction of one label can influence the prediction of subsequent labels in the chain.
  
- **Use Cases**: CC is effective when there are label dependencies or correlations in the dataset.

## Label Powerset (LP)

- **Description**: Label Powerset is a multi-label classification approach that transforms the problem into a multi-class problem by creating a unique class for each possible combination of labels.
  
- **Methodology**: LP simplifies the multi-label problem but can lead to an exponential increase in the number of classes when dealing with many labels. It's suitable when labels are not highly correlated, and it treats each label combination as a distinct class.
  
- **Use Cases**: LP is used when dealing with a wide range of label combinations, and the labels are not strongly correlated.

## Binary Relevance (BR)

- **Description**: Binary Relevance is a straightforward approach to multi-label classification where a separate binary classifier is trained for each label independently.
  
- **Methodology**: Each classifier predicts the presence or absence of one label without considering the other labels. BR doesn't capture label dependencies or correlations but is computationally efficient and easy to implement.
  
- **Use Cases**: BR works well when labels are mostly independent, and computational efficiency is a priority.


## Project Overview

### Features

The project comprises several key features:

- **Data Quality Checks**: The code includes functions to check data quality, such as identifying missing values, duplicated rows, and the number of unique values in each column. These checks help ensure the integrity of the data.

- **Data Visualization**: Visualizing data can be crucial for understanding its characteristics. The code offers options for creating box plots, histograms, and visualizations of the target variable.

- **Data Preprocessing**: The repository provides various data preprocessing steps, including removing carriers, feature scaling using Min-Max scaling, extracting time components, generating additional features, binarizing the target variable, and applying anomaly detection techniques like Gaussian Mixture and Isolation Forest.

- **Model Optimization with Optuna**: If you want to optimize your models, you can enable the Optuna functionality. The code supports optimizing XGBoost, LightGBM, and CatBoost models.

- **Multi-Label Classification Models**: Multi-label classification is available, with options for Classifier Chains, Label Powerset, and Binary Relevance. You can use XGBoost, LightGBM, and CatBoost for these multi-label classification tasks.

- **Cross-Validation and Test Prediction**: Cross-validation is included, allowing you to assess your model's performance. Additionally, you can make predictions on the test dataset.

## Installation

To get started with this project, follow these steps:

1. Clone this repository to your local machine using Git:

         git clone https://github.com/ahmetdzdrr/Multi-Label-Classification.git

2. Install the required Python libraries by running:

          pip install -r requirements.txt

## Usage Code

Usage
To run the project, follow these steps:

- To use this code, you can customize its behavior by modifying the CFG class in the main script (multi_label_classification.ipynb). 

- Each flag in the CFG class controls whether a specific functionality is enabled or disabled. Set the flags to True or False based on your requirements.

- Open the Jupyter Notebook file (multi_label_classification.ipynb) in your Jupyter Notebook environment.

- Run each cell in the notebook sequentially. The code in the notebook will process the data, perform the selected operations, and generate the desired output.

## How to Use Optuna?

- Optuna is a Python library for optimizing machine learning model hyperparameters. You can use it with various machine learning frameworks, including XGBoost, LightGBM (LGBM), and CatBoost. Here's a short guide on how to do that:

### Step 1: Install Optuna

- You need to install Optuna in your Python environment. You can do this using pip:

       pip install optuna

### Step 2: Import Optuna and the Machine Learning Library

- In your Jupyter Notebook or Python script, import Optuna and the machine learning library you want to optimize (e.g., XGBoost, LGBM, or CatBoost).

      import optuna
      import xgboost as xgb
      import lightgbm as lgb
      from catboost import CatBoostClassifier

### Step 3: Define an Objective Function

- Create an objective function that Optuna will optimize. This function takes an Optuna trial object as an argument and returns a score that you want to minimize or maximize. This score is typically a metric of your model's performance.

Here's an example for optimizing the AUC score with XGBoost:

      def objective(trial):
          params = {
              "objective": "binary:logistic",
              "eval_metric": "auc",
              "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
              "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
              # Add more hyperparameters to tune
          }
      
       dtrain = xgb.DMatrix(X_train, label=y_train)
       model = xgb.train(params, dtrain)
       predictions = model.predict(dtest)
      
       auc = sklearn.metrics.roc_auc_score(y_test, predictions)
       return auc
       
### Step 4: Create an Optuna Study

- Create an Optuna study to run the optimization. You can specify the optimization direction, e.g., "maximize" or "minimize" based on your chosen objective (e.g., AUC).

    ```bash
      study = optuna.create_study(direction="maximize")

### Step 5: Start the Optimization

-Run the optimization process with a specified number of trials. Optuna will search for the best hyperparameters based on the objective function.
    
      study.optimize(objective, n_trials=100)
      
### Step 6: Retrieve the Best Parameters

- Once the optimization is complete, you can retrieve the best set of hyperparameters from the study object:

      best_params = study.best_params

- You can then use these best hyperparameters to train your final model with XGBoost, LGBM, or CatBoost.

#### Additional Notes

- Make sure to adapt the objective function and hyperparameters to your specific machine learning problem and dataset.
  
- The same steps can be applied for optimizing hyperparameters with LightGBM and CatBoost, replacing the relevant library and objective function accordingly.

- Optuna is a powerful tool for automating hyperparameter tuning, and it can significantly improve the performance of your machine learning models.
