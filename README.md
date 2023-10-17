# Isbankasi Machine Learning Project

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Details](#code-details)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to the Isbankasi Machine Learning project repository. This project is designed for machine learning practitioners and data scientists who are interested in working with Isbankasi data. The repository includes code for various data preprocessing tasks, data visualization, model optimization, and multi-label classification.

In this README, we provide an overview of the code and its functionalities.

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

   ```bash
   git clone https://github.com/ahmetdzdrr/Multi-Label-Classification.git

2. Install the required Python libraries by running:
   ```bash
    pip install -r requirements.txt

## Usage

Usage
To run the project, follow these steps:

- To use this code, you can customize its behavior by modifying the CFG class in the main script (multi_label_classification.ipynb). 

- Each flag in the CFG class controls whether a specific functionality is enabled or disabled. Set the flags to True or False based on your requirements.

- Open the Jupyter Notebook file (multi_label_classification.ipynb) in your Jupyter Notebook environment.

- Run each cell in the notebook sequentially. The code in the notebook will process the data, perform the selected operations, and generate the desired output.
