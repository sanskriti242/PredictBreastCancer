# Breast Cancer Detection using Machine Learning

This repository contains a breast cancer detection model using machine learning algorithms, including Logistic Regression, Decision Tree, and Random Forest. The model is trained on a preprocessed dataset and includes exploratory data analysis (EDA) to gain insights into the data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)

## Introduction

Breast cancer is one of the most common types of cancer affecting millions of people worldwide. Early detection of breast cancer can significantly improve the chances of successful treatment and survival. Machine learning algorithms can be used to build predictive models that can help in the early detection of breast cancer.

In this repository, we have developed a breast cancer detection model using three different machine learning algorithms: Logistic Regression, Decision Tree, and Random Forest. We have also performed preprocessing and exploratory data analysis (EDA) to gain insights into the data and improve the performance of our models.

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>), which contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. The dataset has 569 instances and 32 attributes, including the diagnosis (M = malignant, B = benign) and 30 real-valued input features.

## Preprocessing

Before training the machine learning models, we performed the following preprocessing steps:

1. Load the dataset and remove unnecessary columns (e.g., ID).
2. Encode the target variable (diagnosis) as binary (0 = benign, 1 = malignant).
3. Split the dataset into training (80%) and testing (20%) sets.
4. Standardize the feature values to have zero mean and unit variance.

## Exploratory Data Analysis

We performed exploratory data analysis (EDA) to gain insights into the data and identify patterns, trends, and relationships between variables. The EDA included the following steps:

1. Visualize the distribution of the target variable (diagnosis) to check for class imbalance.
2. Analyze the correlation between features using a heatmap.
3. Visualize the distribution of features using histograms and box plots.

## Model Training and Evaluation

We trained and evaluated three different machine learning models for breast cancer detection:

1. **Logistic Regression**: A linear model for binary classification that estimates the probability of an instance belonging to a particular class.
   - Accuracy: 96.49%
2. **Decision Tree**: A non-linear model that recursively splits the data into subsets based on the values of input features, resulting in a tree-like structure.
   - Accuracy: 91.22%
3. **Random Forest**: An ensemble method that constructs multiple decision trees and combines their predictions to improve the overall performance and reduce overfitting.
   - Accuracy: 99.12%

For each model, we performed the following steps:

1. Train the model on the preprocessed training dataset.
2. Make predictions on the testing dataset.
3. Evaluate the model's performance using accuracy score, classification report, and confusion matrix.

## Conclusion

In this project, we developed a breast cancer detection model using Logistic Regression, Decision Tree, and Random Forest algorithms. We performed preprocessing and exploratory data analysis to gain insights into the data and improve the performance of our models. The trained models can be used to predict the presence of breast cancer in new instances, potentially aiding in early detection and improving treatment outcomes.
