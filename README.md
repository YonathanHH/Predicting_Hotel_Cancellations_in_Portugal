# Hotel Booking Cancellation Prediction - Capstone Project Module 3

## Overview

This capstone project develops a machine learning model to predict whether customers are likely to cancel their hotel bookings at a Portuguese hotel. The project implements an end-to-end machine learning workflow, from data exploration and preprocessing to model development, evaluation, and deployment through an interactive Streamlit application.

**Target Problem:** Binary classification to identify booking cancellations before they occur, enabling the hotel to implement proactive retention strategies and optimize resource allocation.

---

## Project Objectives

- **Business Goal:** Reduce booking cancellation losses through predictive analytics and early intervention strategies
- **Technical Goal:** Develop a robust classification model with high predictive accuracy and explainability
- **Stakeholders:** Hotel management, revenue management team, customer service department
- **Impact:** Measurable improvement in booking revenue and operational efficiency

---

## Dataset Information

- **Source:** Hotel booking demand data from Portugal
- **Records:** Customer booking information with anonymized personal details

`Target`: 
is canceled (Whether it is canceling or not)

`Features`:
- Market Segment 
- Customer type 
- Local or International Tourist 
- Booking Changes
- Previous Cancellations 
- Parking space requirement
- Special Request 
- Waiting List

## Project Structure

```
capstone-project-3-hotel-cancellation/
│
├── README.md                                    # Project documentation
├── Hotel_Cancelation_End_to_End_ML.ipynb        # Jupyter notebook with complete analysis
├── final_model.sav                              # Trained ML model (pickle format)
├── app.py                                       # Streamlit web application
├── CapstoneModule3PPT.pdf                       # Presentation slides
├── Hotel_booking_demand.csv                     # Original raw dataset
└── hotel_dataset_cleaned.csv                    # Cleaned and transformed data
```

## Prerequisites
- Python 3.8+ (used Python 3.13.9 Kernel)
- pip or conda package manager

## Final Model
- **Algorithm:** XGBoost
- **Learning Rate:** 0.27
- **Max Depth:** 12
- **n_estimator:** 185

## Running Jupyter Notebook
It is advised to use VS Code to run the code

## Model deployment testing
```
import pandas as pd
import pickle

df = pd.from_csv("hotel_dataset_cleaned.csv")
pipe = pickle.load(open("final_model.sav", "rb")) ### Openning the data
```
testing example
```
print('predict class :',pipe.predict(df[51:55]))
print('predict proba :',pipe.predict_proba(df[51:55]))
```
## Model Performance

- **Accuracy:** 0.801
- **Precision:** 0.750
- **Recall:** 0.693
- **F1-Score:** 0.720
- **ROC-AUC:** 0.876

## Required Libraries
- Matplotlib 3.0.2
- Numpy 2.3.5
- Pandas 2.3.3
- Sklearn (ScikitLearn) 1.7.2
- Shap 0.50.0
- Seaborn 0.13.2
- catboost 1.2.8
- lightgbm 4.6.0
- xgboost 3.1.2
- pickle (built-in)
- imbalanced-learn 0.14.0

## Conclusions

This machine learning project successfully developed a predictive model for hotel booking cancellations, enabling data-driven decision-making for revenue optimization. The model demonstrates strong performance on evaluation metrics and provides actionable insights for hotel management. With proper implementation and continuous monitoring, this model can significantly improve booking revenue and operational efficiency.

## Project Metadata

- **Project Type:** ML Capstone Project - Machine Learning
- **Problem Type:** Binary Classification
- **Model Type:** Extreme Gradient Boost (XGB)
- **Created:** December 2025
- **Author:** Yonathan Hary Hutagalung
- **Institution:** Purwadhika Digital Technology School


## References & Resources

- Dataset: Hotel Booking Demand (Portugal)
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- Project Guidelines: Capstone Project Module 3 - Machine Learning Problem

