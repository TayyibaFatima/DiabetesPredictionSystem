# Diabetes Prediction System using ANN

## Project Overview
This project is a Diabetes Prediction System built using Artificial Neural Networks (ANN) with TensorFlow/Keras and deployed using Streamlit.
The system predicts whether a person is likely to have diabetes based on medical input values such as glucose level, BMI, age, blood pressure, insulin level, etc.

## Dataset
Pima Indians Diabetes Dataset
Source:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Joblib
- 
## Features
- Data preprocessing and normalization
- Train/Test split
- ANN model using TensorFlow/Keras
- ReLU hidden layers
- Sigmoid output layer
- Model accuracy evaluation
- Saved trained model (.h5)
- Streamlit web application
- User-friendly prediction interface

## Project Structure
Diabetes_Prediction_Project
│
├── app.py
├── train_model.py
├── diabetes.csv
├── diabetes_model.h5
├── scaler.pkl
├── requirements.txt
└── README.md

## Model Architecture
- Input Layer
- Dense Hidden Layer (ReLU)
- Dense Hidden Layer (ReLU)
- Output Layer (Sigmoid)

## Input Features
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Prediction Output
The system predicts:
- Diabetes
or
- No Diabetes
It also displays prediction probability.


Developed as a Machine Learning and Streamlit deployment project.
