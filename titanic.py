import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load


st.title("Titanic Dataset")

pClass = st.number_input("Enter Pclass (1, 2, 3)")
sex = st.radio("Female/Male", [0, 1])
age = st.number_input("Enter Age")
sibsp = st.number_input("Enter Sibsp (0, 1, 2, 3)")
parch = st.number_input("Enter Parch (0, 1, 2)")
fare = st.number_input("Enter Fare")

# https://titanic-prediction.streamlit.app/

clicked = st.button("Get Prediction")

if clicked:
    data = [pClass, sex, age, parch, fare]

    model = load("model.joblib")

    if model.predict(data) == 0:
        print("Not Survived")
    else:
        print("Survived")
