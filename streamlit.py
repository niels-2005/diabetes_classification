from joblib import load 
import pandas as pd
import streamlit as st

def load_pipeline():
    pipeline = load("final_model.pkl")
    return pipeline


smoking_history_mapping = {
    'No Info': 0,
    'Current': 1,
    'Ever': 2,
    'Former': 3,
    'Never': 4,
    'Not current': 5
}


age_bin_mapping = {
    'medium': 0, 
    'old': 1, 
    'very young': 2, 
    'young': 3}


bins = [0, 6, 21, 55, 120]
labels = ["very young", "young", "medium", "old"]

def get_age_bin(age: int):
    for i in range(len(bins) - 1):
        if bins[i] <= age < bins[i + 1]:
            return age_bin_mapping[labels[i]]


def predict_diabetes(df: pd.DataFrame):
    pipeline = load_pipeline()
    prediction = pipeline.predict(df)

    if prediction[0] == 0:
        pred = "No Diabetes"
    else:
        pred = "Diabetes"

    return pred


def main():
    st.title("Welcome to Diabetes Classification")
    st.header("Please enter your Details!")
    Age = st.number_input("Age")
    Blood_Glucose = st.number_input("Blood Glucose Level")
    BMI = st.number_input("BMI")
    HbA1c_level = st.number_input("HbA1c Level")
    Smoking_History = st.selectbox("Smoking History", ("No Info", "Current", "Ever", "Former", "Never", "Not Current"))
    Hypertension = st.radio('Hypertension?', ['Yes','No'], horizontal=True)

    if st.button("Predict"):
        smoking_int = smoking_history_mapping.get(Smoking_History)
        age_bin = get_age_bin(Age)

        if Hypertension == "Yes":
            hypertension = 1
        else:
            hypertension = 0

        data = {
            "age": Age,
            "blood_glucose_level": Blood_Glucose,
            "bmi": BMI,
            "HbA1c_level": HbA1c_level,
            "smoking_history": smoking_int,
            "hypertension": hypertension,
            "ages_bins": age_bin
        }

        df = pd.DataFrame([data])

        pred = predict_diabetes(df=df)

        if pred == "No Diabetes":
            st.success("Predicted: No Diabetes")
        else:
            st.error("Predicted: Diabetes")

        
if __name__ == "__main__":
    main()