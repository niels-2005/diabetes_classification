# What we need? age: int, blood_glucose_level: int, bmi, HbA1c_level: float, smoking_history: int, hypertension: int, ages_bins: int 

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load 
import pandas as pd


def load_pipeline():
    pipeline = load("final_model.pkl")
    return pipeline


app = FastAPI()


class DiabetesPred(BaseModel):
    age: int 
    blood_glucose_level: int
    bmi: float 
    HbA1c_level: float
    smoking_history: str 
    hypertension: int


smoking_history_mapping = {
    'No Info': 0,
    'current': 1,
    'ever': 2,
    'former': 3,
    'never': 4,
    'not current': 5
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


@app.post("/predict")
def predict_diabetes(details: DiabetesPred):
    data = details.model_dump()
    pipeline = load_pipeline()
    smoking_int = smoking_history_mapping.get(data["smoking_history"])
    age_bin = get_age_bin(data["age"])

    new_data = {
        "age": data["age"],
        "blood_glucose_level": data["blood_glucose_level"],
        "bmi": data["bmi"],
        "HbA1c_level": data["HbA1c_level"],
        "smoking_history": smoking_int,
        "hypertension": data["hypertension"],
        "ages_bins": age_bin
    }

    df = pd.DataFrame([new_data])

    prediction = pipeline.predict(df)

    if prediction[0] == 0:
        pred = "No Diabetes"
    else:
        pred = "Diabetes"

    return {"pred": pred}



