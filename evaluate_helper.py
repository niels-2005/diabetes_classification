import numpy as np 
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


def calculate_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"):
    target_names = ["No Diabetes", "Diabetes"]
    report = classification_report(y_pred=y_pred, y_true=y_true, target_names=target_names, output_dict=True)   

    acc = accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average=average)
    precision = precision_score(y_pred=y_pred, y_true=y_true, average=average)
    recall = recall_score(y_pred=y_pred, y_true=y_true, average=average)

    dict = {
        "name": name,
        "accuracy": [acc],
        f"f1-score_{average}": [f1],
        f"precision_{average}": [precision],
        f"recall_{average}": [recall],
        target_names[0]: target_names[0],
        "precision_no_dia": [report[target_names[0]]["precision"]],
        "recall_no_dia": [report[target_names[0]]["recall"]],
        "f1-score_no_dia": [report[target_names[0]]["f1-score"]],
        target_names[1]: target_names[1],
        "precision_dia": [report[target_names[1]]["precision"]],
        "recall_dia": [report[target_names[1]]["recall"]],
        "f1-score_dia": [report[target_names[1]]["f1-score"]],
    }

    df_metrics = pd.DataFrame(dict)
    return df_metrics
