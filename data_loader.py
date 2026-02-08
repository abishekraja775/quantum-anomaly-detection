import pandas as pd

def load_creditcard_data(path):
    df = pd.read_csv(path)

    normal = df[df["Class"] == 0].drop(columns=["Class"])
    anomaly = df[df["Class"] == 1].drop(columns=["Class"])

    return normal, anomaly
