import pandas as pd
import joblib

df = joblib.load(r"Models_metrics/y_binary.pkl")   # change path
print(len(df))