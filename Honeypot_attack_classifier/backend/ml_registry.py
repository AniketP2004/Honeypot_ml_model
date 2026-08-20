import os
import joblib

PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "Models_metrics")

PATHS= {
    "random_forest": os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    "xgboost": os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    "gradient_boosting": os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"),
    "binary": os.path.join(MODELS_DIR, "binary_models.pkl"),
    "preprocessor": os.path.join(MODELS_DIR, "preprocessor.pkl"),
    "le_multi": os.path.join(MODELS_DIR, "label_encoder_multi.pkl"),
    "le_binary": os.path.join(MODELS_DIR, "label_encoder_binary.pkl")
}

REG = {}

def load_all():
    for k, p in PATHS.items():
        REG[k] = joblib.load(p)

def reload():
    load_all()