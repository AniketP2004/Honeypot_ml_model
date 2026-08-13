import joblib

PATHS = {
    "random_forest": r"Models_metrics\metrics\random_forest_model.pkl",
    "xgboost": r"Models_metrics\metrics\xgboost_model.pkl",
    "gradient_boosting": r"Models_metrics\metrics\gradient_boosting_model.pkl",
    "binary": r"Models_metrics\metrics\binary_models.pkl",
    "preprocessor": r"Models_metrics\metrics\preprocessor.pkl",
    "le_multi": r"Models_metrics\metrics\label_encoder_multi.pkl",
    "le_binary": r"Models_metrics\metrics\label_encoder_binary.pkl"
}

REG = {}

def load_all():
    for k, p in PATHS.items():
        REG[k] = joblib.load(p)

def reload():
    load_all()