import joblib 
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
import sys


# Add the project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Path for scratch models 
# sys.path.append(r"C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\Models_metrics\models\Classical")

# Sklearn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Library models
from xgboost import XGBClassifier

# Scratch models

# from KNN import KNN
# from Naive_bayes import NaiveBayes  
# from Decision_tree import DecisionTree
# from SVM import MulticlassSVM
# from Logistic_regression import LogisticRegression

# Import data cleaning
sys.path.append(r"C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\Data_preprocessing")

from data_cleaning import get_preprocessor

# Load data
def load_data():
    BASE = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics'
    X = joblib.load(rf'{BASE}\X_cleaned.pkl')
    y_multi = joblib.load(rf'{BASE}\y_multiclass.pkl')
    y_binary = joblib.load(rf'{BASE}\y_binary.pkl')
    le_multi = joblib.load(rf'{BASE}\label_encoder_multi.pkl')
    le_binary = joblib.load(rf'{BASE}\label_encoder_binary.pkl')
    print(f"Loaded X: {X.shape}, y_multi:{y_multi.shape}, y_binary{y_binary.shape}")
    return X, y_multi, y_binary, le_multi, le_binary

def balance_train_only(X_train, ym_train, yb_train, le_multi, le_binary, min_samples=2000):
    from Data_preprocessing.data_cleaning import balance_classes
    df = X_train.copy()
    df['label'] = le_multi.inverse_transform(ym_train)
    df['binary_label'] = le_binary.inverse_transform(yb_train)
    df = balance_classes(df, label_col='label', min_samples=min_samples)
    y_multi_bal= le_multi.transform(df['label'])
    y_binary_bal= le_binary.transform(df['binary_label'])
    X_bal = df.drop(columns=['label', 'binary_label'])
    return X_bal, y_multi_bal, y_binary_bal

def split_data(X, y_multi, y_binary):
    X_train, X_test, ym_train, ym_test, yb_train, yb_test = train_test_split(
        X, y_multi, y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_multi
    )
    print(f"Train:{X_train.shape} Test: {X_test.shape}")
    return X_train, X_test, ym_train, ym_test, yb_train, yb_test

def fit_preprocessor(X_train, y_multi_train):
    preprocessor = get_preprocessor()
    X_train_transformed= preprocessor.fit_transform(X_train, y_multi_train)

    save_path= r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\preprocessor.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    print(f"X_train_transformed shape: {X_train_transformed.shape}")
    return preprocessor, X_train_transformed


def train_binary_model(X_train_transformed, y_binary_train):
    model = GradientBoostingClassifier(
        n_estimators=170,
        learning_rate=0.09,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        verbose=1,    
    )


    sample_weights = compute_sample_weight('balanced', y_binary_train)
    model.fit(X_train_transformed, y_binary_train, sample_weight=sample_weights)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\binary_models.pkl')
    print("Binary model saved")
    return model

if __name__=='__main__':
    X, y_multi, y_binary, le_multi, le_binary = load_data()
    X_train, X_test, ym_train, ym_test, yb_train, yb_test = split_data(X, y_multi, y_binary)

    X_train, ym_train, yb_train = balance_train_only(X_train, ym_train, yb_train, le_multi, le_binary)

    preprocessor, X_train_transformed = fit_preprocessor(X_train, ym_train)
    X_test_transformed = preprocessor.transform(X_test)
    X_scratch = X_test_transformed[:600]
    y_scratch = ym_test[:600]

    # Training binary model
    binary_model = train_binary_model(X_train_transformed, yb_train)

    # Multiclass models
    LIBRARY= {
        'xgboost' : XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=170, max_depth=4, random_state=42, subsample=0.8, learning_rate=0.09)
    }
    sample_weights_multi = compute_sample_weight('balanced', ym_train)

    for name, model in LIBRARY.items():
        print(f"Training {name}..")
        if name == 'gradient_boosting':
            model.fit(X_train_transformed, ym_train, sample_weight=sample_weights_multi)
        else:
            model.fit(X_train_transformed, ym_train, sample_weight= sample_weights_multi)
        joblib.dump(model, rf'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\{name}_model.pkl')

    # # Scratch models
    # SCRATCH={
    #     'knn' : KNN(k=5),
    #     'naive_bayes': NaiveBayes(),
    #     'decision_tree': DecisionTree(),
    #     'svm': MulticlassSVM(),
    #     'logreg': LogisticRegression()
    # }
    # for name, model in SCRATCH.items():
    #     print(f"Training scratch {name}..")
    #     model.fit(X_scratch, y_scratch)
    #     joblib.dump(model, rf'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\{name}_model.pkl')
    #     print(f"Saved {name}")

    joblib.dump((X_test, ym_test, yb_test), r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\test_split.pkl')
    print("Test split saved= models/test_split.pkl")
import joblib, pandas as pd

X = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\X_cleaned.pkl')
y_multi = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\y_multiclass.pkl')
le_multi = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\label_encoder_multi.pkl')

labels = le_multi.inverse_transform(y_multi)
brute = X[labels == 'brute_force']

print(brute[['ttl','window_size','payload_size','dst_port','src_port',
             'hour','minute','day_of_week','ip_request_rate',
             'is_exploit_path','is_known_scanner']].describe())
