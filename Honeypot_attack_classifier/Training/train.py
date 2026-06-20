import joblib 
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Path for scratch models
sys.path.append(r"C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\Models_metrics\models\Classical")

# Sklearn models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Library models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Scratch models

from KNN import KNN
from Naive_bayes import NaiveBayes  
from Decision_tree import DecisionTree
from SVM import MulticlassSVM
from Logistic_regression import LogisticRegression

#  Pytorch models
# import torch


# Load data
X = joblib.load(r'C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\X_transformed.pkl')
y = joblib.load(r'C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\y_encoded.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Folders to save metrics
os.makedirs(r'Models_metrics\models\classical', exist_ok=True)
os.makedirs(r'Models_metrics\metrics\classical', exist_ok=True)

# os.makedirs(r'Models_metrics\models\Classical', exist_ok=True)
# os.makedirs(r'Models_metrics\models\Deep_learning',exist_ok=True)
# os.makedirs(r'Models_metrics\metrics\Classical_metric', exist_ok=True)
# os.makedirs(r'Models_metrics\metrics\Deep_learning_metrics',exist_ok=True)

# Save function (Classical)
def save_model_metrics(name, model, y_test, y_pred):
    joblib.dump(model, f'Models_metrics/models/classical/{name}.pkl')
        
    metrics ={
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()

    }

    with open(f"Models_metrics/metrics/classical/{name}.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"{name} accuracy: {metrics['accuracy']: .4f}")
    print("Training complete")

def cross_validation_model(name, model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV: {scores.mean():.4f} ± {scores.std():.4f}")


# Save function (Deep learning)
# def save_dl_model_metrics(name, model, y_test, y_pred):
#     torch.save(model.state_dict(), f"Models_metrics/models/deep_learning/{name}.pt")
#     metrics = {
#         'accuracy' : accuracy_score(y_test, y_pred),
#         'classification_report' : classification_report(y_test, y_pred),
#         'confusion_matrix' : confusion_matrix(y_test, y_pred).tolist()
#     }

print("Training Classical Models")

#  Classical_models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cross_validation_model('random_forest', rf, X, y)
rf.fit(X_train, y_train)
save_model_metrics('random_forest', rf, y_test, rf.predict(X_test))


# XGBoost
xgb = XGBClassifier(random_state = 42, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
save_model_metrics('xgboost', xgb, y_test, xgb.predict(X_test))


# # LightGBM
# lgb = LGBMClassifier(random_state=42)
# lgb.fit(X_train, y_train)
# save_model_metrics('lightgbm', lgb, y_test, lgb.predict(X_test))


# Adaboost
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
save_model_metrics('Adaboost', ada, y_test, ada.predict(X_test))


# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
save_model_metrics('gradient_boosting', gb, y_test, gb.predict(X_test))


# SVM (scratch)
svm = MulticlassSVM()
svm.fit(X_train, y_train)
save_model_metrics('svm', svm, y_test, svm.predict(X_test))


# Logistic regression (scratch)
lr = LogisticRegression()
lr.fit(X_train, y_train)
save_model_metrics('logistic_regression', lr, y_test, lr.predict(X_test))


# KNN (scratch)
knn = KNN(k=5)
knn.fit(X_train, y_train)
save_model_metrics('KNN', knn, y_test, knn.predict(X_test))


# Decision Tree (scratch)
dt = DecisionTree()
dt.fit(X_train, y_train)
save_model_metrics('Decsion_tree', dt, y_test, dt.predict(X_test))


# Naive Bayes (scratch)
nb = NaiveBayes()
nb.fit(X_train, y_train)
save_model_metrics('Naive_Bayes', nb, y_test, nb.predict(X_test))
