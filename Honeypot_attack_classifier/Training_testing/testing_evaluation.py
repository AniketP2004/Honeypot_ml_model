import joblib
import numpy as np
import json
import os
import sys
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ────────────────────────────────────────────────────
MODELS_DIR   = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics'
SCRATCH_DIR  = r'C:\Users\project\Desktop\Honeypot_New_Repo\Honeypot_attack_classifier\Models_metrics\models\Classical'
PREPROC_DIR  = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier'
PLOTS_DIR    = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Scratch models import
sys.path.append(SCRATCH_DIR)
from KNN import KNN
from Naive_bayes import NaiveBayes
from Decision_tree import DecisionTree
from SVM import MulticlassSVM
from Logistic_regression import LogisticRegression

# Sklearn models import
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Library models
LIBRARY_MODEL_KEYS = ['xgboost', 'random_forest', 'gradient_boosting']

# Scratch models
SCRATCH_MODEL_KEYS = ['knn', 'naive_bayes', 'decision_tree', 'svm', 'logreg']

# Loading models
def load_models():
    store = {}

    # preprocessor
    store['preprocessor'] = joblib.load(os.path.join(PREPROC_DIR, 'preprocessor.pkl'))

    # Label encoder
    store['le_multi'] = joblib.load(os.path.join(PREPROC_DIR, 'label_encoder_multi.pkl'))
    store['le_binary'] = joblib.load(os.path.join(PREPROC_DIR, 'label_encoder_binary.pkl'))

    # binary model
    store['binary_model'] = joblib.load(os.path.join(MODELS_DIR, 'binary_models.pkl'))


    # library multiclass models
    lib_files= {
          'xgboost'           : 'xgboost_model.pkl',
        'random_forest'     : 'random_forest_model.pkl',
        'gradient_boosting' : 'gradient_boosting_model.pkl'
    }

    for key, fname in lib_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            store[key] = joblib.load(path)
        else:
            print(f"[WARN] Missing: {fname}")
            store[key] = None
    
    # Scratch multiclass models
    scratch_files= {
        'knn'          : 'knn_model.pkl',
        'naive_bayes'  : 'naive_bayes_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'svm'          : 'svm_model.pkl',
        'logreg'       : 'logreg_model.pkl',
    }
    for key, fname in scratch_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            store[key] = joblib.load(path)
        else:
            print(f"[WARN] Missing scratch: {fname}")
            store[key] = None

    return store

def load_test_split():
    path = os.path.join(MODELS_DIR, 'test_split.pkl')
    X_test, ym_test, yb_test = joblib.load(path)
    print(f"Test split loaded: X= {X_test.shape}")
    return X_test, ym_test, yb_test

# confusion matrix
def _plot_confusion(cm, labels, title, fname):
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels)-1)))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved plots {path}")


# Evaluation 
def evaluate_binary(models, X_test_transformed, yb_test):
    model = models['binary_model']
    le = models['le_binary']
    
    y_pred = model.predict(X_test_transformed)

    accuracy = accuracy_score(yb_test, y_pred)
    f1 = f1_score(yb_test, y_pred, average='weighted')
    precision = precision_score(yb_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(yb_test, y_pred, average='weighted', zero_division=0)

    print("\nBinary Models")
    print(classification_report(
        yb_test, y_pred, target_names=le.classes_,
        zero_division=0))
    
    cm = confusion_matrix(yb_test, y_pred)
    _plot_confusion(cm, le.classes_, 'Binary Confusion Matrix', 'binary_confusion.png')
    
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

def evaluate_one_model(model, model_key, X_test_transformed, ym_test, le_multi):
    y_pred = model.predict(X_test_transformed)

    report = classification_report(
        ym_test, y_pred,
        target_names=le_multi.classes_,
        output_dict=True,
        zero_division=0
    )

    print(f"Multiclass: {model_key.upper()}")
    print(classification_report(ym_test, y_pred,
                                target_names=le_multi.classes_,
                                zero_division=0))
    
    cm = confusion_matrix(ym_test, y_pred)
    _plot_confusion(cm, le_multi.classes_, 
                    f'Confusion - {model_key}',
                    f'confusion_{model_key}.png')
    return{
        'accuracy': report['accuracy'],
        'f1_weighted':  report['weighted avg']['f1-score'],
        'per_class_f1': {cls: report[cls]['f1-score']
                         for cls in le_multi.classes_ if cls in report}
    }


def evaluate_all_models(models, X_test_transformed, ym_test):
    le_multi = models['le_multi']
    multi_metrics = {}

    # Library models
    print( "\n Library models")
    for key in LIBRARY_MODEL_KEYS:
        model = models.get(key)
        if model is None:
            print(f"Skip {key} not loaded")
            continue
        multi_metrics[key] = evaluate_one_model(
            model, key, X_test_transformed, ym_test, le_multi
        )

        # scratch models experiments
    print('\n Scratch_models (experiments)')
    SCRATCH_SUBSET = 500
    X_scratch = X_test_transformed[:SCRATCH_SUBSET]
    y_scratch = ym_test[: SCRATCH_SUBSET]


    for key in SCRATCH_MODEL_KEYS:
            model = models.get(key)
            if model is None:
                print(f"[SKIP] {key} not loaded")
                continue
            try:
                multi_metrics[key] = evaluate_one_model(
                    model, key, X_scratch, y_scratch, le_multi
                )
            except Exception as e:
                print(f"Error {key}: {e}")
                multi_metrics[key] = {'error': str(e)}
    return multi_metrics
        
def cross_validate_models(models, X_train, ym_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    print("Cross Validation")

    for key in LIBRARY_MODEL_KEYS:
        model = models.get(key)
        if model is None:
            print(f"[SKIP CV] {key}")
            continue

        pipe = Pipeline([
            ('preprocessor', models['preprocessor']),
            ('model', model)
        ])

        scores = cross_val_score(
            pipe, X_train, ym_train,
            cv = cv, scoring='f1_weighted', n_jobs=-1
        )

        print(f"[{key:<20s}] F1 mean= {scores.mean():.4f} std={scores.std():.4f}")
        cv_results[key] = {
            'cv_f1_mean': float(scores.mean()),
            'cv_f1_std' : float(scores.std())
        }

    return cv_results

def xgboosts_features_importance(models, X_test):
    xgb = models.get('xgboost')
    if xgb is None:
        print("[SKIP] XGboost is not loaded, skipping feature importance")
        return
    
    preprocessor = models['preprocessor']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f'f{i}' for i in range(xgb.n_features_in_)]

    importances = xgb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]


    print("\n XGBoost top 15 features")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
        print(f" {rank: 2d}. {name:40s} {importances[idx]:.4f}")


def save_metrics(binary_metrics, multi_metrics, cv_results, path='metrics.json'):
    output= {
        'binary': binary_metrics,
        'multiclass': multi_metrics,
        'cv_results': cv_results,
        'notes': {
            'main_model': 'xgboost',
            'scratch_model': 'Scratch models - experimental',
            'next_step': 'Tune XGboost hyperparameters'
        }
    }

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {path}")



if __name__=='__main__':
    # Loading test data
    X_test, ym_test, yb_test = load_test_split()

    # loading all mdoels
    models = load_models()
    
    preprocessor = models['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)

    # Binary evaluation
    bin_metrics = evaluate_binary(models, X_test_transformed, yb_test)

    # Multiclass evaluation
    multi_metrics = evaluate_all_models(models, X_test_transformed, ym_test)

    # Applying cross valdiation
    X_full = joblib.load(os.path.join(PREPROC_DIR, 'X_cleaned.pkl'))
    y_full = joblib.load(os.path.join(PREPROC_DIR, 'y_multiclass.pkl'))
    y_bin_full = joblib.load(os.path.join(PREPROC_DIR, 'y_binary.pkl'))


    X_train, _, ym_train, _, _, _ = train_test_split(
        X_full, y_full, y_bin_full,
        test_size=0.2, random_state=42, stratify=y_full
    )

    cv_results = cross_validate_models(models, X_train, ym_train)

    # Xgboost importance
    xgboosts_features_importance(models, X_test)

    # save all models
    save_metrics(bin_metrics, multi_metrics, cv_results)
