# config.py
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
LOG_PATH  = os.path.join(BASE_DIR, 'data', 'attack_logs.json')
RAW_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
# create dirs if missing
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(RAW_DIR,   exist_ok=True)