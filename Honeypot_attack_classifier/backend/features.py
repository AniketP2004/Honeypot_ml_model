import pandas as pd
import os 
import sys
from collections import defaultdict

ip_counter = defaultdict(int)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NUMERIC_COLS = [
    'first_octet', 'is_private', 'hour', 'minute', 'day_of_week',
    'src_port', 'dst_port', 'ttl', 'window_size',
    'ip_request_rate', 'is_exploit_path', 'is_known_scanner', 'payload_size'

]


from Data_preprocessing.data_cleaning import (
    extract_ip_request_rate,
    extract_path_features,
    extract_ua_features,
    extract_ip_features,
    engineer_features,
    get_preprocessor
    
)

def get_live_ip(ip: str):
    ip_counter[ip]+= 1
    return ip_counter[ip]

def build_features_row(raw):
    df = pd.DataFrame([raw])
    org_ip, org_ts = df['src_ip'].iloc[0], df['timestamp'].iloc[0]

    override = raw.get('request_rate_override')
    df['ip_request_rate'] = override if override is not None else get_live_ip(org_ip)

    df = extract_path_features(df)
    df = extract_ua_features(df)
    df = extract_ip_features(df)
    df = engineer_features(df)

    df['src_ip'] = org_ip
    df['timestamp'] = org_ts
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype('float32')
    return df

