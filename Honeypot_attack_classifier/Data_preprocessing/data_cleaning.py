import pandas as pd
import numpy as np
import ipaddress
import joblib
import random
import sys, os
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder


KNOWN_SCANNER_UAS = [
    "masscan", "zgrab", "nmap", "nikto", "hydra", "medusa",
    "python-requests", "go-http-client", "curl/7", "libwww-perl",
    "sqlmap", "dirbuster", "gobuster", "wfuzz", "nuclei",
    "shodan", "censys", "zmap", "httpx", "nuclei"
]
 
KNOWN_EXPLOIT_PATHS = [
    "/wp-login.php", "/xmlrpc.php", "/.env", "/shell.php",
    "/cmd.php", "/c99.php", "/.git", "/phpmyadmin",
    "/wp-admin", "/admin", "/backup.zip", "/config.php",
    "/etc/passwd", "/cgi-bin/", "/solr/", "/.kube/config",
    "/latest/meta-data", "/actuator", "/console", "/manager/"
]
 
SAFE_PATHS = [
    '/', '/index.html', '/index.htm', '/favicon.ico',
    '/about', '/about.html', '/contact', '/contact.html',
    '/home', '/robots.txt', '/sitemap.xml', '/style.css',
    '/main.js', '/logo.png', '/images/banner.jpg',
    '/api/health', '/api/status', '/privacy', '/terms'
]
 
REAL_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
]
 
SSH_CLIENTS = [
    'SSH-2.0-OpenSSH_8.9p1',
    'SSH-2.0-OpenSSH_9.0',
    'SSH-2.0-OpenSSH_9.3p1',
    'SSH-2.0-OpenSSH_8.4p1',
    'SSH-2.0-PuTTY_0.80',
]
 
TCP_FLAGS_NORMAL = ['ACK', 'SYN-ACK', 'PSH-ACK']
 
# Public IP first octets (not private, not loopback)
COMMON_PUBLIC_OCTETS = [8, 17, 34, 52, 66, 74, 104, 142, 157, 172, 185, 198, 203, 216]
 
# Hour weights: low at night, high 9-18
HOUR_WEIGHTS = [1,1,1,1,1,1,1,2,3,5,5,5,5,5,5,5,5,5,4,3,2,2,1,1]
 
# Day weights: weekdays heavier
DOW_WEIGHTS = [5,5,5,5,5,2,1]
 
def extract_ip_request_rate(df):
    ip_counts = df['src_ip'].value_counts().to_dict()
    df['ip_request_rate'] = df['src_ip'].map(ip_counts).fillna(1)
    return df

def extract_path_features(df):
    def is_exploit_path(path):
        if pd.isna(path):
            return 0    
        path_lower = str(path).lower()
        return 1 if any (k in path_lower for k in KNOWN_EXPLOIT_PATHS) else 0
    
    df['is_exploit_path'] = df['http_path'].apply(is_exploit_path)
    return df

def extract_ua_features(df):
    def is_scanner_ua(ua):
        if pd.isna(ua):
            return 0
        ua_lower = str(ua).lower()
        return 1 if any(s in ua_lower for s in KNOWN_SCANNER_UAS) else 0
    df['is_known_scanner'] = df['user_agent'].apply(is_scanner_ua)
    return df

def extract_ip_features(df):
    def check_if_private(ip_string):
        if pd.isna(ip_string):
            return 0
        try:
            return 1 if ipaddress.ip_address(ip_string).is_private else 0
        except ValueError:
            return 0
        
    df['first_octet'] = df['src_ip'].apply(
        lambda x: int(str(x).split('.')[0])if pd.notna(x) else -1
    )
    df['is_private'] = df['src_ip'].apply(check_if_private)
    df = df.drop(columns=['src_ip'], errors='ignore')
    return df

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df = df.drop(columns= ['timestamp'], errors='ignore')
    return df


def balance_classes(df, label_col='label', min_samples=50):
    class_counts = df[label_col].value_counts()
    balanced_dfs= []

    for cls, count in class_counts.items():
        cls_df = df[df[label_col]== cls]
        if count < min_samples:
            cls_df = resample(cls_df, replace= True, n_samples = min_samples, random_state= 42)
        balanced_dfs.append(cls_df)
    
    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

#  Normal Traffic Generator

def _clip_payload():
    return float(np.clip(np.random.normal(500, 200), 50, 2000))

def gen_http_row():
    return {
        'protocol':     'HTTP',
        'dst_port':     random.choice([80, 443]),
        'ttl':          None,
        'window_size':  None,
        'tcp_flag':     None,
        'src_port':     random.randint(1024, 65535),
        'http_method':  random.choices(['GET', 'POST'], weights=[0.85, 0.15])[0],
        'http_path':    random.choice(SAFE_PATHS),
        'user_agent':   random.choice(REAL_USER_AGENTS),
        'ssh_client':   None,
        'payload_size': _clip_payload(),
    }

def gen_tcp_row():
    base_ttl = random.choice([64, 128])
    noisy_ttl = int(np.clip(base_ttl + np.random.normal(0, 2), 40, 225))

    base_win = random.choice([8192, 65535])
    noisy_win = int(np.clip(base_win + np.random.normal(0, 500), 1024, 65535))

    return {
        'protocol':     'TCP',
        'dst_port':     random.choice([80, 443, 22]),
        'ttl':          noisy_ttl,
        'window_size':  noisy_win,
        'tcp_flag':     random.choice(TCP_FLAGS_NORMAL),
        'src_port':     random.randint(1024, 65535),
        'http_method':  None,
        'http_path':    None,
        'user_agent':   None,
        'ssh_client':   None,
        'payload_size': _clip_payload(),
    }

def gen_ssh_row():
    return {
        'protocol':     'SSH',
        'dst_port':     22,
        'ttl':          None,
        'window_size':  None,
        'tcp_flag':     None,
        'src_port':     random.randint(1024, 65535),
        'http_method':  None,
        'http_path':    None,
        'user_agent':   None,
        'ssh_client':   random.choice(SSH_CLIENTS),
        'payload_size': float(np.clip(np.random.normal(200, 50), 50, 500)),
    }
 
def gen_common_fields(row):

    row['ip_request_rate'] = float(np.clip(np.random.normal(2, 1.5), 0.5, 8))
    row['is_exploit_path'] = 0
    row['is_known_scanner'] = 1 if random.random() < 0.02 else 0
    row['first_octet'] = random.choice(COMMON_PUBLIC_OCTETS)
    row['is_private'] = 1 if random.random() < 0.05 else 0
    row['hour'] = random.choices(range(24), weights=HOUR_WEIGHTS)[0]
    row['minute'] = random.randint(0, 59)
    row['day_of_week'] = random.choices(range(7), weights=DOW_WEIGHTS)[0]
    return row


def generate_normal_samples(n_http=2000, n_tcp=1000, n_ssh=500):
    samples = []

    for _ in range(n_http):
        row = gen_http_row()
        row = gen_common_fields(row)
        samples.append(row)


    for _ in range(n_tcp):
        row = gen_tcp_row()
        row = gen_common_fields(row)
        samples.append(row)


    for _ in range(n_ssh):
        row = gen_ssh_row()
        row = gen_common_fields(row)
        samples.append(row)

    random.shuffle(samples)
    df = pd.DataFrame(samples)
    df['label'] = 'normal'
    df['binary_label'] = 'normal'
    return df

# Preprocessor logic
 
def get_preprocessor():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    numerical_cols = [
        'first_octet', 'is_private', 'hour', 'minute', 'day_of_week',
        'src_port', 'dst_port', 'ttl', 'window_size', 'payload_size',
        'ip_request_rate', 'is_exploit_path', 'is_known_scanner'
    ]
    one_hot_cols= ['protocol', 'tcp_flag', 'http_method']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler())
    ])
    one_hot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ohe', one_hot_pipeline, one_hot_cols),
    ])

#  Main
if __name__ == '__check__':
    # Quick sanity check — run once separately
    import joblib, pandas as pd
    X_raw = joblib.load('X_raw.pkl')
    print("X_raw cols:", sorted(X_raw.columns.tolist()))

    from data_cleaning import generate_normal_samples
    normal_df = generate_normal_samples(n_http=10, n_tcp=5, n_ssh=2)
    normal_df = normal_df.drop(columns=['label','binary_label'])
    print("normal cols:", sorted(normal_df.columns.tolist()))

    # must be identical sets
    diff = set(X_raw.columns) ^ set(normal_df.columns)
    print("MISMATCH:", diff)  # must be empty set


if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder

    NUMERICAL_COLS = [
        'first_octet', 'is_private', 'hour', 'minute', 'day_of_week',
        'src_port', 'dst_port', 'ttl', 'window_size',
        'ip_request_rate', 'is_exploit_path', 'is_known_scanner', 'payload_size'
    ]

    print("Loading raw data...")
    X_raw: pd.DataFrame = joblib.load('X_raw.pkl')
    y_multi_raw = joblib.load('y_encoded.pkl')
    le_multi: LabelEncoder = joblib.load('label_encoder.pkl')


    X_raw['label'] = le_multi.inverse_transform(y_multi_raw)
    X_raw['binary_label'] = X_raw['label'].apply(
        lambda l: 'normal' if l == 'normal' else 'attack'
    )
    LABEL_MERGE = {'port_scan': 'automated_scan', 'ssh_probe': 'automated_scan'}
    X_raw['label'] = X_raw['label'].replace(LABEL_MERGE)

    print("Extracting features...")
    X_raw = extract_ip_request_rate(X_raw)
    X_raw = extract_path_features(X_raw)
    X_raw = extract_ua_features(X_raw)
    X_raw = extract_ip_features(X_raw)
    X_raw = engineer_features(X_raw)

    print("Generating normal sampels...")
    normal_df = generate_normal_samples()

    print("Merging attack + normal samples...")
    combined = pd.concat([X_raw, normal_df], ignore_index=True)

    print("Casting numeric cols...")
    combined[NUMERICAL_COLS] = combined[NUMERICAL_COLS].apply(
        pd.to_numeric, errors = 'coerce'
    )
    
    y_multi_labels = combined['label'].values
    y_binary_labels = combined['binary_label'].values
    X_cleaned = combined.drop(columns=['label', 'binary_label'])

    le_multi_new = LabelEncoder()
    le_binary = LabelEncoder()
    y_multi_enc = le_multi_new.fit_transform(y_multi_labels)
    y_binary_enc = le_binary.fit_transform(y_binary_labels)

    le_multi_new = LabelEncoder()
    le_binary = LabelEncoder()
    y_multi_enc = le_multi_new.fit_transform(y_multi_labels)
    y_binary_enc = le_binary.fit_transform(y_binary_labels)

    SAVE_DIR = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics'
    joblib.dump(X_cleaned,     os.path.join(SAVE_DIR, 'X_cleaned.pkl'))
    joblib.dump(y_multi_enc,   os.path.join(SAVE_DIR, 'y_multiclass.pkl'))
    joblib.dump(y_binary_enc,  os.path.join(SAVE_DIR, 'y_binary.pkl'))
    joblib.dump(le_multi_new,  os.path.join(SAVE_DIR, 'label_encoder_multi.pkl'))
    joblib.dump(le_binary,     os.path.join(SAVE_DIR, 'label_encoder_binary.pkl'))

    print(f"X_cleaned shape : {X_cleaned.shape}")
    print(f"Multiclass labels: {le_multi_new.classes_}")
    print(f"Binary labels    : {le_binary.classes_}")
    print(f"Label dist (multi): {pd.Series(y_multi_enc).value_counts().to_dict()}")
    print(f"Label dist (binary): {pd.Series(y_binary_enc).value_counts().to_dict()}")