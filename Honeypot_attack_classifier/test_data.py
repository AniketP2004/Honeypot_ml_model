import pandas as pd
import numpy as np
import random
import joblib
from sklearn.preprocessing import LabelEncoder
import sys

KNOWN_EXPLOIT_PATHS = [
    "/wp-login.php", "/xmlrpc.php", "/.env", "/shell.php",
    "/cmd.php", "/c99.php", "/.git", "/phpmyadmin",
    "/wp-admin", "/admin", "/backup.zip", "/config.php",
    "/etc/passwd", "/cgi-bin/", "/solr/", "/.kube/config",
    "/latest/meta-data", "/actuator", "/console", "/manager/"
]

EXPLOIT_USER_AGENTS = [
    "sqlmap/1.6", "Nikto/2.1.6", "Mozilla/5.0 (compatible; Nuclei)",
    "python-requests/2.28", "curl/7.68.0", "Gobuster/3.1",
    "WPScan v3.8", "Hydra"
]

def generate_exploit_attempt_rows(n=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        row = {
            'protocol': 'HTTP',
            'http_path': random.choice(KNOWN_EXPLOIT_PATHS),
            'user_agent': random.choice(EXPLOIT_USER_AGENTS),
            'ssh_client': None,
            'tcp_flag': random.choice(['SYN', 'ACK', 'PSH-ACK']),
            'http_method': random.choice(['GET', 'POST']),
            'src_port': np.random.randint(1024, 65535),
            'dst_port': random.choice([80, 443, 8080]),
            'ttl': int(np.random.choice([64, 128, 255])) + np.random.randint(-2, 3),
            'window_size': int(np.random.choice([8192, 65535, 29200])) + np.random.randint(-500, 500),
            'payload_size': np.random.randint(50, 1500),
            'is_exploit_path': 1,
            'is_known_scanner': 1,
            'ip_request_rate': np.random.randint(3, 20),
            'first_octet': np.random.randint(1, 223),
            'is_private': 0,
            'hour': np.random.randint(0, 24),
            'minute': np.random.randint(0, 60),
            'day_of_week': np.random.randint(0, 7),
            'label': 'exploit_attempt'
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    sys.path.append(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Data_preprocessing')
    from data_cleaning import get_preprocessor

    # Load already-merged dataset (has label column as strings already)
    combined = pd.read_pickle('X_raw_merged.pkl')
    print("Before adding synthetic:")
    print(combined.shape)
    print(combined['label'].value_counts())

    synthetic_exploit = generate_exploit_attempt_rows(n=100)
    final_combined = pd.concat([combined, synthetic_exploit], ignore_index=True)

    print("\nFinal label distribution:")
    print(final_combined['label'].value_counts())

    le_new = LabelEncoder()
    y_new = le_new.fit_transform(final_combined['label'])
    X_new = final_combined.drop(columns=['label'])

    joblib.dump(le_new, 'label_encoder.pkl')

    preprocessor = get_preprocessor()
    X_transformed_new = preprocessor.fit_transform(X_new, y_new)

    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(X_transformed_new, 'X_transformed.pkl')
    joblib.dump(y_new, 'y_encoded.pkl')

    print(X_transformed_new.shape)
    print(pd.Series(y_new).value_counts())