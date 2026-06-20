import pandas as pd
import numpy as np
import ipaddress
import joblib
import sys, os
from sklearn.preprocessing import LabelEncoder

KNOWN_SCANNER_UAS = [
    "masscan", "zgrab", "nmap", "nikto", "hydra", "medusa",
    "python-requests", "go-http-client", "curl/7", "libwww-perl",
    "sqlmap", "dirbuster", "gobuster", "wfuzz", "nuclei"
]

KNOWN_EXPLOIT_PATHS = [
    "/wp-login.php", "/xmlrpc.php", "/.env", "/shell.php",
    "/cmd.php", "/c99.php", "/.git", "/phpmyadmin",
    "/wp-admin", "/admin", "/backup.zip", "/config.php",
    "/etc/passwd", "/cgi-bin/", "/solr/", "/.kube/config",
    "/latest/meta-data", "/actuator", "/console", "/manager/"
]


def extract_value(line, keyword):
    if keyword in line:
        if ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[1].strip()
        elif "=" in line:
            chunks = line.split()
            for chunk in chunks:
                if chunk.startswith(keyword + '='):
                    return chunk.split('=')[1].strip()
    return None

def get_packet_type(line):
    if 'HTTPS/TLS' in line: return 'TLS'
    elif 'HTTP' in line: return 'HTTP'
    elif 'BANNER' in line: return 'SSH_BANNER'
    elif '┌' in line: return 'TCP'
    return None

_BASE_PACKET = lambda proto: {
    'src_ip': None, 'dst_port': None, 'protocol': proto,
    'timestamp': None, 'ttl': None, 'window_size': None,
    'tcp_flag': None, 'src_port': None, 'http_method': None,
    'http_path': None, 'user_agent': None, 'ssh_client': None,
    'payload_size': None,  # NEW feature
}

def extract_tcp(lines):
    p = _BASE_PACKET('TCP')
    for line in lines:
        for keyword, key in {'SOURCE PORT':'src_port','SOURCE IP':'src_ip',
                        'DEST PORT':'dst_port','TTL':'ttl',
                        'Window Size':'window_size','Flags':'tcp_flag',
                        'Payload':'payload_size'}.items():
            value = extract_value(line, keyword)
            if value: p[key] = value
    return p

def extract_http(lines):
    p = _BASE_PACKET('HTTP')
    for line in lines:
        for keyword, key in {'SOURCE PORT':'src_port','SOURCE IP':'src_ip',
                        'DEST PORT':'dst_port','Method':'http_method',
                        'Path':'http_path','User-Agent':'user_agent',
                        'Content-Length':'payload_size'}.items():
            value = extract_value(line, keyword)
            if value: p[key] = value
    return p

def extract_ssh(lines):
    p = _BASE_PACKET('SSH')
    for line in lines:
        for keyword, key in {'SOURCE PORT':'src_port','SOURCE IP':'src_ip',
                        'DEST PORT':'dst_port',
                        'CLIENT REPLIED':'ssh_client',
                        'Client Banner':'ssh_client'}.items():
            value = extract_value(line, keyword)
            if value: p[key] = value
    return p

def extract_tls(lines):
    p = _BASE_PACKET('TLS')
    for line in lines:
        for keyword, key in {'Window':'window_size','SOURCE PORT':'src_port',
                        'SOURCE IP':'src_ip','DEST PORT':'dst_port',
                        'TTL':'ttl'}.items():
            value = extract_value(line, keyword)
            if value: p[key] = value
    return p

def parse_log(filename='data3.log'):
    with open(filename, 'r', encoding='utf-8') as f:
        packet = []
        packet_type = None
        current_lines = []
        last_timestamp = None
        for line in f:
            if line.strip()[:4].isdigit():
                last_timestamp = line.strip()
            elif '┌' in line:
                packet_type = get_packet_type(line)
                current_lines = []
            elif '└' in line:
                if packet_type is not None:
                    value = None
                    if packet_type == 'TCP': value = extract_tcp(current_lines)
                    elif packet_type == 'HTTP': value = extract_http(current_lines)
                    elif packet_type == 'SSH_BANNER': value = extract_ssh(current_lines)
                    elif packet_type == 'TLS': value = extract_tls(current_lines)
                    if value:
                        value['timestamp'] = last_timestamp
                        packet.append(value)
            else:
                current_lines.append(line)
    return pd.DataFrame(packet)

def assign_label(row, timestamp_counts, ip_counts):
    protocol  = row.get('protocol', '')
    path      = str(row.get('http_path', '') or '')
    ua        = str(row.get('user_agent', '') or '').lower()
    timestamp = row.get('timestamp')
    src_ip    = row.get('src_ip', '')

    ts_count  = timestamp_counts.get(timestamp, 0)
    ip_count  = ip_counts.get(src_ip, 0)

    # SSH attack
    if protocol == 'SSH':
        if ts_count > 5 or ip_count > 10:
            return 'brute_force'
        return 'ssh_probe'
    
    # HTTP attacks
    elif protocol == 'HTTP':
        path_lower = path.lower()

        # Web shell RCE probe
        shell_keywords = ['shell', 'cmd', 'c99', 'r57', 'webshell',
                          'exec', 'passthru', 'system(', '/proc/self']
        
        if any(k in path_lower for k in shell_keywords):
            return 'web_shell_probe'
        
        # Cloud/ Kubernetes        
        cloud_keywords = ['meta-data', '.kube', 'actuator',
                          'v1/namespace', 'docker', 'grafana']
        
        if any (k in path_lower for k in cloud_keywords):
            return 'cloud_metadata_probe'
        
        # Known exploits
        exploit_keywords = ['xmlrpc', 'wp-login', 'wp-config',
                            'phpmyadmin', 'administrator', '.env',
                            'passwd', 'cgi-bin', 'struts', 'solr',
                            'jndi', 'log4j', 'confluence']
        
        if any(k in path_lower for k in exploit_keywords):
            return 'exploit_attempt'
        
        # Known scanner user agent
        if any (s in ua for s in KNOWN_SCANNER_UAS):
            return 'web_recon'
        
        # High rate = Automated
        if ts_count > 3 or ip_count > 15:
            return 'automated_scan'
        
        return 'web_recon'
    
    # TLS probe
    elif protocol == 'TLS':
        if ip_count > 10:
            return 'automated_scan'
        return 'web_recon'
    
    # Raw TCP
    elif protocol == 'TCP':
        if ts_count > 3 or ip_count > 10:
            return 'automated_scan'
        return 'port_scan'
    
    return 'port_scan'

#  Extracting the features

def check_if_private(ip_string):
    if pd.isna(ip_string): return 0
    try: return 1 if ipaddress.ip_address(ip_string).is_private else 0
    except ValueError: return 0

def extract_ip_request_rate(df):
    
    ip_counts = df['src_ip'].value_counts().to_dict()
    df['ip_request_rate'] = df['src_ip'].map(ip_counts).fillna(1)
    return df

def extract_path_features(df):
    
    def is_exploit_path(path):
        if pd.isna(path):
            return 0
        path_lower = str(path).lower()
        return 1 if any(k in path_lower for k in KNOWN_EXPLOIT_PATHS) else 0

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
    df['first_octet'] = df['src_ip'].apply(lambda x: int(str(x).split('.')[0]) if pd.notna(x) else -1)
    df['is_private'] = df['src_ip'].apply(check_if_private)
    df = df.drop(columns=['src_ip'])
    return df


def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df = df.drop(columns=['timestamp'])
    return df

def get_preprocessor():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    numerical_cols = ['first_octet', 'is_private', 'hour', 'minute', 'day_of_week', 'src_port', 'dst_port', 'ttl', 'window_size', 'ip_request_rate', 'is_exploit_path', 'is_known_scanner', 'payload_size']
    high_cardinality_cols = ['http_path', 'user_agent', 'ssh_client']
    one_hot_cols = ['protocol', 'tcp_flag', 'http_method']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler())
    ])
    one_hot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    cardinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', TargetEncoder())
    ])

    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ohe', one_hot_pipeline, one_hot_cols),
        ('high_cardinal', cardinal_pipeline, high_cardinality_cols)
    ])


if __name__ == '__main__':

    print("Parsing log...")
    df = parse_log(r'C:\Users\project\Desktop\Honeypot_New_Repo\Data_preprocessing\Raw_data')

    print("Assigning labels...")
    timestamp_counts = df['timestamp'].value_counts().to_dict()
    ip_counts = df['src_ip'].value_counts().to_dict()
    df['label'] = df.apply(lambda row: assign_label(row, timestamp_counts, ip_counts), axis=1)
    print("Engineering features...")
    df = extract_ip_request_rate(df)
    df = extract_path_features(df)
    df = extract_ua_features(df)
    df = extract_ip_features(df)
    df = engineer_features(df)

    numerical_cols = ['first_octet', 'is_private', 'hour', 'minute', 'day_of_week',
                      'src_port', 'dst_port', 'ttl', 'window_size', 'ip_request_rate', 'is_exploit_path', 'is_known_scanner', 'payload_size']
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

    # SPLIT X AND y, SAVE SEPARATELY 
    X = df.drop(columns=['label'])
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    joblib.dump(X, 'X_raw.pkl')
    joblib.dump(y_encoded, 'y_encoded.pkl')
    joblib.dump(le, 'label_encoder.pkl')    

    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X, y_encoded)
    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(X_transformed, 'X_transformed.pkl')
    print(X_transformed.shape)

    print(f"Saved X_raw.pkl {X.shape}, y_encoded.pkl {y_encoded.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Label distribution: {pd.Series(y_encoded).value_counts().to_dict()}")
