import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import os

SKIP_PREFIXES = ('Traceback', 'File "', 'raise ', 'paramiko.', 
                 'EOFError', 'During handling', 'Exception (server)')

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

_BASE_PACKET = lambda proto: {
    'src_ip': None, 'dst_port': None, 'protocol': proto,
    'timestamp': None, 'ttl': None, 'window_size': None,
    'tcp_flag': None, 'src_port': None, 'http_method': None,
    'http_path': None, 'user_agent': None, 'ssh_client': None,
    'payload_size': None, 'packet_len': None,      
    'ssh_username': None, 'country': None, 'isp': None, 'as_number': None 
}
ATTACK_LOG_STORE = {}

DROP_COLS = ['packet_len', 'ssh_username', 'country', 'isp', 'as_number']

def parse_ip_header_line(lines):
    header = {}
    chunks = lines.split()
    for chunk in chunks:
        if '=' in chunk:
            key = chunk.split('=')[0].strip().lower()
            value = chunk.split('=')[1].strip()

            if key == 'ttl': header['ttl'] = value
            elif key == 'len': header['packet_len'] = value
            elif key == 'flags': header['flags'] = value
            elif key == 'window': header['window'] = value
    return header


def parse_geoip_block(lines):
    geoip = {'country': None, 'isp': None, 'as_number': None}
    for line in lines:
        if 'Country' in line:
            try:
                geoip['country'] = line.split('(')[1].split(')')[0].strip()
                if geoip['country'] in ('--', '-', ''):
                    geoip['country'] = None
            except (ValueError, IndexError): pass
        elif 'ISP' in line:
            try:
                geoip['isp'] = line.split(':', 1)[1].strip()
                if geoip['isp'] in ('-', ''):
                    geoip['isp'] = None
            except Exception: pass
        elif 'AS' in line:
            chunks = line.split()
            for chunk in chunks:
                if chunk.startswith('AS') and chunk[2:].isdigit():
                    try:
                        geoip['as_number'] = int(chunk[2:])
                        break
                    except ValueError: pass
    return geoip


def get_packet_type(line):
    if 'FAKE SSH' in line: return 'FAKE_SSH'
    elif 'FAKE HTTP' in line: return 'FAKE_HTTP'
    elif 'HTTPS/TLS' in line: return 'TLS'
    elif 'SSH PORT 22' in line: return 'SSH_PASSIVE'
    elif 'TCP' in line: return 'TCP'
    return None


def extract_ssh_passive(lines, geoip):
    packet = _BASE_PACKET('SSH')
    packet['dst_port'] = 22
    for line in lines:
        if 'SOURCE IP' in line:
            packet['src_ip'] = line.split(':', 1)[1].strip()
        elif 'SOURCE PORT' in line:
            packet['src_port'] = int(line.split(':', 1)[1].strip())
        elif 'TTL' in line:
            parse_header = parse_ip_header_line(line)
            packet['ttl'] = parse_header.get('ttl')
            packet['window_size'] = parse_header.get('window')
            packet['tcp_flag'] = parse_header.get('flags')
            packet['packet_len'] = parse_header.get('packet_len')
    packet.update(geoip)
    return packet


def extract_fake_ssh(lines, geoip):
    packet = _BASE_PACKET('SSH')
    packet['dst_port'] = 2222
    for line in lines:
        if 'SOURCE IP' in line:
            packet['src_ip'] = line.split(':', 1)[1].strip()
        elif 'SOURCE PORT' in line:
            packet['src_port'] = int(line.split(':', 1)[1].strip())
        elif 'Client Banner' in line:
            val = line.split(':', 1)[1].strip()
            packet['ssh_client'] = None if val == '(no banner)' else val
        elif 'Username' in line:
            packet['ssh_username'] = line.split(':', 1)[1].strip()
    packet.update(geoip)
    return packet


def extract_fake_http(lines, geoip):
    packet = _BASE_PACKET('HTTP')
    packet['dst_port'] = 80
    in_request = False

    for line in lines:
        if 'SOURCE IP' in line:
            packet['src_ip'] = line.split(':', 1)[1].strip()
            continue
        elif 'SOURCE PORT' in line:
            packet['src_port'] = int(line.split(':', 1)[1].strip())
            continue
        
        if '├─ REQUEST' in line:
            in_request = True
            continue
        
        if in_request:
            stripped = line.strip().strip('|').strip().replace('├─', '').strip()
            if stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    packet['http_method'] = parts[0]
                    packet['http_path'] = parts[1]
                in_request = False
                continue

        if 'User-Agent' in line:
            val = line.split(':', 1)[1].strip()
            packet['user_agent'] = None if val == '-' else val

    packet.update(geoip)
    return packet


def extract_tls(lines, geoip):
    packet = _BASE_PACKET('TLS')
    packet['dst_port'] = 443
    for line in lines:
        if 'SOURCE IP' in line:
            packet['src_ip'] = line.split(':', 1)[1].strip()
        elif 'SOURCE PORT' in line:
            packet['src_port'] = int(line.split(':', 1)[1].strip())
        elif 'TTL' in line:
            parse_header = parse_ip_header_line(line)
            packet['ttl'] = parse_header.get('ttl')
            packet['window_size'] = parse_header.get('window')
            packet['tcp_flag'] = parse_header.get('flags')
            packet['packet_len'] = parse_header.get('packet_len')
    packet.update(geoip)
    return packet


def parse_log(filename):
    packets = []
    packet_type = None
    current_lines = []
    geoip_lines = []
    in_geoip = False
    last_timestamp = None
    geoip = {}

    load_logs()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip()[:4].isdigit():
                last_timestamp = line.strip()
                continue

            if '┌' in line:
                packet_type = get_packet_type(line)
                current_lines = []
                geoip_lines = []
                geoip = {}
                in_geoip = False
                continue

            if '├─ GEOIP' in line:
                in_geoip = True
                continue

            if '├─' in line and in_geoip:
                geoip = parse_geoip_block(geoip_lines)
                in_geoip = False
                # Do not pass/continue here, allow the current block label row to process down below

            if '└' in line:
                if in_geoip:
                    geoip = parse_geoip_block(geoip_lines)
                    in_geoip = False
            
                p = None
                if packet_type == 'SSH_PASSIVE': p = extract_ssh_passive(current_lines, geoip)
                elif packet_type == 'FAKE_SSH': p = extract_fake_ssh(current_lines, geoip)
                elif packet_type == 'FAKE_HTTP': p = extract_fake_http(current_lines, geoip)
                elif packet_type == 'TLS': p = extract_tls(current_lines, geoip)

                if p:
                    p['timestamp'] = last_timestamp
                    packets.append(p)
                continue

            if in_geoip:
                geoip_lines.append(line)
                continue

            if any(line.strip().startswith(p) for p in SKIP_PREFIXES):
                continue 
            else:
                current_lines.append(line)
    
    return pd.DataFrame(packets)


def assign_label_v2(row, timestamp_counts, ip_counts):
    protocol = row.get('protocol', '')
    path = str(row.get('http_path', '') or '')
    ua = str(row.get('user_agent', '') or '').lower()
    timestamp = row.get('timestamp')
    src_ip = row.get('src_ip', '')
    dst_port = row.get('dst_port')

    ts_count = timestamp_counts.get(timestamp, 0)
    ip_count = ip_counts.get(src_ip, 0)

    if dst_port == 2222:
        return 'brute_force'
    
    if protocol == 'SSH':
        if ts_count > 5 or ip_count > 10:
            return 'brute_force'
        return 'ssh_probe'
    
    if protocol == 'TLS':
        if ip_count > 10: return 'automated_scan'
        return 'port_scan'
    
    if protocol == 'HTTP':
        path_lower = path.lower()
        shell_kw  = ['shell','cmd','c99','r57','exec','passthru','system(','/proc/self']
        cloud_kw  = ['meta-data','.kube','actuator','docker','grafana']
        exploit_kw= ['xmlrpc','wp-login','phpmyadmin','.env','passwd','cgi-bin','log4j']

        if any(k in path_lower for k in shell_kw): return 'web_shell_probe'
        if any(k in path_lower for k in cloud_kw): return 'cloud_metadata_probe'
        if any(k in path_lower for k in exploit_kw): return 'exploit_attempt'
        if any(s in ua for s in KNOWN_SCANNER_UAS): return 'web_recon'
        if ts_count > 3 or ip_count > 15: return 'automated_scan'
        return 'web_recon'
    
    return 'normal'


def assign_binary_label(label):
    return 'normal' if label == 'normal' else 'attack'


def save_attack_log(packet_dic, attack_type):
    if attack_type == 'normal':
        return
    entry = {
        'timestamp': packet_dic.get('timestamp'),
        'attack_type': attack_type,
        'src_ip': packet_dic.get('src_ip'),
        'dst_port': packet_dic.get('dst_port'),
        'protocol': packet_dic.get('protocol'),
        'country': packet_dic.get('country'),
        'isp': packet_dic.get('isp'),
        'as_number': packet_dic.get('as_number'),
        'ssh_client': packet_dic.get('ssh_client'),
        'ssh_username': packet_dic.get('ssh_username'),
        'http_path': packet_dic.get('http_path'),
        'user_agent': packet_dic.get('user_agent'),
    }

    if attack_type not in ATTACK_LOG_STORE:
        ATTACK_LOG_STORE[attack_type] = []
    ATTACK_LOG_STORE[attack_type].append(entry)


def persist_logs(path=r'Data_preprocessing\attack_logs.json'):
    with open(path, 'w') as f:
        json.dump(ATTACK_LOG_STORE, f, indent=2)


def load_logs(path=r'Data_preprocessing\attack_logs.json'):
    global ATTACK_LOG_STORE
    if os.path.exists(path):
        with open(path, 'r') as f:
            ATTACK_LOG_STORE = json.load(f)
    else:
        ATTACK_LOG_STORE = {}


if __name__ == '__main__':
    load_logs()

    
    # Change 'logfile.log' to your actual target log file path inside your folder
    log_file_path = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Data_preprocessing\Raw_data'
    
    if os.path.exists(log_file_path):
        df = parse_log(log_file_path)
        
        print("Shape:", df.shape)
        print("\nProtocol counts:")
        print(df['protocol'].value_counts())
        
        #     print("\nNull counts:")
        print(df.isnull().sum())
        
        # 5. GEOIP CHECK — should have values
        print("\nSample geoip:")
        print(df[['country','isp','as_number']].dropna().head(5))
        
        # 6. LABEL CHECK
        timestamp_counts = df['timestamp'].value_counts().to_dict()
        ip_counts = df['src_ip'].value_counts().to_dict()
        df['label'] = df.apply(lambda r: assign_label_v2(r, timestamp_counts, ip_counts), axis=1)
        df['binary_label'] = df['label'].apply(assign_binary_label)
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        print("\nBinary distribution:")
        print(df['binary_label'].value_counts())
        
        # 7. ATTACK LOG CHECK
        ATTACK_LOG_STORE.clear()
        df.apply(lambda r: save_attack_log(r.to_dict(), r['label']), axis=1)
        persist_logs()
        print("\nAttack log keys:", list(ATTACK_LOG_STORE.keys()))
        print("Total logged:", sum(len(v) for v in ATTACK_LOG_STORE.values()))

        le = LabelEncoder()
        y_encoded = le.fit_transform(df['label'])
        DROP_COLS = ['packet_len', 'ssh_username', 'country', 'isp', 'as_number']
        df = df.drop(columns=DROP_COLS, errors='ignore')
        X_to_save = df.drop(columns=['label', 'binary_label'], errors='ignore')
        joblib.dump(X_to_save, os.path.join('Data_preprocessing', 'X_raw.pkl.pkl'))
        joblib.dump(y_encoded, os.path.join('Data_preprocessing', 'y_encoded.pkl'))
        joblib.dump(le, os.path.join('Data_preprocessing', 'label_encoder.pkl'))

        