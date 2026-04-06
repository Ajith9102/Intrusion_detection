"""
Intrusion Detection System - Flask Web App
With Login & Register
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle, os, numpy as np, json, hashlib

app = Flask(__name__)
app.secret_key = 'ids_secret_key_2024'

# ── Users file ────────────────────────────────────────────────────────────────
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'random_forest_model.pkl')

model_data = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Model not found at {MODEL_PATH} — predictions will use demo mode")

# ── Feature lists ─────────────────────────────────────────────────────────────
PROTOCOLS = ['tcp', 'udp', 'icmp']
FLAGS      = ['SF', 'S0', 'REJ', 'RSTO', 'SH', 'RSTR', 'S1', 'S2', 'S3', 'OTH']
SERVICES   = ['http','ftp','smtp','ssh','domain_u','auth','finger','telnet',
              'eco_i','other','private','ftp_data','urp_i','tim_i','red_i',
              'irc','X11','Z39_50','aol','auth','bgp','courier','csnet_ns',
              'ctf','daytime','discard','domain','echo','efs','exec','gopher',
              'harvest','hostnames','http_443','http_8001','imap4','iso_tsap',
              'klogin','kshell','ldap','link','login','mtp','name','netbios_dgm',
              'netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','pm_dump',
              'pop_2','pop_3','printer','remote_job','rje','shell','sql_net',
              'ssh','sunrpc','supdup','systat','time','uucp','uucp_path','vmnet',
              'whois']

FEATURE_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        users = load_users()
        if username in users and users[username] == hash_password(password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        users = load_users()
        if not username or not password:
            flash('Username and password are required', 'error')
        elif username in users:
            flash('Username already exists', 'error')
        elif password != confirm:
            flash('Passwords do not match', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
        else:
            users[username] = hash_password(password)
            save_users(users)
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ── App routes ────────────────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html',
                           protocols=PROTOCOLS,
                           flags=FLAGS,
                           services=sorted(set(SERVICES)),
                           username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    try:
        data = request.form

        raw = {
            'duration':           float(data.get('duration', 0)),
            'protocol_type':      data.get('protocol_type', 'tcp'),
            'service':            data.get('service', 'http'),
            'flag':               data.get('flag', 'SF'),
            'src_bytes':          float(data.get('src_bytes', 0)),
            'dst_bytes':          float(data.get('dst_bytes', 0)),
            'land':               float(data.get('land', 0)),
            'wrong_fragment':     float(data.get('wrong_fragment', 0)),
            'urgent':             float(data.get('urgent', 0)),
            'hot':                float(data.get('hot', 0)),
            'num_failed_logins':  float(data.get('num_failed_logins', 0)),
            'logged_in':          float(data.get('logged_in', 0)),
            'num_compromised':    float(data.get('num_compromised', 0)),
            'root_shell':         float(data.get('root_shell', 0)),
            'su_attempted':       float(data.get('su_attempted', 0)),
            'num_root':           float(data.get('num_root', 0)),
            'num_file_creations': float(data.get('num_file_creations', 0)),
            'num_shells':         float(data.get('num_shells', 0)),
            'num_access_files':   float(data.get('num_access_files', 0)),
            'num_outbound_cmds':  0.0,
            'is_host_login':      float(data.get('is_host_login', 0)),
            'is_guest_login':     float(data.get('is_guest_login', 0)),
            'count':              float(data.get('count', 0)),
            'srv_count':          float(data.get('srv_count', 0)),
            'serror_rate':        float(data.get('serror_rate', 0)),
            'srv_serror_rate':    float(data.get('srv_serror_rate', 0)),
            'rerror_rate':        float(data.get('rerror_rate', 0)),
            'srv_rerror_rate':    float(data.get('srv_rerror_rate', 0)),
            'same_srv_rate':      float(data.get('same_srv_rate', 1)),
            'diff_srv_rate':      float(data.get('diff_srv_rate', 0)),
            'srv_diff_host_rate': float(data.get('srv_diff_host_rate', 0)),
            'dst_host_count':     float(data.get('dst_host_count', 0)),
            'dst_host_srv_count': float(data.get('dst_host_srv_count', 0)),
            'dst_host_same_srv_rate':      float(data.get('dst_host_same_srv_rate', 0)),
            'dst_host_diff_srv_rate':      float(data.get('dst_host_diff_srv_rate', 0)),
            'dst_host_same_src_port_rate': float(data.get('dst_host_same_src_port_rate', 0)),
            'dst_host_srv_diff_host_rate': float(data.get('dst_host_srv_diff_host_rate', 0)),
            'dst_host_serror_rate':        float(data.get('dst_host_serror_rate', 0)),
            'dst_host_srv_serror_rate':    float(data.get('dst_host_srv_serror_rate', 0)),
            'dst_host_rerror_rate':        float(data.get('dst_host_rerror_rate', 0)),
            'dst_host_srv_rerror_rate':    float(data.get('dst_host_srv_rerror_rate', 0)),
        }

        if model_data:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            proto_map = {'tcp':0,'udp':1,'icmp':2}
            raw['protocol_type'] = proto_map.get(raw['protocol_type'], 0)
            svc_list = sorted(set(SERVICES))
            raw['service'] = svc_list.index(raw['service']) if raw['service'] in svc_list else 0
            flag_map = {f:i for i,f in enumerate(FLAGS)}
            raw['flag'] = flag_map.get(raw['flag'], 0)
            X = np.array([[raw[c] for c in FEATURE_COLS]])
            X_scaled = model_data['scaler'].transform(X)
            pred = model_data['model'].predict(X_scaled)[0]
            prob = model_data['model'].predict_proba(X_scaled)[0]
            confidence = float(max(prob)) * 100
            result = 'Attack' if pred == 1 else 'Normal'
        else:
            is_attack = (raw['src_bytes'] > 10000 or
                         raw['serror_rate'] > 0.5 or
                         raw['flag'] in ['S0','REJ'] or
                         raw['num_failed_logins'] > 0)
            result = 'Attack' if is_attack else 'Normal'
            confidence = 87.5

        return render_template('result.html', result=result, confidence=confidence,
                               duration=data.get('duration',0),
                               protocol=data.get('protocol_type','tcp'),
                               service=data.get('service','http'),
                               flag=data.get('flag','SF'),
                               src_bytes=data.get('src_bytes',0),
                               dst_bytes=data.get('dst_bytes',0))

    except Exception as e:
        return render_template('result.html', result='Error', confidence=0,
                               error=str(e), duration=0, protocol='', service='', flag='', src_bytes=0, dst_bytes=0)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Intrusion Detection System - Web App")
    print("  Open browser at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=10000)
