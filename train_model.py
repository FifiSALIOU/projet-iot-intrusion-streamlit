# train_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Génération des données (mêmes règles que dans ton projet original)
np.random.seed(42)
n_samples = 5000

data = {
    'packet_size': np.random.normal(512, 200, n_samples),
    'duration': np.random.exponential(2, n_samples),
    'src_bytes': np.random.normal(1024, 500, n_samples),
    'dst_bytes': np.random.normal(768, 300, n_samples),
    'protocol_tcp': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'protocol_udp': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'service_http': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'service_ftp': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'urgent': np.random.randint(0, 5, n_samples),
    'hot': np.random.randint(0, 10, n_samples),
    'num_failed_logins': np.random.randint(0, 5, n_samples),
    'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'num_compromised': np.random.randint(0, 3, n_samples),
    'root_shell': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'su_attempted': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'num_root': np.random.randint(0, 5, n_samples),
    'num_file_creations': np.random.randint(0, 10, n_samples),
    'count': np.random.randint(1, 100, n_samples),
    'srv_count': np.random.randint(1, 50, n_samples),
    'dst_host_count': np.random.randint(1, 255, n_samples)
}

df = pd.DataFrame(data)

# Génération de la target
target = []
for i in range(n_samples):
    score = 0
    if df.iloc[i]['num_failed_logins'] > 2: score += 2
    if df.iloc[i]['root_shell'] == 1: score += 3
    if df.iloc[i]['num_compromised'] > 0: score += 2
    if df.iloc[i]['su_attempted'] == 1: score += 1
    if df.iloc[i]['urgent'] > 2: score += 1
    if df.iloc[i]['packet_size'] > 1000: score += 1
    if np.random.random() < 0.1: score += np.random.randint(1, 3)
    target.append(1 if score > 3 else 0)

df['target'] = target

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Sauvegarde du modèle et du scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Modèle, scaler et features sauvegardés.")
