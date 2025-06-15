# ===========================================
# SCRIPT D'ENTRAÎNEMENT POUR DÉPLOIEMENT
# ===========================================

print("🔄 Démarrage de l'entraînement des modèles...")

# 1. IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("✅ Bibliothèques chargées avec succès!")

# 2. GÉNÉRATION DU DATASET
print("\n🔍 Génération des données d'entraînement...")
np.random.seed(42)
n_samples = 5000

# Features pour simulation d'un réseau IoT
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

# Création du DataFrame
df = pd.DataFrame(data)

# Création de la variable cible
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

print(f"✅ Dataset créé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"📊 Distribution: Normal={sum(df['target']==0)}, Anomalie={sum(df['target']==1)}")

# 3. PRÉPARATION DES DONNÉES
print("\n🔧 Préparation des données...")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données préparées: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")

# 4. ENTRAÎNEMENT DES MODÈLES
print("\n🤖 Entraînement des modèles...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    print(f"🔄 Entraînement: {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Précision: {accuracy:.4f}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

print(f"\n🏆 Meilleur modèle: {best_name} (Précision: {best_score:.4f})")

# 5. OPTIMISATION DU MEILLEUR MODÈLE
print(f"\n⚡ Optimisation de {best_name}...")

if best_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
elif best_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf', 'linear']
    }
else:  # Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

# GridSearch
grid_search = GridSearchCV(
    best_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Modèle final optimisé
final_model = grid_search.best_estimator_
final_pred = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, final_pred)

print(f"✅ Optimisation terminée!")
print(f"📈 Précision finale: {final_accuracy:.4f}")
print(f"🔧 Meilleurs paramètres: {grid_search.best_params_}")

# 6. SAUVEGARDE DES MODÈLES
print("\n💾 Sauvegarde des modèles...")

# Sauvegarde du modèle final
with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)
print("✅ model.pkl sauvegardé")

# Sauvegarde du scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl sauvegardé")

# Sauvegarde des noms des features
with open("features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
print("✅ features.pkl sauvegardé")

# 7. RAPPORT FINAL
print("\n" + "="*50)
print("📋 RAPPORT D'ENTRAÎNEMENT FINAL")
print("="*50)
print(f"🎯 Objectif: Classification d'intrusions IoT")
print(f"📊 Dataset: {n_samples} échantillons, {X.shape[1]} features")
print(f"🏆 Modèle sélectionné: {best_name}")
print(f"📈 Précision finale: {final_accuracy:.4f}")
print(f"🔧 Optimisation: GridSearchCV")
print(f"💾 Fichiers générés: model.pkl, scaler.pkl, features.pkl")
print("✅ Prêt pour le déploiement!")
print("="*50)

print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("💡 Vous pouvez maintenant exécuter: streamlit run app.py")
