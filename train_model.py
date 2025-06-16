# ===========================================
# SCRIPT D'ENTRAÎNEMENT AVEC VRAI DATASET N-BaIoT
# ===========================================

print("🔄 Démarrage de l'entraînement des modèles avec le dataset N-BaIoT...")

# 1. IMPORTS
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
import zipfile
from urllib.request import urlretrieve
warnings.filterwarnings('ignore')

# 2. TÉLÉCHARGEMENT ET PRÉTRAITEMENT DU DATASET
print("\n🔍 Téléchargement et prétraitement du dataset N-BaIoT...")

# URL du dataset
DATASET_URL = "https://www.kaggle.com/datasets/mkashifn/nbaiot-Dataset"
FILES = [
    "1.benign.csv",
    "4.mirai.scan.csv",
    "4.mirai.ack.csv",
    "4.mirai.syn.csv",
    "4.mirai.udp.csv",
    "4.gafgyt.udp.csv"
]

# Télécharger les fichiers
# Vérification que les fichiers sont présents dans le même dossier que le script
for file in FILES:
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Le fichier {file} est introuvable dans le dossier courant. Place tous les fichiers CSV à côté de train_model.py.")



# Charger et combiner les données
dfs = []
for file in FILES:
    df = pd.read_csv(os.path.join(".", file))
    # Ajouter une colonne 'label'
    if "benign" in file:
        df['label'] = 'benign'
    else:
        attack_type = file.split('.')[1]
        df['label'] = attack_type
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Dataset chargé: {full_df.shape[0]} lignes, {full_df.shape[1]} colonnes")

# 3. PRÉPARATION DES DONNÉES
print("\n🔧 Préparation des données...")

# Sélection des features importantes (simplifié pour l'exemple)
SELECTED_FEATURES = [
    'MI_dir_L5_weight', 'MI_dir_L5_mean', 'MI_dir_L5_variance',
    'MI_dir_L3_weight', 'MI_dir_L3_mean', 'MI_dir_L3_variance',
    'MI_dir_L1_weight', 'MI_dir_L1_mean', 'MI_dir_L1_variance',
    'MI_dir_L0.1_weight'
]

# Filtrer les features
X = full_df[SELECTED_FEATURES]
y = full_df['label']

# Encoder les labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Conversion binaire: 0 = benign, 1 = malicious
y_binary = np.where(y_encoded == 0, 0, 1)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données préparées: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
print(f"📊 Distribution: Normal={sum(y_binary==0)}, Anomalie={sum(y_binary==1)}")

# 4. ENTRAÎNEMENT DES MODÈLES
print("\n🤖 Entraînement des modèles...")

models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Stockage des modèles entraînés
trained_models = {}

for name, model in models.items():
    print(f"🔄 Entraînement: {name}")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✅ Précision: {accuracy:.4f}")
    trained_models[name] = model

# 5. OPTIMISATION DES MODÈLES
print("\n⚡ Optimisation des modèles...")

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2']
    }
}

best_models = {}

for name in trained_models:
    print(f"🔧 Optimisation de {name}...")
    grid_search = GridSearchCV(
        trained_models[name], 
        param_grids[name], 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"   ✅ Optimisation terminée! Précision: {grid_search.best_score_:.4f}")
    print(f"   🔧 Meilleurs paramètres: {grid_search.best_params_}")

# 6. SAUVEGARDE DES MODÈLES
print("\n💾 Sauvegarde des modèles...")

# Sauvegarde des modèles
for name, model in best_models.items():
    with open(f"{name.lower().replace(' ', '_')}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅ {name} sauvegardé")

# Sauvegarde du scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl sauvegardé")

# Sauvegarde des noms des features
with open("features.pkl", "wb") as f:
    pickle.dump(SELECTED_FEATURES, f)
print("✅ features.pkl sauvegardé")

# 7. RAPPORT FINAL
print("\n" + "="*50)
print("📋 RAPPORT D'ENTRAÎNEMENT FINAL")
print("="*50)
print(f"🎯 Objectif: Classification d'intrusions IoT")
print(f"📊 Dataset: N-BaIoT ({full_df.shape[0]} échantillons)")
print(f"📈 Features utilisées: {len(SELECTED_FEATURES)}")

print("\n🏆 Performances des modèles (précision):")
for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   - {name}: {accuracy:.4f}")

print("\n💾 Fichiers générés:")
print("   - random_forest_model.pkl")
print("   - svm_model.pkl")
print("   - logistic_regression_model.pkl")
print("   - scaler.pkl")
print("   - features.pkl")
print("="*50)

print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("💡 Vous pouvez maintenant exécuter: streamlit run app.py")