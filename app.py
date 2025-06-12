# ===========================================
# PROJET: Classification d'intrusion réseau NB1-IoT
# Instructions: Copiez tout ce code dans UNE SEULE cellule
# ===========================================
import streamlit as st
import matplotlib.pyplot as plt
# 1. IMPORTS - NE PAS SÉPARER
print("🔄 Chargement des bibliothèques...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings('ignore')
    print("✅ Toutes les bibliothèques chargées avec succès!")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les packages manquants avec: !pip install scikit-learn matplotlib seaborn")

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

print("\n" + "="*60)
print("📊 PROJET: Classification d'intrusion réseau NB1-IoT")
print("="*60)

# 2. GÉNÉRATION DU DATASET D'EXEMPLE
print("\n🔍 ÉTAPE 1: GÉNÉRATION DES DONNÉES")
print("-" * 40)

# Création d'un dataset d'exemple IoT
np.random.seed(42)
n_samples = 5000

print(f"📝 Génération de {n_samples} échantillons...")

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

# Création de la variable cible basée sur des règles logiques
target = []
for i in range(n_samples):
    score = 0
    # Règles pour détecter les anomalies
    if df.iloc[i]['num_failed_logins'] > 2: score += 2
    if df.iloc[i]['root_shell'] == 1: score += 3
    if df.iloc[i]['num_compromised'] > 0: score += 2
    if df.iloc[i]['su_attempted'] == 1: score += 1
    if df.iloc[i]['urgent'] > 2: score += 1
    if df.iloc[i]['packet_size'] > 1000: score += 1

    # Ajout d'aléatoire pour plus de réalisme
    if np.random.random() < 0.1: score += np.random.randint(1, 3)

    target.append(1 if score > 3 else 0)  # 1=anomaly, 0=normal

df['target'] = target

print(f"✅ Dataset créé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"📊 Distribution des classes:")
print(f"   - Normal (0): {sum(df['target'] == 0)} ({sum(df['target'] == 0)/len(df)*100:.1f}%)")
print(f"   - Anomalie (1): {sum(df['target'] == 1)} ({sum(df['target'] == 1)/len(df)*100:.1f}%)")

# 3. PRÉPARATION DES DONNÉES
print("\n🔧 ÉTAPE 2: PRÉPARATION DES DONNÉES")
print("-" * 40)

# Séparation features et target
X = df.drop('target', axis=1)
y = df['target']

print(f"📋 Features: {list(X.columns)}")
print(f"🎯 Target: {len(y)} échantillons")

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données préparées:")
print(f"   - Training: {X_train_scaled.shape}")
print(f"   - Test: {X_test_scaled.shape}")

# 4. ENTRAÎNEMENT DES 3 MODÈLES
print("\n🤖 ÉTAPE 3: ENTRAÎNEMENT DES MODÈLES")
print("-" * 40)

# Définition des 3 modèles
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Entraînement et évaluation
results = {}
model_objects = {}

for name, model in models.items():
    print(f"\n🔄 Entraînement: {name}")

    # Entraînement
    model.fit(X_train_scaled, y_train)
    model_objects[name] = model

    # Prédictions
    y_pred = model.predict(X_test_scaled)

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_score = report['macro avg']['f1-score']

    # Validation croisée
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')

    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'report': report
    }

    print(f"   ✅ Précision: {accuracy:.4f}")
    print(f"   📊 F1-Score: {f1_score:.4f}")
    print(f"   🔄 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 5. COMPARAISON ET VISUALISATION
print("\n📊 ÉTAPE 4: COMPARAISON DES MODÈLES")
print("-" * 40)

# Création du tableau de comparaison
comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Modèle': name,
        'Précision': result['accuracy'],
        'F1-Score': result['f1_score'],
        'CV Score': result['cv_score']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n📈 TABLEAU DE COMPARAISON:")
print(comparison_df.round(4))

# Identification du meilleur modèle
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Modèle']
print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"   F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'F1-Score']:.4f}")

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Analyse des Modèles de Classification', fontsize=16)

# 1. Matrices de confusion
for i, (name, result) in enumerate(results.items()):
    if i < 3:  # Seulement 3 modèles
        row = i // 2
        col = i % 2 if i < 2 else 0

        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'{name}\nF1-Score: {result["f1_score"]:.4f}')
        axes[row, col].set_xlabel('Prédictions')
        axes[row, col].set_ylabel('Vraies valeurs')

# 2. Comparaison des métriques
ax = axes[1, 1]
x_pos = range(len(comparison_df))
width = 0.25

ax.bar([p - width for p in x_pos], comparison_df['Précision'], width,
       label='Précision', alpha=0.8)
ax.bar(x_pos, comparison_df['F1-Score'], width,
       label='F1-Score', alpha=0.8)
ax.bar([p + width for p in x_pos], comparison_df['CV Score'], width,
       label='CV Score', alpha=0.8)

ax.set_xlabel('Modèles')
ax.set_ylabel('Score')
ax.set_title('Comparaison des Métriques')
ax.set_xticks(x_pos)
ax.set_xticklabels([name.replace(' ', '\n') for name in comparison_df['Modèle']])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. OPTIMISATION AVEC ANALYSE DÉTAILLÉE DES HYPERPARAMÈTRES
print(f"\n⚡ ÉTAPE 5: OPTIMISATION DÉTAILLÉE DE {best_model_name}")
print("-" * 50)

# Paramètres d'optimisation étendus selon le modèle
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    base_model = SVC(random_state=42, probability=True)
else:  # Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [1000, 2000, 3000]
    }
    base_model = LogisticRegression(random_state=42)

print("🔍 Recherche exhaustive des hyperparamètres...")
print(f"🎯 Espace de recherche: {np.prod([len(v) for v in param_grid.values()])} combinaisons")

# Sauvegarde des performances initiales
initial_model = model_objects[best_model_name]
initial_f1 = results[best_model_name]['f1_score']
initial_accuracy = results[best_model_name]['accuracy']

print(f"\n📊 PERFORMANCES INITIALES:")
print(f"   - F1-Score: {initial_f1:.4f}")
print(f"   - Précision: {initial_accuracy:.4f}")

# GridSearch avec plus de détails
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,  # Plus de folds pour plus de précision
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ MEILLEURS PARAMÈTRES TROUVÉS:")
for param, value in grid_search.best_params_.items():
    print(f"   - {param}: {value}")

# Évaluation du modèle optimisé
optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test_scaled)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
report_optimized = classification_report(y_test, y_pred_optimized, output_dict=True)
f1_optimized = report_optimized['macro avg']['f1-score']

print(f"\n🎯 RÉSULTATS DE L'OPTIMISATION:")
print(f"   - F1-Score original: {initial_f1:.4f}")
print(f"   - F1-Score optimisé: {f1_optimized:.4f}")
improvement_f1 = f1_optimized - initial_f1
print(f"   - Amélioration F1: {improvement_f1:+.4f} ({improvement_f1/initial_f1*100:+.2f}%)")

print(f"   - Précision originale: {initial_accuracy:.4f}")
print(f"   - Précision optimisée: {accuracy_optimized:.4f}")
improvement_acc = accuracy_optimized - initial_accuracy
print(f"   - Amélioration Précision: {improvement_acc:+.4f} ({improvement_acc/initial_accuracy*100:+.2f}%)")

# 7. ANALYSE DÉTAILLÉE D'IMPACT DES HYPERPARAMÈTRES
print("\n📌 ÉTAPE 6: ANALYSE D'IMPACT DES HYPERPARAMÈTRES")
print("-" * 50)

# Extraction des résultats détaillés
cv_results = pd.DataFrame(grid_search.cv_results_)

# Identification des 3 hyperparamètres les plus impactants
param_impact = {}
for param in param_grid.keys():
    param_col = f'param_{param}'
    if param_col in cv_results.columns:
        # Calcul de la variance des scores pour chaque valeur du paramètre
        param_scores = cv_results.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'count'])
        variance = param_scores['mean'].var()
        param_impact[param] = variance

# Tri des paramètres par impact (variance des scores)
sorted_params = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)
top_3_params = [param for param, _ in sorted_params[:3]]

print(f"🔝 TOP 3 HYPERPARAMÈTRES LES PLUS IMPACTANTS:")
for i, (param, impact) in enumerate(sorted_params[:3], 1):
    print(f"   {i}. {param}: variance = {impact:.6f}")

# Visualisation de l'impact des 3 hyperparamètres les plus influents
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Impact des Hyperparamètres sur les Performances - {best_model_name}", fontsize=16)

for i, param in enumerate(top_3_params):
    param_col = f'param_{param}'

    # Boxplot pour chaque paramètre
    sns.boxplot(data=cv_results, x=param_col, y='mean_test_score', ax=axes[i])
    axes[i].set_title(f"Impact de '{param}'\n(Variance: {param_impact[param]:.6f})")
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('F1-Score Moyen')
    axes[i].tick_params(axis='x', rotation=45)

    # Ajout de statistiques
    param_stats = cv_results.groupby(param_col)['mean_test_score'].agg(['mean', 'std'])
    best_value = param_stats['mean'].idxmax()
    worst_value = param_stats['mean'].idxmin()

    # Annotation des meilleures/pires valeurs
    axes[i].axhline(y=param_stats.loc[best_value, 'mean'], color='green',
                   linestyle='--', alpha=0.7, label=f'Meilleur: {best_value}')
    axes[i].axhline(y=param_stats.loc[worst_value, 'mean'], color='red',
                   linestyle='--', alpha=0.7, label=f'Pire: {worst_value}')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Analyse statistique détaillée
print(f"\n📈 ANALYSE STATISTIQUE DÉTAILLÉE:")
print("-" * 40)

for param in top_3_params:
    param_col = f'param_{param}'
    param_stats = cv_results.groupby(param_col)['mean_test_score'].agg(['mean', 'std', 'min', 'max'])

    print(f"\n🔧 Hyperparamètre: {param}")
    print(f"   Valeurs testées: {list(cv_results[param_col].unique())}")

    best_value = param_stats['mean'].idxmax()
    worst_value = param_stats['mean'].idxmin()

    print(f"   ✅ Meilleure valeur: {best_value} (F1-Score: {param_stats.loc[best_value, 'mean']:.4f})")
    print(f"   ❌ Pire valeur: {worst_value} (F1-Score: {param_stats.loc[worst_value, 'mean']:.4f})")

    impact_range = param_stats['mean'].max() - param_stats['mean'].min()
    print(f"   📊 Écart de performance: {impact_range:.4f}")
    print(f"   📈 Impact relatif: {impact_range/param_stats['mean'].mean()*100:.2f}%")

# Matrice de confusion finale avec plus de détails
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Matrice de confusion originale
cm_original = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Anomalie'], yticklabels=['Normal', 'Anomalie'])
ax1.set_title(f'{best_model_name} - Original\nF1-Score: {initial_f1:.4f}')
ax1.set_xlabel('Prédictions')
ax1.set_ylabel('Vraies valeurs')

# Matrice de confusion optimisée
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Normal', 'Anomalie'], yticklabels=['Normal', 'Anomalie'])
ax2.set_title(f'{best_model_name} - Optimisé\nF1-Score: {f1_optimized:.4f}')
ax2.set_xlabel('Prédictions')
ax2.set_ylabel('Vraies valeurs')

plt.tight_layout()
plt.show()

# Rapport final détaillé
print(f"\n📋 RAPPORT DE CLASSIFICATION FINAL:")
print("=" * 60)
print("MODÈLE ORIGINAL:")
print(classification_report(y_test, results[best_model_name]['predictions'],
                          target_names=['Normal', 'Anomalie']))

print("\nMODÈLE OPTIMISÉ:")
print(classification_report(y_test, y_pred_optimized,
                          target_names=['Normal', 'Anomalie']))

# Résumé exécutif pour le rapport
print(f"\n📝 RÉSUMÉ EXÉCUTIF POUR LE RAPPORT:")
print("=" * 50)
print(f"🎯 Objectif: Classification d'intrusions réseau IoT")
print(f"📊 Dataset: {n_samples} échantillons avec {X.shape[1]} features")
print(f"🤖 Modèles comparés: {', '.join(models.keys())}")
print(f"🏆 Meilleur modèle: {best_model_name}")
print(f"📈 Performance initiale: F1-Score = {initial_f1:.4f}")
print(f"⚡ Performance optimisée: F1-Score = {f1_optimized:.4f}")
print(f"📊 Amélioration: {improvement_f1:+.4f} ({improvement_f1/initial_f1*100:+.2f}%)")
print(f"🔧 Technique d'optimisation: GridSearchCV avec {np.prod([len(v) for v in param_grid.values()])} combinaisons")
print(f"🎯 Hyperparamètres les plus impactants: {', '.join(top_3_params)}")
print(f"✅ Méthode d'évaluation: Validation croisée 5-folds avec F1-Score macro")

print(f"\n🎉 PROJET TERMINÉ AVEC SUCCÈS!")
print("=" * 60)
st.success("🎉 Déploiement du modèle terminé avec succès !")
