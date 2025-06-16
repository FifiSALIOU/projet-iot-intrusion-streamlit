import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Intrusions IoT",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🔒 Système de Détection d'Intrusions Réseau IoT")
st.markdown("""
**Classifiez le trafic réseau comme normal ou malveillant**  
Basé sur le dataset N-BaIoT (UCI Machine Learning Repository)
""")
st.markdown("---")

# Fonction pour charger les modèles
@st.cache_resource
def load_models_and_scaler():
    try:
        # Charger les 3 modèles
        with open("random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open("logistic_regression_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
        
        # Charger le scaler et les features
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
            
        return {
            "Random Forest": rf_model,
            "SVM": svm_model,
            "Logistic Regression": lr_model
        }, scaler, features
        
    except FileNotFoundError:
        st.error("⚠️ Modèles non trouvés. Veuillez d'abord exécuter le script d'entraînement.")
        return None, None, None

# Chargement des modèles
models, scaler, features = load_models_and_scaler()

if models is not None:
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choisissez une option:",
        ["🔍 Prédiction Simple", "📊 Prédiction par Batch", "📈 Statistiques des Modèles"]
    )

    # Liste des features N-BaIoT
    N_BAIOT_FEATURES = [
        'MI_dir_L5_weight', 'MI_dir_L5_mean', 'MI_dir_L5_variance',
        'MI_dir_L3_weight', 'MI_dir_L3_mean', 'MI_dir_L3_variance',
        'MI_dir_L1_weight', 'MI_dir_L1_mean', 'MI_dir_L1_variance',
        'MI_dir_L0.1_weight'
    ]

    # Valeurs réalistes par défaut
    DEFAULT_VALUES = {
        'MI_dir_L5_weight': 0.35,
        'MI_dir_L5_mean': 0.02,
        'MI_dir_L5_variance': 0.001,
        'MI_dir_L3_weight': 0.28,
        'MI_dir_L3_mean': 0.015,
        'MI_dir_L3_variance': 0.0008,
        'MI_dir_L1_weight': 0.15,
        'MI_dir_L1_mean': 0.01,
        'MI_dir_L1_variance': 0.0005,
        'MI_dir_L0.1_weight': 0.05
    }

    if option == "🔍 Prédiction Simple":
        st.header("🔍 Prédiction pour un Échantillon")

    st.markdown("Saisissez les caractéristiques du trafic réseau pour détecter une intrusion.")

        # Création de colonnes pour l'interface
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            st.subheader("📡 Caractéristiques du Trafic")
            for i in range(5):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(
                    label=feat,
                    min_value=0.0,
                    value=DEFAULT_VALUES[feat],
                    step=0.01,
                    format="%.4f"
                )
        
        with col2:
            st.subheader("")
            for i in range(5, len(N_BAIOT_FEATURES)):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(
                    label=feat,
                    min_value=0.0,
                    value=DEFAULT_VALUES[feat],
                    step=0.01,
                    format="%.4f"
                )
            
        # Sélection du modèle
        selected_model = st.selectbox(
            "Sélectionnez le modèle à utiliser:",
            list(models.keys())
        )
        model = models[selected_model]
        
        # Bouton de prédiction
        if st.button("🔍 Analyser", type="primary"):
            # Création du vecteur de caractéristiques
            input_array = np.array([[input_data[feat] for feat in N_BAIOT_FEATURES]])
            
            # Normalisation
            input_scaled = scaler.transform(input_array)
            
            # Prédiction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Affichage des résultats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **INTRUSION DÉTECTÉE**")
                    st.markdown("⚠️ Trafic malveillant identifié")
                else:
                    st.success("✅ **TRAFIC NORMAL**")
                    st.markdown("🔒 Aucune menace détectée")
            
            with col2:
                st.metric("Probabilité Normal", f"{probability[0]:.2%}")
                st.metric("Probabilité Intrusion", f"{probability[1]:.2%}")
            
            with col3:
                # Graphique de probabilité
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Normal', 'Intrusion']
                colors = ['#2E8B57', '#DC143C']
                ax.bar(labels, probability, color=colors, alpha=0.7)
                ax.set_ylabel('Probabilité')
                ax.set_title(f'Probabilités de Classification ({selected_model})')
                ax.set_ylim(0, 1)
                for i, v in enumerate(probability):
                    ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
                st.pyplot(fig)

    elif option == "📊 Prédiction par Batch":
        st.header("📊 Analyse de Fichier CSV")
        st.markdown("Choisissez un fichier CSV local ou uploadez un fichier CSV pour analyser plusieurs échantillons.")
        
        # Information sur le format attendu
        with st.expander("ℹ️ Format de fichier requis"):
            st.write("Le fichier CSV doit contenir les colonnes suivantes:")
            st.write(", ".join(N_BAIOT_FEATURES))
            st.write("Exemple de fichier: [Télécharger un exemple](https://example.com/sample.csv)")
        
        # Liste des fichiers CSV présents dans le dossier courant
        files_dispo = [f for f in os.listdir() if f.endswith(".csv")]
        selected_local_file = st.selectbox("Ou choisissez un fichier CSV local disponible:", files_dispo)
        
        uploaded_file = st.file_uploader("Ou uploadez un fichier CSV", type="csv")
        
        df = None
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé via upload : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier uploadé: {e}")
        elif selected_local_file:
            try:
                df = pd.read_csv(selected_local_file)
                st.success(f"✅ Fichier local chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier local: {e}")
        
        if df is not None:
            # Vérification des colonnes
            if set(N_BAIOT_FEATURES).issubset(set(df.columns)):
                # Sélection du modèle
                selected_model = st.selectbox(
                    "Sélectionnez le modèle à utiliser:",
                    list(models.keys())
                )
                model = models[selected_model]
                
                if st.button("🔍 Analyser le fichier", type="primary"):
                    # Préparation des données
                    X = df[N_BAIOT_FEATURES]
                    X_scaled = scaler.transform(X)
                    
                    # Prédictions
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)
                    
                    # Ajout des résultats au DataFrame
                    df['Prédiction'] = predictions
                    df['Probabilité_Normal'] = probabilities[:, 0]
                    df['Probabilité_Intrusion'] = probabilities[:, 1]
                    df['Statut'] = df['Prédiction'].map({0: 'Normal', 1: 'Intrusion'})
                    
                    # Statistiques
                    st.subheader("📈 Résultats de l'Analyse")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total = len(df)
                        st.metric("Total Échantillons", total)
                    
                    with col2:
                        normal_count = sum(predictions == 0)
                        st.metric("Trafic Normal", f"{normal_count} ({normal_count/total*100:.1f}%)")
                    
                    with col3:
                        intrusion_count = sum(predictions == 1)
                        st.metric("Intrusions Détectées", f"{intrusion_count} ({intrusion_count/total*100:.1f}%)")
                    
                    # Graphiques
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        df['Statut'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                                                       colors=['#2E8B57', '#DC143C'])
                        ax.set_title('Distribution des Prédictions')
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(df['Probabilité_Intrusion'], bins=20, alpha=0.7, color='orange')
                        ax.set_xlabel('Probabilité d\'Intrusion')
                        ax.set_ylabel('Fréquence')
                        ax.set_title('Distribution des Probabilités d\'Intrusion')
                        st.pyplot(fig)
                    
                    # Affichage des résultats détaillés
                    st.subheader("📋 Résultats Détaillés")
                    st.dataframe(df[['Statut', 'Probabilité_Normal', 'Probabilité_Intrusion'] + N_BAIOT_FEATURES])
                    
                    # Téléchargement des résultats
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Télécharger les résultats",
                        data=csv,
                        file_name="resultats_analyse.csv",
                        mime="text/csv"
                    )
                    
            else:
                st.error("❌ Le fichier ne contient pas les colonnes requises.")
                st.write("Colonnes requises:", N_BAIOT_FEATURES)
                st.write("Colonnes trouvées:", list(df.columns))

    elif option == "📈 Statistiques des Modèles":
        st.header("📈 Informations sur les Modèles")
        
        # Afficher les statistiques pour chaque modèle
        for model_name, model in models.items():
            st.subheader(f"🤖 Modèle: {model_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type de modèle:** {type(model).__name__}")
                
                # Afficher les paramètres spécifiques au modèle
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.write("**Paramètres:**")
                    for key, value in list(params.items())[:5]:  # Afficher les 5 premiers
                        st.text(f"{key}: {value}")
            
            with col2:
                # Importance des features (si disponible)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("🎯 Importance des Features")
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                    ax.set_title('Top 10 des Features Importantes')
                    st.pyplot(fig)
            
            st.markdown("---")

    # Sidebar avec informations supplémentaires
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ À Propos")
    st.sidebar.info(
        "Cette application utilise des techniques de Machine Learning "
        "pour détecter les intrusions dans les réseaux IoT basée sur le dataset N-BaIoT."
    )
    
else:
    st.error("⚠️ Impossible de charger les modèles. Assurez-vous d'avoir exécuté le script d'entraînement.")
    st.info("💡 Exécutez d'abord le script train_model.py pour générer les fichiers nécessaires")
