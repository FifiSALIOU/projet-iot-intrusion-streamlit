import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
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
        st.markdown("Uploadez un fichier CSV pour analyser plusieurs échantillons.")
        
        # Info sur le format
        with st.expander("ℹ️ Format de fichier requis", expanded=False):
            st.write("Le fichier CSV doit contenir **exactement** les colonnes suivantes :")
            st.code(", ".join(N_BAIOT_FEATURES), language='text')

        uploaded_file = st.file_uploader("📁 Choisissez un fichier CSV", type="csv", key="file_uploader_batch")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if not set(N_BAIOT_FEATURES).issubset(df.columns):
                    st.error("❌ Le fichier ne contient pas les colonnes requises.")
                    st.write("Colonnes requises :", N_BAIOT_FEATURES)
                    st.write("Colonnes fournies :", list(df.columns))
                else:
                    st.success(f"✅ Fichier chargé avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")

                    selected_model = st.selectbox("🧠 Sélectionnez le modèle :", list(models.keys()), key="model_batch")
                    model = models[selected_model]

                    if st.button("🔍 Lancer l'analyse", key="analyze_batch"):
                        # Préparation
                        X = df[N_BAIOT_FEATURES]
                        X_scaled = scaler.transform(X)

                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)

                        df['Prédiction'] = predictions
                        df['Probabilité_Normal'] = probabilities[:, 0]
                        df['Probabilité_Intrusion'] = probabilities[:, 1]
                        df['Statut'] = df['Prédiction'].map({0: 'Normal', 1: 'Intrusion'})

                        # Résumé des stats
                        st.subheader("📈 Statistiques de Résultats")
                        col1, col2, col3 = st.columns(3)
                        total = len(df)
                        normal = (predictions == 0).sum()
                        intrusion = (predictions == 1).sum()

                        with col1:
                            st.metric("Total", total)
                        with col2:
                            st.metric("Normal", f"{normal} ({normal/total:.1%})")
                        with col3:
                            st.metric("Intrusions", f"{intrusion} ({intrusion/total:.1%})")

                        # Graphiques
                        col4, col5 = st.columns(2)

                        with col4:
                            fig1, ax1 = plt.subplots()
                            df['Statut'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#2E8B57', '#DC143C'], ax=ax1)
                            ax1.set_ylabel("")
                            ax1.set_title("Distribution des Statuts")
                            st.pyplot(fig1)

                        with col5:
                            fig2, ax2 = plt.subplots()
                            ax2.hist(df['Probabilité_Intrusion'], bins=20, color='orange', alpha=0.7)
                            ax2.set_title("Distribution des Probabilités d'Intrusion")
                            ax2.set_xlabel("Probabilité")
                            ax2.set_ylabel("Fréquence")
                            st.pyplot(fig2)

                        # Table
                        st.subheader("📋 Détails")
                        st.dataframe(df[['Statut', 'Probabilité_Normal', 'Probabilité_Intrusion'] + N_BAIOT_FEATURES])

                        # Téléchargement
                        csv_result = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Télécharger les résultats",
                            data=csv_result,
                            file_name="resultats_intrusion.csv",
                            mime="text/csv",
                            key="download_batch"
                        )

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")

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