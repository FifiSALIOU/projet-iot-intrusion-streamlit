import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
        with open("random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open("logistic_regression_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
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
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choisissez une option:",
        ["🔍 Prédiction Simple", "📊 Prédiction par Batch", "📈 Statistiques des Modèles"]
    )

    N_BAIOT_FEATURES = [
        'MI_dir_L5_weight', 'MI_dir_L5_mean', 'MI_dir_L5_variance',
        'MI_dir_L3_weight', 'MI_dir_L3_mean', 'MI_dir_L3_variance',
        'MI_dir_L1_weight', 'MI_dir_L1_mean', 'MI_dir_L1_variance',
        'MI_dir_L0.1_weight'
    ]

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
            for i in range(5, len(N_BAIOT_FEATURES)):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(
                    label=feat,
                    min_value=0.0,
                    value=DEFAULT_VALUES[feat],
                    step=0.01,
                    format="%.4f"
                )

        selected_model = st.selectbox("Sélectionnez le modèle à utiliser:", list(models.keys()))
        model = models[selected_model]

        if st.button("🔍 Analyser", type="primary"):
            input_array = np.array([[input_data[feat] for feat in N_BAIOT_FEATURES]])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            try:
                probability = model.predict_proba(input_scaled)[0]
            except AttributeError:
                probability = [0.0, 0.0]  # fallback si pas de predict_proba

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("🚨 **INTRUSION DÉTECTÉE**")
                else:
                    st.success("✅ **TRAFIC NORMAL**")

            with col2:
                st.metric("Probabilité Normal", f"{probability[0]:.2%}")
                st.metric("Probabilité Intrusion", f"{probability[1]:.2%}")

            with col3:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Normal', 'Intrusion'], probability, color=['#2E8B57', '#DC143C'])
                ax.set_ylim(0, 1)
                for i, v in enumerate(probability):
                    ax.text(i, v + 0.02, f'{v:.2%}', ha='center')
                ax.set_title(f'Probabilités de Classification ({selected_model})')
                st.pyplot(fig)

    elif option == "📊 Prédiction par Batch":
        st.header("📊 Analyse de Fichier CSV")
        st.markdown("Uploadez un fichier CSV pour analyser plusieurs échantillons.")
        with st.expander("ℹ️ Format de fichier requis"):
            st.write("Le fichier CSV doit contenir les colonnes suivantes:")
            st.code(", ".join(N_BAIOT_FEATURES))

        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé: {df.shape[0]} lignes")
                if set(N_BAIOT_FEATURES).issubset(df.columns):
                    selected_model = st.selectbox("Sélectionnez le modèle à utiliser:", list(models.keys()))
                    model = models[selected_model]

                    if st.button("🔍 Analyser le fichier", type="primary"):
                        X = df[N_BAIOT_FEATURES]
                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)

                        try:
                            probabilities = model.predict_proba(X_scaled)
                        except AttributeError:
                            probabilities = np.zeros((X.shape[0], 2))

                        df["Statut"] = np.where(predictions == 1, "Intrusion", "Normal")
                        df["Probabilité_Normal"] = probabilities[:, 0]
                        df["Probabilité_Intrusion"] = probabilities[:, 1]

                        st.subheader("📈 Statistiques")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total", len(df))
                        col2.metric("Normaux", (df["Statut"] == "Normal").sum())
                        col3.metric("Intrusions", (df["Statut"] == "Intrusion").sum())

                        st.subheader("📊 Distribution des prédictions")
                        fig1, ax1 = plt.subplots()
                        df["Statut"].value_counts().plot.pie(autopct="%.1f%%", colors=['#2E8B57', '#DC143C'], ax=ax1)
                        st.pyplot(fig1)

                        fig2, ax2 = plt.subplots()
                        ax2.hist(df["Probabilité_Intrusion"], bins=20, color='orange')
                        ax2.set_title("Distribution des Probabilités d'Intrusion")
                        st.pyplot(fig2)

                        st.subheader("📋 Résultats")
                        st.dataframe(df)

                        st.download_button(
                            label="💾 Télécharger les résultats",
                            data=df.to_csv(index=False),
                            file_name="resultats_analyse.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Colonnes manquantes dans le fichier.")
            except Exception as e:
                st.error(f"Erreur de lecture: {str(e)}")

    elif option == "📈 Statistiques des Modèles":
        st.header("📈 Informations sur les Modèles")
        for model_name, model in models.items():
            st.subheader(f"🔍 {model_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Type:", type(model).__name__)
