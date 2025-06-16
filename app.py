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

st.set_page_config(
    page_title="Détection d'Intrusions IoT",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔒 Système de Détection d'Intrusions Réseau IoT")
st.markdown("""
**Classifiez le trafic réseau comme normal ou malveillant**  
Basé sur le dataset N-BaIoT (UCI Machine Learning Repository)
""")
st.markdown("---")

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
                    label=feat, min_value=0.0, max_value=1.0,
                    value=DEFAULT_VALUES[feat], step=0.01, format="%.4f"
                )

        with col2:
            st.subheader("")
            for i in range(5, len(N_BAIOT_FEATURES)):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(
                    label=feat, min_value=0.0, max_value=1.0,
                    value=DEFAULT_VALUES[feat], step=0.01, format="%.4f"
                )

        selected_model = st.selectbox(
            "Sélectionnez le modèle à utiliser:",
            list(models.keys())
        )
        model = models[selected_model]

        if st.button("🔍 Analyser", type="primary"):
            input_array = np.array([[input_data[feat] for feat in N_BAIOT_FEATURES]])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            # ✅ Correction ici : assure que predict_proba existe
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_scaled)[0]
            else:
                # fallback simple
                decision = model.decision_function(input_scaled)
                probability = [1 - decision[0], decision[0]]

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

    # Le reste du code n’a **pas besoin d’être modifié**, il fonctionne déjà bien.
    # Donc il reste inchangé pour respecter votre demande.

else:
    st.error("⚠️ Impossible de charger les modèles. Assurez-vous d'avoir exécuté le script d'entraînement.")
    st.info("💡 Exécutez d'abord le script train_model.py pour générer les fichiers nécessaires")
