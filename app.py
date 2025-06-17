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

# Chargement
models, scaler, features = load_models_and_scaler()

# Si chargement OK
if models:
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
        'MI_dir_L5_weight': 0.35, 'MI_dir_L5_mean': 0.02, 'MI_dir_L5_variance': 0.001,
        'MI_dir_L3_weight': 0.28, 'MI_dir_L3_mean': 0.015, 'MI_dir_L3_variance': 0.0008,
        'MI_dir_L1_weight': 0.15, 'MI_dir_L1_mean': 0.01, 'MI_dir_L1_variance': 0.0005,
        'MI_dir_L0.1_weight': 0.05
    }

    if option == "🔍 Prédiction Simple":
        st.header("🔍 Prédiction pour un Échantillon")
        col1, col2 = st.columns(2)
        input_data = {}

        with col1:
            for i in range(5):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(feat, 0.0, value=DEFAULT_VALUES[feat], step=0.01, format="%.4f")
        with col2:
            for i in range(5, len(N_BAIOT_FEATURES)):
                feat = N_BAIOT_FEATURES[i]
                input_data[feat] = st.number_input(feat, 0.0, value=DEFAULT_VALUES[feat], step=0.01, format="%.4f")

        selected_model = st.selectbox("Sélectionnez le modèle à utiliser:", list(models.keys()), key="simple_model")
        model = models[selected_model]

        if st.button("🔍 Analyser", type="primary"):
            input_array = np.array([[input_data[feat] for feat in N_BAIOT_FEATURES]])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

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
                fig, ax = plt.subplots()
                ax.bar(['Normal', 'Intrusion'], probability, color=['#2E8B57', '#DC143C'])
                ax.set_ylim(0, 1)
                st.pyplot(fig)

    elif option == "📊 Prédiction par Batch":
        st.header("📊 Analyse de Fichier CSV")
        st.markdown("Uploadez un fichier CSV pour analyser plusieurs échantillons.")

        with st.expander("ℹ️ Format de fichier requis"):
            st.write("Colonnes requises:", ", ".join(N_BAIOT_FEATURES))

        uploaded_file = st.file_uploader("📁 Choisissez un fichier CSV", type="csv")
        if uploaded_file is not None:
            st.write(f"Taille du fichier uploadé : {uploaded_file.size / (1024*1024):.2f} Mo")

            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes chargées.")
                if set(N_BAIOT_FEATURES).issubset(df.columns):
                    selected_model = st.selectbox("Sélectionnez le modèle à utiliser:", list(models.keys()), key="batch_model")
                    model = models[selected_model]

                    if st.button("🔍 Analyser le fichier", type="primary"):
                        X = df[N_BAIOT_FEATURES]
                        X_scaled = scaler.transform(X)
                        preds = model.predict(X_scaled)
                        probas = model.predict_proba(X_scaled)

                        df['Prédiction'] = preds
                        df['Probabilité_Normal'] = probas[:, 0]
                        df['Probabilité_Intrusion'] = probas[:, 1]
                        df['Statut'] = df['Prédiction'].map({0: 'Normal', 1: 'Intrusion'})

                        # Stats
                        st.subheader("📈 Résultats de l'Analyse")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Échantillons", len(df))
                        with col2:
                            st.metric("Trafic Normal", f"{(df['Prédiction'] == 0).sum()} ({(df['Prédiction'] == 0).mean()*100:.1f}%)")
                        with col3:
                            st.metric("Intrusions", f"{(df['Prédiction'] == 1).sum()} ({(df['Prédiction'] == 1).mean()*100:.1f}%)")

                        # Graphs
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1, ax1 = plt.subplots()
                            df['Statut'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=['#2E8B57', '#DC143C'])
                            ax1.set_ylabel("")
                            ax1.set_title("Répartition des prédictions")
                            st.pyplot(fig1)

                        with col2:
                            fig2, ax2 = plt.subplots()
                            ax2.hist(df['Probabilité_Intrusion'], bins=20, color='orange', alpha=0.7)
                            ax2.set_title("Distribution des probabilités d'intrusion")
                            st.pyplot(fig2)

                        st.subheader("📋 Détails")
                        st.dataframe(df[['Statut', 'Probabilité_Normal', 'Probabilité_Intrusion'] + N_BAIOT_FEATURES])

                        csv = df.to_csv(index=False)
                        st.download_button("💾 Télécharger les résultats", data=csv, file_name="resultats.csv", mime="text/csv")
                else:
                    st.error("❌ Le fichier ne contient pas toutes les colonnes requises.")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier : {e}")

    elif option == "📈 Statistiques des Modèles":
        st.header("📈 Informations sur les Modèles")
        for model_name, model in models.items():
            st.subheader(f"🤖 Modèle: {model_name}")
            st.write(f"**Type**: {type(model).__name__}")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                st.write("**Paramètres:**")
                for key, val in list(params.items())[:5]:
                    st.text(f"{key}: {val}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Importance des Features")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)
            elif hasattr(model, 'coef_'):
                st.subheader("🎯 Importance des Features (coefs)")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': np.abs(model.coef_[0])
                }).sort_values('Importance', ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                st.pyplot(fig)
            else:
                st.info("ℹ️ Ce modèle ne fournit pas d'importances de features.")

    st.sidebar.markdown("---")
    st.sidebar.info("Cette application détecte les intrusions dans les réseaux IoT via Machine Learning.")
else:
    st.error("⚠️ Impossible de charger les modèles. Veuillez exécuter le script d'entraînement.")
