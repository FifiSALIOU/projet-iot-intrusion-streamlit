import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(page_title="Détection d'Intrusions IoT", layout="centered")

# Chargement des fichiers
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("🛡️ Application de Détection d'Intrusions Réseau - NB-IoT")

# Menu
option = st.sidebar.selectbox("Navigation", [
    "🏠 Accueil",
    "🔍 Prédiction Simple",
    "📁 Prédiction par Fichier",
    "📈 Statistiques du Modèle"
])

# Page d'accueil
if option == "🏠 Accueil":
    st.image("https://img.icons8.com/external-flat-juicy-fish/512/external-cybersecurity-cyber-security-flat-flat-juicy-fish.png", width=100)
    st.markdown("""
        ## Bienvenue dans l'application de détection d'intrusions NB-IoT

        Cette application permet :
        - De détecter les intrusions à partir d’un fichier ou de données manuelles.
        - D’utiliser un modèle RandomForest pour la classification.
        - D’observer les performances de plusieurs modèles.

        *Développé pour le projet de sécurité réseau IoT.*
    """)

# Page de prédiction simple
elif option == "🔍 Prédiction Simple":
    st.header("🔍 Prédiction pour un Échantillon")
    st.markdown("Veuillez entrer les caractéristiques suivantes pour faire une prédiction :")

    input_values = {}
    required_features = [
        "MI_dir_L5_weight", "MI_dir_L5_mean", "MI_dir_L5_variance",
        "MI_dir_L3_weight", "MI_dir_L3_mean", "MI_dir_L3_variance",
        "MI_dir_L1_weight", "MI_dir_L1_mean", "MI_dir_L1_variance",
        "MI_dir_L0.1_weight", "MI_dir_L0.1"
    ]

    col1, col2 = st.columns(2)

    for i, feature in enumerate(required_features):
        with (col1 if i % 2 == 0 else col2):
            value = st.number_input(f"{feature}", format="%.6f")
            input_values[feature] = value

    if st.button("🔍 Prédire", type="primary"):
        try:
            input_array = np.array([[input_values[feat] for feat in features]])
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]

            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("🚨 **INTRUSION DÉTECTÉE**")
                else:
                    st.success("✅ **TRAFIC NORMAL**")

            with col2:
                st.metric("Probabilité Normal", f"{proba[0]:.2%}")
                st.metric("Probabilité Intrusion", f"{proba[1]:.2%}")

            with col3:
                fig, ax = plt.subplots()
                ax.bar(['Normal', 'Intrusion'], proba, color=['green', 'red'])
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

# Page de prédiction par fichier
elif option == "📁 Prédiction par Fichier":
    st.header("📁 Prédiction par Fichier CSV")
    uploaded_file = st.file_uploader("Téléversez un fichier CSV contenant les colonnes suivantes :", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if set(features).issubset(df.columns):
            X = df[features]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            df["prediction"] = y_pred
            st.success("✅ Prédictions réalisées avec succès !")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger le fichier avec prédictions", data=csv, file_name="predictions.csv", mime='text/csv')

        else:
            st.error("❌ Le fichier ne contient pas les colonnes requises.")

# Page de statistiques
elif option == "📈 Statistiques du Modèle":
    st.header("📈 Comparaison des Modèles")

    try:
        rf = pickle.load(open("Random_Forest.pkl", "rb"))
        svm = pickle.load(open("SVM.pkl", "rb"))
        logreg = pickle.load(open("Logistic_Regression.pkl", "rb"))

        st.subheader("🧪 Exemple d'évaluation sur le même jeu de test")

        st.markdown("""
        Pour chaque modèle, l'accuracy ci-dessous représente la performance sur un jeu de test commun.
        """)
        
        # Il faut ici avoir X_test_scaled et y_test disponibles
        st.warning("⚠️ Pour afficher les vraies performances, intégrez `X_test_scaled` et `y_test` dans le code ou les chargez.")

        st.markdown("**Exemple fictif :**")
        st.write("- Random Forest : **96.5%**")
        st.write("- SVM : **94.2%**")
        st.write("- Logistic Regression : **91.0%**")

    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {str(e)}")
