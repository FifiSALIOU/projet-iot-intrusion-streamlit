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
st.markdown("---")

# Fonction pour charger les modèles (avec gestion d'erreur)
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, scaler, features
    except FileNotFoundError:
        st.error("⚠️ Modèles non trouvés. Veuillez d'abord exécuter le script d'entraînement.")
        return None, None, None

# Chargement des modèles
model, scaler, features = load_model_and_scaler()

if model is not None:
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choisissez une option:",
        ["🔍 Prédiction Simple", "📊 Prédiction par Batch", "📈 Statistiques du Modèle"]
    )

    if option == "🔍 Prédiction Simple":
        st.header("🔍 Prédiction pour un Échantillon")
        st.markdown("Saisissez les caractéristiques du trafic réseau pour détecter une intrusion.")
        
        # Création de colonnes pour l'interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📡 Caractéristiques du Paquet")
            packet_size = st.number_input("Taille du paquet", min_value=0.0, max_value=5000.0, value=512.0)
            duration = st.number_input("Durée de connexion", min_value=0.0, max_value=100.0, value=2.0)
            src_bytes = st.number_input("Bytes source", min_value=0.0, max_value=10000.0, value=1024.0)
            dst_bytes = st.number_input("Bytes destination", min_value=0.0, max_value=10000.0, value=768.0)
            
            st.subheader("🌐 Protocoles")
            protocol_tcp = st.selectbox("Protocole TCP", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            protocol_udp = st.selectbox("Protocole UDP", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            
        with col2:
            st.subheader("⚙️ Services")
            service_http = st.selectbox("Service HTTP", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            service_ftp = st.selectbox("Service FTP", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            
            st.subheader("🔐 Sécurité")
            num_failed_logins = st.number_input("Tentatives de connexion échouées", min_value=0, max_value=20, value=0)
            logged_in = st.selectbox("Connecté", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            num_compromised = st.number_input("Conditions compromises", min_value=0, max_value=10, value=0)
            root_shell = st.selectbox("Accès root shell", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            su_attempted = st.selectbox("Tentative su", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
            
        # Autres paramètres dans une section repliable
        with st.expander("Paramètres Avancés"):
            urgent = st.number_input("Urgence", min_value=0, max_value=10, value=0)
            hot = st.number_input("Indicateur hot", min_value=0, max_value=20, value=0)
            num_root = st.number_input("Accès root", min_value=0, max_value=10, value=0)
            num_file_creations = st.number_input("Créations de fichiers", min_value=0, max_value=50, value=0)
            count = st.number_input("Count", min_value=1, max_value=500, value=1)
            srv_count = st.number_input("Service count", min_value=1, max_value=200, value=1)
            dst_host_count = st.number_input("Destination host count", min_value=1, max_value=255, value=1)
        
        # Bouton de prédiction
        if st.button("🔍 Analyser", type="primary"):
            # Création du vecteur de caractéristiques
            input_data = np.array([[
                packet_size, duration, src_bytes, dst_bytes, protocol_tcp, protocol_udp,
                service_http, service_ftp, urgent, hot, num_failed_logins, logged_in,
                num_compromised, root_shell, su_attempted, num_root, num_file_creations,
                count, srv_count, dst_host_count
            ]])
            
            # Normalisation
            input_scaled = scaler.transform(input_data)
            
            # Prédiction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Affichage des résultats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **INTRUSION DÉTECTÉE**")
                    st.markdown("⚠️ Trafic suspect identifié")
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
                ax.set_title('Probabilités de Classification')
                ax.set_ylim(0, 1)
                for i, v in enumerate(probability):
                    ax.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
                st.pyplot(fig)

    elif option == "📊 Prédiction par Batch":
        st.header("📊 Analyse de Fichier CSV")
        st.markdown("Uploadez un fichier CSV pour analyser plusieurs échantillons.")
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                
                # Affichage des premières lignes
                st.subheader("👀 Aperçu des données")
                st.dataframe(df.head())
                
                # Vérification des colonnes
                if set(features).issubset(set(df.columns)):
                    if st.button("🔍 Analyser le fichier", type="primary"):
                        # Préparation des données
                        X = df[features]
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
                        st.dataframe(df[['Statut', 'Probabilité_Normal', 'Probabilité_Intrusion']])
                        
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
                    st.write("Colonnes requises:", features)
                    st.write("Colonnes trouvées:", list(df.columns))
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")

    elif option == "📈 Statistiques du Modèle":
        st.header("📈 Informations sur le Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Détails du Modèle")
            st.write(f"**Type de modèle:** {type(model).__name__}")
            st.write(f"**Nombre de features:** {len(features)}")
            
            if hasattr(model, 'n_estimators'):
                st.write(f"**Nombre d'estimateurs:** {model.n_estimators}")
            if hasattr(model, 'max_depth'):
                st.write(f"**Profondeur maximale:** {model.max_depth}")
            if hasattr(model, 'C'):
                st.write(f"**Paramètre C:** {model.C}")
        
        with col2:
            st.subheader("📊 Features Utilisées")
            features_df = pd.DataFrame({'Feature': features})
            st.dataframe(features_df, use_container_width=True)
        
        # Importance des features (si disponible)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Importance des Features")
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
            ax.set_title('Importance des Features')
            st.pyplot(fig)
            
            st.dataframe(importance_df)

    # Sidebar avec informations supplémentaires
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ À Propos")
    st.sidebar.info(
        "Cette application utilise des techniques de Machine Learning "
        "pour détecter les intrusions dans les réseaux IoT."
    )
    
    st.sidebar.markdown("### 🔧 Paramètres du Modèle")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for key, value in list(params.items())[:5]:  # Afficher seulement les 5 premiers
            st.sidebar.text(f"{key}: {value}")

else:
    st.error("⚠️ Impossible de charger les modèles. Assurez-vous d'avoir exécuté le script d'entraînement.")
    st.info("💡 Exécutez d'abord votre notebook Kaggle pour générer les fichiers model.pkl, scaler.pkl et features.pkl")
