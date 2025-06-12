# app.py

import streamlit as st
import numpy as np
import pickle

# Chargement des objets
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("🔐 Détection d'intrusion IoT")

# Formulaire utilisateur
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", format="%.4f")
    user_input.append(value)

if st.button("Prédire"):
    # Mise en forme et normalisation
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    result = "🔴 Intrusion détectée" if prediction[0] == 1 else "🟢 Trafic normal"
    st.success(f"Résultat de la prédiction : {result}")
