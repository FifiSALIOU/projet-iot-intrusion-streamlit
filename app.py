import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Chargement
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

st.title("🔐 Détection d'intrusion IoT")

inputs = []
for feature in features:
    value = st.number_input(f"{feature}", format="%.4f")
    inputs.append(value)

if st.button("Prédire"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1]
    else:
        proba = 0.5  # valeur neutre par défaut

    # Affichage du résultat
    if prediction == 1:
        st.error("🔴 Intrusion détectée")
    else:
        st.success("🟢 Trafic normal")

    # Graphique de probabilité
    fig, ax = plt.subplots()
    bars = ax.bar(["Normal", "Intrusion"], [1 - proba, proba], color=["green", "red"])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probabilité")
    ax.set_title("Prédiction du modèle")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    st.pyplot(fig)
