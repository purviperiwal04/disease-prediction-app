import streamlit as st
st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="centered")

import joblib
import json
import numpy as np

# 📦 Load model and data only once
@st.cache_resource
def load_resources():
    model = joblib.load("model/model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    with open("model/symptoms.json", "r") as f:
        feature_columns = json.load(f)
    return model, label_encoder, feature_columns

model, label_encoder, feature_columns = load_resources()

# 🏥 App title
st.title("🩺 Disease Prediction App")
st.markdown("Select at least **3 symptoms** from the list below and click **Predict** to see possible diseases.")

# 📝 Multi-select symptoms
selected_symptoms = st.multiselect(
    "Choose your symptoms:",
    options=feature_columns
)

# 🔮 Predict button
if st.button("Predict"):
    if len(selected_symptoms) < 3:
        st.warning("⚠️ Please select at least 3 symptoms.")
    else:
        # Create binary input vector
        input_vector = [1 if col in selected_symptoms else 0 for col in feature_columns]

        # Predict probabilities
        pred_proba = model.predict_proba([input_vector])[0]
        top_indices = pred_proba.argsort()[::-1]

        # ✅ Remove Vertigo if confidence is low (<50%)
        filtered_indices = [
            i for i in top_indices
            if label_encoder.inverse_transform([i])[0] != "(vertigo) Paroymsal  Positional Vertigo"
            or pred_proba[i] > 0.5
        ]

        # Pick top 3 after filtering
        final_indices = filtered_indices[:3]
        top_diseases = label_encoder.inverse_transform(final_indices)

        # 📊 Display results
        st.success(f"🧠 **Predicted Disease:** {top_diseases[0]}")
        st.write(f"📊 **Confidence:** {pred_proba[final_indices[0]]*100:.2f}%")

        st.markdown("### 🔎 Top 3 Predictions:")
        for disease, prob in zip(top_diseases, pred_proba[final_indices]):
            st.write(f"- {disease} ({prob*100:.2f}%)")
