import streamlit as st
import pickle
import numpy as np

# Fungsi untuk load masing-masing file model/vectorizer/encoder
@st.cache_resource
def load_models():
    models = {}
    aspects = ['accessibility', 'facility', 'activity']
    for aspect in aspects:
        with open(f'model_{aspect}.pkl', 'rb') as f_model, \
             open(f'vectorizer_{aspect}.pkl', 'rb') as f_vec, \
             open(f'label_encoder_{aspect}.pkl', 'rb') as f_le:
            
            models[aspect] = {
                'model': pickle.load(f_model),
                'vectorizer': pickle.load(f_vec),
                'label_encoder': pickle.load(f_le)
            }
    return models

# Fungsi prediksi multi-aspek dengan probabilitas
def predict_sentiments(text, models):
    results = {}
    for aspect, components in models.items():
        vectorizer = components['vectorizer']
        model = components['model']
        label_encoder = components['label_encoder']

        transformed = vectorizer.transform([text]).toarray()
        proba = model.predict_proba(transformed)[0]
        predicted_index = np.argmax(proba)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        confidence = round(proba[predicted_index] * 100, 2)
        
        results[aspect] = {
            'label': predicted_label,
            'confidence': confidence
        }
    return results

# === Streamlit UI ===
st.set_page_config(page_title="Aspect Based Sentiment Analysis (ABSA)", layout="centered")
st.title("📊 Aspect Based Sentiment Analysis - Tourist Review")

# Input teks dari user
user_input = st.text_area("Masukkan kalimat review:", height=150)

# Load semua model sekali saja
models = load_models()

# Tombol prediksi
if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        hasil = predict_sentiments(user_input, models)
        st.subheader("Hasil Sentimen:")
        for aspek, info in hasil.items():
            st.markdown(f"**{aspek.capitalize()}**: {info['label']} ({info['confidence']}%)")