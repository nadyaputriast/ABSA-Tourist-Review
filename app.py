import streamlit as st
import pickle
import numpy as np
from collections import Counter

# Fungsi manual TF, IDF, TF-IDF
def compute_tf(documents):
    tf = []
    for doc in documents:
        term_count = Counter(doc)
        doc_len = len(doc)
        tf.append({term: count / doc_len for term, count in term_count.items()})
    return tf

def compute_tfidf(documents, idf):
    tfidf = []
    for doc_tf in compute_tf(documents):
        tfidf.append({term: tf * idf.get(term, 0) for term, tf in doc_tf.items()})
    return tfidf

def convert_to_array(tfidf, vocabulary):
    vectors = []
    for doc in tfidf:
        vector = [doc.get(term, 0) for term in vocabulary]
        vectors.append(vector)
    return np.array(vectors)

class ManualMultinomialNB:
    def __init__(self):
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = []

    def predict(self, X):
        result = []
        for x in X:
            scores = {}
            for c in self.classes_:
                score = self.class_log_prior_[c]
                for i, val in enumerate(x):
                    score += val * self.feature_log_prob_[c][i]
                scores[c] = score
            pred = max(scores, key=scores.get)
            result.append(pred)
        return np.array(result)

    def predict_proba(self, X):
        result = []
        for x in X:
            scores = {}
            for c in self.classes_:
                score = self.class_log_prior_[c]
                for i, val in enumerate(x):
                    score += val * self.feature_log_prob_[c][i]
                scores[c] = score
            # Convert to probabilities
            exp_scores = {c: np.exp(score) for c, score in scores.items()}
            total = sum(exp_scores.values())
            probas = {c: exp_scores[c] / total for c in exp_scores}
            ordered = [probas[c] for c in sorted(self.classes_)]
            result.append(ordered)
        return np.array(result)
        
# Load model dan komponen lain sekali saja
@st.cache_resource
def load_models():
    models = {}
    aspects = ['accessibility', 'facility', 'activity']
    for aspect in aspects:
        with open(f'model_{aspect}.pkl', 'rb') as f_model, \
             open(f'idf_{aspect}.pkl', 'rb') as f_idf, \
             open(f'vocab_{aspect}.pkl', 'rb') as f_vocab, \
             open(f'label_encoder_{aspect}.pkl', 'rb') as f_le:
            
            models[aspect] = {
                'model': pickle.load(f_model),
                'idf': pickle.load(f_idf),
                'vocab': pickle.load(f_vocab),
                'label_encoder': pickle.load(f_le)
            }
    return models

# Prediksi manual menggunakan TF-IDF dan model yang sudah dilatih
def predict_sentiments(text, models):
    results = {}
    tokens = text.lower().split()  # Pastikan sesuai dengan preprocessing kamu

    for aspect, components in models.items():
        idf = components['idf']
        vocab = components['vocab']
        model = components['model']
        label_encoder = components['label_encoder']

        tfidf_vector = compute_tfidf([tokens], idf)
        X_input = convert_to_array(tfidf_vector, vocab)

        y_pred = model.predict(X_input)[0]
        y_proba = model.predict_proba(X_input)[0]

        label = label_encoder['index_to_label'][y_pred]
        confidence = round(y_proba[y_pred] * 100, 2)

        results[aspect] = {
            'label': label,
            'confidence': confidence
        }
    return results

# === Streamlit UI ===
st.set_page_config(page_title="Aspect Based Sentiment Analysis (ABSA)", layout="centered")
st.title("📊 Aspect Based Sentiment Analysis - Tourist Review")

# Input dari user
user_input = st.text_area("Masukkan kalimat review:", height=150)

# Load model sekali saja
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