import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# === TF, IDF, TF-IDF manual ===
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

# === Manual Multinomial Naive Bayes ===
class ManualMultinomialNB:
    def __init__(self):
        self.class_log_prior_ = None       # np.array, shape (n_classes,)
        self.feature_log_prob_ = None      # np.array, shape (n_classes, n_features)
        self.classes_ = None               # np.array, shape (n_classes,)

    def predict(self, X):
        X = np.array(X)
        joint_log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_
        indices = np.argmax(joint_log_likelihood, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        X = np.array(X)
        joint_log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_
        # Stabilize softmax with max trick
        max_ll = np.max(joint_log_likelihood, axis=1, keepdims=True)
        exp_ll = np.exp(joint_log_likelihood - max_ll)
        probs = exp_ll / np.sum(exp_ll, axis=1, keepdims=True)
        return probs

# === Fungsi Load model ===
@st.cache_resource(show_spinner=False)
def load_model_dict():
    with open('model_sentimen.pkl', 'rb') as f:
        model_dict = pickle.load(f)

    # Aspect classifier
    model = model_dict['aspect_classifier']['model']
    if not isinstance(model.class_log_prior_, np.ndarray):
        model.class_log_prior_ = np.array(model.class_log_prior_)
    if not isinstance(model.feature_log_prob_, np.ndarray):
        model.feature_log_prob_ = np.array(model.feature_log_prob_)
    if not isinstance(model.classes_, np.ndarray):
        model.classes_ = np.array(model.classes_)

    # Fix vocab_index dan idf jika perlu
    vocab = model_dict['aspect_classifier']['vocab_index']
    if isinstance(vocab, list):
        model_dict['aspect_classifier']['vocab_index'] = {term: i for i, term in enumerate(vocab)}

    idf = model_dict['aspect_classifier']['idf']
    if isinstance(idf, list):
        model_dict['aspect_classifier']['idf'] = dict(idf)  # asumsikan list of pairs

    # Sentiment models
    for aspect in model_dict['sentiment_models']:
        smodel = model_dict['sentiment_models'][aspect]
        if not isinstance(smodel.class_log_prior_, np.ndarray):
            smodel.class_log_prior_ = np.array(smodel.class_log_prior_)
        if not isinstance(smodel.feature_log_prob_, np.ndarray):
            smodel.feature_log_prob_ = np.array(smodel.feature_log_prob_)
        if not isinstance(smodel.classes_, np.ndarray):
            smodel.classes_ = np.array(smodel.classes_)

        # Sentiment vocab and idf fix
        svocab = model_dict['sentiment_vectorizers'][aspect]['vocab_index']
        if isinstance(svocab, list):
            model_dict['sentiment_vectorizers'][aspect]['vocab_index'] = {term: i for i, term in enumerate(svocab)}

        sidf = model_dict['sentiment_vectorizers'][aspect]['idf']
        if isinstance(sidf, list):
            model_dict['sentiment_vectorizers'][aspect]['idf'] = dict(sidf)

    return model_dict

# === Prediksi Aspek ===
def predict_aspect_probs(texts, model_dict):
    tokens_list = [text.lower().split() for text in texts]
    idf = model_dict['aspect_classifier']['idf']
    vocab = model_dict['aspect_classifier']['vocab_index']
    tfidf = compute_tfidf(tokens_list, idf)
    X = convert_to_array(tfidf, vocab)

    model = model_dict['aspect_classifier']['model']
    probas = model.predict_proba(X)[0]  # probabilitas semua kelas untuk satu dokumen
    classes = model.classes_

    # Buat list tuple (aspect, prob) dalam persen
    results = [(cls, prob * 100) for cls, prob in zip(classes, probas)]
    return results
    
# === Prediksi Sentimen ===
def predict_sentiment_probs(text, aspect, model_dict):
    tokens = text.lower().split()
    idf = model_dict['sentiment_vectorizers'][aspect]['idf']
    vocab = model_dict['sentiment_vectorizers'][aspect]['vocab_index']
    tfidf = compute_tfidf([tokens], idf)
    X = convert_to_array(tfidf, vocab)

    model = model_dict['sentiment_models'][aspect]
    probas = model.predict_proba(X)[0]
    classes = model.classes_

    results = [(cls, prob * 100) for cls, prob in zip(classes, probas)]
    return results

# === Streamlit UI ===
st.set_page_config(page_title="ABSA - Tourist Reviews", layout="wide")  # layout wide agar penuh layar
st.title("📊 Aspect Based Sentiment Analysis - Tourist Review")

user_input = st.text_area("Masukkan kalimat review:", height=150)
model_dict = load_model_dict()

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:
        # Ambil semua probabilitas aspek
        aspect_probs = predict_aspect_probs([user_input], model_dict)

        # Ambil aspek dengan probabilitas tertinggi
        aspect_pred, aspect_conf = max(aspect_probs, key=lambda x: x[1])

        # Prediksi sentimen untuk aspek terpilih
        sentiment_probs = predict_sentiment_probs(user_input, aspect_pred, model_dict)
        sentiment_label, sentiment_conf = max(sentiment_probs, key=lambda x: x[1])

        st.subheader("Hasil Prediksi:")
        st.markdown(f"**🔷 Aspect Terpilih:** {aspect_pred} ({aspect_conf:.2f}%)")
        st.markdown(f"**🗣️ Sentiment Terpilih:** {sentiment_label} ({sentiment_conf:.2f}%)")

        # Pie chart untuk aspek
        labels_aspect = [ap[0] for ap in aspect_probs]
        sizes_aspect = [ap[1] for ap in aspect_probs]

        # Pie chart untuk sentimen
        labels_sentiment = [sp[0] for sp in sentiment_probs]
        sizes_sentiment = [sp[1] for sp in sentiment_probs]

        cols = st.columns(2)

        with cols[0]:
            st.subheader("Distribusi Probabilitas Aspek")
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes_aspect, labels=labels_aspect, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax1.axis('equal')
            st.pyplot(fig1)
        
        with cols[1]:
            st.subheader(f"Distribusi Probabilitas Sentimen (Aspek: {aspect_pred})")
            fig2, ax2 = plt.subplots()
            ax2.pie(sizes_sentiment, labels=labels_sentiment, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
            ax2.axis('equal')
            st.pyplot(fig2)