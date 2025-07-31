import streamlit as st
import os
import joblib
import zipfile
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
import gdown

# --- Fungsi Load Resource (dengan gdown dan cache) ---
@st.cache_resource
def load_all_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    os.makedirs('models', exist_ok=True)
    os.makedirs('bert_model_demo', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    file_ids = {
        "rf": "105xu-FQYHViUEmAcACJPaBwIO7CI8nhp",
        "svm": "1SYFrDHRp96Fa51BajwggaIY5ONytKLlN",
        "knn": "1jhk0feNUrNA058WcOyo3Wpzh159EBZID",
        "bert_zip": "1DNXDvX3I7r-mqspkdnCnx4IiLinjNWUl",
        "slang": "1AerLVwpX9eGXkKIcNGcJ8FFwEJ6XBbmA"
    }

    paths = {
        "rf": "models/rf_model_demo.joblib",
        "svm": "models/svm_model_demo.joblib",
        "knn": "models/knn_model_demo.joblib",
        "bert_zip": "bert_model_demo.zip",
        "bert_dir": "bert_model_demo",
        "slang": "data/slang.txt"
    }

    with st.spinner("Mempersiapkan model saat pertama kali dijalankan... Ini mungkin memakan waktu beberapa menit."):
        if not os.path.exists(paths["rf"]): gdown.download(id=file_ids["rf"], output=paths["rf"], quiet=True)
        if not os.path.exists(paths["svm"]): gdown.download(id=file_ids["svm"], output=paths["svm"], quiet=True)
        if not os.path.exists(paths["knn"]): gdown.download(id=file_ids["knn"], output=paths["knn"], quiet=True)
        if not os.path.exists(os.path.join(paths["bert_dir"], "tokenizer")): 
            gdown.download(id=file_ids["bert_zip"], output=paths["bert_zip"], quiet=True)
            with zipfile.ZipFile(paths["bert_zip"], 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(paths["bert_zip"])
        if not os.path.exists(paths["slang"]): gdown.download(id=file_ids["slang"], output=paths["slang"], quiet=True)

    models = {
        'Random Forest': joblib.load(paths["rf"]),
        'SVM': joblib.load(paths["svm"]),
        'KNN': joblib.load(paths["knn"])
    }

    tokenizer = BertTokenizer.from_pretrained(os.path.join(paths["bert_dir"], 'tokenizer'))
    bert_model = BertModel.from_pretrained(os.path.join(paths["bert_dir"], 'model'))

    normalisasi_dict = {}
    with open(paths["slang"], "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                f = line.strip().split(":")
                normalisasi_dict[f[0].strip()] = f[1].strip()

    stopword_list = set(stopwords.words('indonesian'))
    penting = ["sangat", "tidak", "kurang", "suka", "bantu", "penting", "benar", "aman", "tertib"]
    for kata in penting:
        stopword_list.discard(kata)
    indo_stopwords = stopword_list

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer

# --- Preprocessing ---
def preprocess_text(text, normalisasi_dict, indo_stopwords, stemmer):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    for slang, baku in normalisasi_dict.items():
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, baku, text, flags=re.IGNORECASE)
    tokens = text.split()
    tokens = [word for word in tokens if word not in indo_stopwords]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# --- BERT Embedding ---
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

# --- Load Semua Resource ---
models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer = load_all_resources()

# --- UI Streamlit ---
st.set_page_config(page_title="Aplikasi Sentimen Demo", page_icon="📢")
st.title("📢 Aplikasi Analisis Sentimen")
st.write("Aplikasi ini menggunakan model dari notebook `demomahasiswa.ipynb`.")

model_choice = st.selectbox("Pilih Model Klasifikasi:", ('Random Forest', 'SVM', 'KNN'))
user_input = st.text_area("Masukkan teks untuk dianalisis:", "aksi demo berjalan dengan tertib dan aman", height=150)

if st.button("Analisis Sentimen", use_container_width=True):
    if user_input:
        cleaned_text = preprocess_text(user_input, normalisasi_dict, indo_stopwords, stemmer)

        if not cleaned_text.strip():
            st.warning("Teks yang Anda masukkan tidak mengandung kata yang dapat dianalisis setelah preprocessing.")
        else:
            with st.spinner(f'Memproses dengan model {model_choice}...'):
                selected_model = models[model_choice]
                bert_features = get_bert_embedding(cleaned_text, tokenizer, bert_model)

                try:
                    sentiment = selected_model.predict(bert_features)[0]
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
                    sentiment = None

                if sentiment is not None:
                    st.subheader(f"Hasil Analisis (Model: {model_choice})")
                    if sentiment == "Positive":
                        st.success(f"**Sentimen: {sentiment}**")
                    elif sentiment == "Neutral":
                        st.info(f"**Sentimen: {sentiment}**")
                    else:
                        st.error(f"**Sentimen: {sentiment}**")

                    if hasattr(selected_model, 'predict_proba'):
                        try:
                            prediction_proba = selected_model.predict_proba(bert_features)
                            positive_class_index = list(selected_model.classes_).index('Positive')
                            positive_proba = prediction_proba[0][positive_class_index]
                            st.progress(positive_proba)
                            st.write(f"Positif: `{positive_proba:.2%}`")
                        except Exception:
                            st.write("Tidak dapat menghitung probabilitas untuk kelas 'Positive'.")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")
