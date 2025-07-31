import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from newspaper import Article 

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Berita Indonesia",
    page_icon="📰",
    layout="wide"
)

# --- Fungsi-Fungsi ---
@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model dan vectorizer yang sudah dilatih."""
    try:
        model = joblib.load('models/model_klasifikasi.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_data
def scrape_article_title(url):
    """Mengambil judul artikel dari URL menggunakan newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        # Mengembalikan judul jika ada, jika tidak, kembalikan teks utama
        return article.title if article.title else article.text, None
    except Exception as e:
        return None, f"Gagal mengambil artikel. Error: {e}"

# Muat model dan vectorizer
model, vectorizer = load_model_and_vectorizer()

# --- UI Utama ---
st.title("Analisis dan Klasifikasi Berita Indonesia")

tab1, tab2 = st.tabs(["🏠 Beranda", "📊 Klasifikasi Berita"])

# --- Konten untuk Tab Beranda ---
with tab1:
    st.header("Selamat Datang!")
    st.markdown("---")
    st.subheader("Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini adalah demo yang menunjukkan bagaimana *Machine Learning* dapat digunakan untuk mengklasifikasikan teks secara otomatis. 
    Dengan menggunakan model yang telah dilatih pada ribuan judul berita, aplikasi ini dapat memprediksi kategori sebuah berita hanya dari judulnya.
    """)

    st.subheader("Panduan Penggunaan")
    st.markdown("""
    1.  **Pilih Tab 'Klasifikasi Berita'**: Klik pada tab di bagian atas untuk memulai.
    2.  **Masukkan Judul atau URL**: Di halaman klasifikasi, Anda bisa mengetik judul berita secara manual atau menempelkan link (URL) dari berita online.
    3.  **Klik Tombol Klasifikasi**: Tekan tombol untuk memulai proses analisis.
    4.  **Lihat Hasil**: Aplikasi akan menampilkan kategori yang paling sesuai untuk berita tersebut, beserta grafik yang menunjukkan tingkat kepercayaan model.
    """)
    st.info("Silakan pilih tab **Klasifikasi Berita** di atas untuk memulai.", icon="👆")

# --- Konten untuk Tab Klasifikasi ---
with tab2:
    if model and vectorizer:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Area Klasifikasi")
            user_input = st.text_area(
                'Masukkan Judul Berita atau tempel URL di sini:', 
                '', 
                height=150, 
                placeholder="Contoh: https://finance.detik.com/..."
            )
            
            if st.button('Klasifikasikan', type="primary", use_container_width=True):
                if user_input:
                    analysis_text = user_input
                    # Cek apakah input adalah URL
                    if user_input.strip().startswith('http'):
                        with st.spinner('Mengambil artikel dari URL...'):
                            analysis_text, error = scrape_article_title(user_input)
                        if error:
                            st.error(error)
                            analysis_text = None # Hentikan proses jika ada error
                    
                    if analysis_text:
                        # Lanjutkan proses klasifikasi
                        vectorized_input = vectorizer.transform([analysis_text])
                        prediction = model.predict(vectorized_input)
                        prediction_proba = model.predict_proba(vectorized_input)
                        confidence_score = np.max(prediction_proba)
                        
                        proba_df = pd.DataFrame(
                            prediction_proba.flatten(),
                            index=model.classes_,
                            columns=['Probabilitas']
                        ).sort_values(by='Probabilitas', ascending=False)
                        
                        st.markdown("---")
                        st.subheader("Hasil Analisis")
                        
                        if user_input.strip().startswith('http'):
                            st.info(f"**Teks yang Dianalisis dari URL:** *'{analysis_text}'*")

                        st.metric(
                            label="Prediksi Kategori",
                            value=prediction[0].upper(),
                            help=f"Model memprediksi kategori ini dengan tingkat kepercayaan {confidence_score:.2%}"
                        )
                        
                        with st.expander("Lihat Detail Tingkat Kepercayaan (Grafik)"):
                            fig = px.bar(
                                proba_df, x=proba_df.index, y='Probabilitas', color=proba_df.index,
                                labels={'index': 'Kategori', 'Probabilitas': 'Tingkat Kepercayaan'},
                                text_auto='.2%', title="Detail Tingkat Kepercayaan per Kategori"
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning('⚠️ Mohon masukkan teks atau URL terlebih dahulu.')
        
        with col2:
            st.subheader("Kategori yang Tersedia")
            st.dataframe(pd.DataFrame(model.classes_, columns=["Kategori"]))

    else:
        st.error("Gagal memuat model. Pastikan file 'model_klasifikasi.joblib' dan 'tfidf_vectorizer.joblib' ada di dalam folder 'models'.")
