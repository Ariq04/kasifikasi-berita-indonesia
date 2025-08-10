import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

class AppUI:
    """Class ini bertanggung jawab untuk membangun semua elemen UI."""

    def __init__(self, classifier_obj):
        """Menerima objek classifier untuk mendapatkan data seperti daftar kategori."""
        self.classifier = classifier_obj
        # Inisialisasi atribut tab agar bisa diakses nanti
        self.tab1 = None
        self.tab2 = None
        st.set_page_config(
            page_title="Analisis Berita Indonesia",
            page_icon="📰",
            layout="wide"
        )

    def render_main_page(self):
        """Membangun UI utama dengan tab."""
        st.title("Analisis dan Klasifikasi Berita Indonesia")
        # Menetapkan tab ke atribut instance (self.tab1, self.tab2)
        self.tab1, self.tab2 = st.tabs(["🏠 Beranda", "📊 Klasifikasi Berita"])

        with self.tab1:
            self._render_home_tab()
        
        with self.tab2:
            # Mengembalikan referensi ke tab untuk digunakan di app.py
            return self._render_classifier_tab()

    def _render_home_tab(self):
        """Metode private untuk membangun konten tab Beranda."""
        st.header("Selamat Datang!")
        st.markdown("---")
        st.subheader("Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini adalah demo yang menunjukkan bagaimana Machine Learning dapat digunakan untuk mengklasifikasikan teks secara otomatis. 
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

    def _render_classifier_tab(self):
        """Metode private untuk membangun konten tab Klasifikasi."""
        if not self.classifier.get_model():
            st.error("Gagal memuat model. Pastikan file 'model_klasifikasi.joblib' dan 'tfidf_vectorizer.joblib' ada di dalam folder 'models'.")
            return

        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Kategori yang Tersedia")
            model = self.classifier.get_model()
            st.dataframe(pd.DataFrame(model.classes_, columns=["Kategori"]))

        # Mengembalikan referensi ke kolom pertama agar app.py bisa menambahkan elemen di sana
        return col1

    def display_results(self, column, prediction, proba_df, original_input, analyzed_text):
        """Metode untuk menampilkan hasil analisis di dalam kolom yang ditentukan."""
        with column:
            st.markdown("---")
            st.subheader("Hasil Analisis")

            if original_input.strip().startswith('http'):
                st.info(f"**Teks yang Dianalisis dari URL:** *'{analyzed_text}'*")

            confidence_score = np.max(proba_df['Probabilitas'])
            st.metric(
                label="Prediksi Kategori",
                value=prediction.upper(),
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
