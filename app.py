import streamlit as st
from classifier import NewsClassifier, ArticleScraper
from ui import AppUI

# --- Main Application Logic ---
def main():
    """Fungsi utama untuk menjalankan aplikasi."""
    
    # 1. Buat Objek dari Class
    classifier = NewsClassifier(
        model_path='models/model_klasifikasi.joblib',
        vectorizer_path='models/tfidf_vectorizer.joblib'
    )
    
    app_ui = AppUI(classifier)

    # 2. Render halaman utama dan dapatkan kolom untuk menempatkan input
    # Metode render_main_page() kini mengembalikan kolom tempat input akan berada
    classifier_column = app_ui.render_main_page()
    
    # Hanya lanjutkan jika kolom berhasil dibuat (artinya model berhasil dimuat)
    if classifier_column:
        # Tempatkan elemen input dan tombol di dalam kolom yang sudah ditentukan oleh UI
        with classifier_column:
            user_input = st.text_area('Masukkan Judul Berita atau tempel URL di sini:', '', height=150, placeholder="Contoh: https://finance.detik.com/...")

            if st.button('Klasifikasikan', type="primary", use_container_width=True):
                if user_input:
                    analysis_text = user_input
                    
                    # Cek apakah input adalah URL
                    if user_input.strip().startswith('http'):
                        with st.spinner('Mengambil artikel dari URL...'):
                            analysis_text, error = ArticleScraper.get_title_from_url(user_input)
                        if error:
                            st.error(error)
                            analysis_text = None
                    
                    if analysis_text:
                        # Lakukan prediksi menggunakan objek classifier
                        prediction, proba_df, error = classifier.predict(analysis_text)
                        if error:
                            st.error(error)
                        else:
                            # Panggil display_results dengan argumen yang benar, termasuk 'classifier_column'
                            app_ui.display_results(classifier_column, prediction, proba_df, user_input, analysis_text)
                else:
                    st.warning('⚠️ Mohon masukkan teks atau URL terlebih dahulu.')

if __name__ == "__main__":
    main()
