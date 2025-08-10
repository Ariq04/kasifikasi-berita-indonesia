import joblib
import pandas as pd
import numpy as np
from newspaper import Article, Config

class NewsClassifier:
    """
    Class ini mengenkapsulasi semua fungsionalitas terkait model.
    Ini adalah contoh Encapsulation dan Abstraction.
    """
    def __init__(self, model_path, vectorizer_path):
        """Constructor untuk memuat model dan vectorizer saat objek dibuat."""
        self._model, self._vectorizer = self._load_model(model_path, vectorizer_path)

    @staticmethod
    def _load_model(model_path, vectorizer_path):
        """Metode private untuk memuat file. Detail implementasi disembunyikan."""
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            return model, vectorizer
        except FileNotFoundError:
            return None, None
    
    def get_model(self):
        """Metode publik untuk mengakses model jika diperlukan."""
        return self._model

    def predict(self, text):
        """
        Metode publik untuk melakukan prediksi. 
        Pengguna tidak perlu tahu bagaimana teks diubah menjadi vektor.
        """
        if not self._model or not self._vectorizer:
            return None, None, "Model atau vectorizer tidak berhasil dimuat."

        vectorized_text = self._vectorizer.transform([text])
        prediction = self._model.predict(vectorized_text)
        probabilities = self._model.predict_proba(vectorized_text)
        
        proba_df = pd.DataFrame(
            probabilities.flatten(),
            index=self._model.classes_,
            columns=['Probabilitas']
        ).sort_values(by='Probabilitas', ascending=False)
        
        return prediction[0], proba_df, None

class ArticleScraper:
    """Class terpisah yang bertanggung jawab hanya untuk scraping."""
    
    @staticmethod
    def get_title_from_url(url):
        """Mengambil judul artikel dari URL."""
        try:
            config = Config()
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            return article.title if article.title else article.text, None
        except Exception as e:
            return None, f"Gagal mengambil artikel. Error: {e}"

