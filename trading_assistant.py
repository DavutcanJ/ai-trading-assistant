import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import logging
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Anahtarı
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# === 1️⃣ Hisse Verilerini Çekme ===
HISSELER = ["AMZN", "TSLA", "NVDA", "META", "BTC-USD"]

# Transformer model için yapılandırma
class TimeSeriesPredictor(Model):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projections
        self.input_projection = layers.Dense(hidden_size)
        self.input_dropout = layers.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append([
                layers.LayerNormalization(epsilon=1e-6),
                layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=hidden_size // num_heads,
                    dropout=dropout
                ),
                layers.Dropout(dropout),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(hidden_size * 4, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(hidden_size)
            ])
        
        # Output layers - değiştirildi
        self.output_norm = layers.LayerNormalization(epsilon=1e-6)
        self.output_dropout = layers.Dropout(dropout)
        self.pre_output = layers.Dense(hidden_size, activation='relu')
        self.output_layer = layers.Dense(4, activation=None)  # Linear activation for regression
        
    def call(self, inputs, training=False):
        # Input projection
        x = self.input_projection(inputs)
        x = self.input_dropout(x, training=training)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1, dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [tf.shape(x)[0], 1])
        positions = tf.expand_dims(positions, axis=-1)
        
        x = x + positions * 0.01
        
        # Transformer blocks
        for block in self.transformer_blocks:
            norm1, attn, drop1, norm2, ff1, drop2, ff2 = block
            
            # Self-attention with residual connection
            attended = norm1(x)
            attended = attn(attended, attended)
            attended = drop1(attended, training=training)
            x = x + attended
            
            # Feed-forward with residual connection
            ff_out = norm2(x)
            ff_out = ff1(ff_out)
            ff_out = drop2(ff_out, training=training)
            ff_out = ff2(ff_out)
            x = x + ff_out
        
        # Final output processing - değiştirildi
        x = self.output_norm(x)
        x = tf.reduce_mean(x, axis=1)  # Global average pooling
        x = self.output_dropout(x, training=training)
        x = self.pre_output(x)
        return self.output_layer(x)  # Linear output for regression

# Model ve scaler'ları saklamak için global sözlükler
models = {}
scalers = {}

def egitim_verisi_hazirla(df, lookback=24):
    """
    Eğitim verisi hazırlama fonksiyonu
    lookback: Kaç saat öncesine bakılacağı
    """
    features = []
    targets = []
    
    for i in range(lookback, len(df)):
        # Input özellikleri
        window = df.iloc[i-lookback:i]
        feature = np.column_stack((
            window["Close"].values,
            window["Volume"].values,
            window["RSI"].values,
            window["EMA_9"].values,
            window["EMA_21"].values,
            np.zeros(lookback),  # FinBERT score placeholder
            np.zeros(lookback)   # AV score placeholder
        ))
        
        # Hedef değerler (gelecek 4 saatin yüzdelik değişimleri)
        if i + 4 <= len(df):
            target = [
                df["Return_1h"].iloc[i],
                df["Return_2h"].iloc[i],
                df["Return_3h"].iloc[i],
                df["Return_4h"].iloc[i]
            ]
            
            features.append(feature)
            targets.append(target)
    
    return np.array(features), np.array(targets)

def model_egit(hisse, epochs=50, batch_size=32):
    """
    Modeli eğitme fonksiyonu
    """
    try:
        # Veriyi al
        df = hisse_verisi_al(hisse, gun_sayisi=30)  # 30 günlük veri
        if df is None or df.empty:
            logger.error(f"{hisse} için eğitim verisi alınamadı")
            return False
            
        # Eğitim verisi hazırla
        X, y = egitim_verisi_hazirla(df)
        if len(X) == 0 or len(y) == 0:
            logger.error(f"{hisse} için yeterli eğitim verisi yok")
            return False
            
        # Model ve scaler'ı al veya oluştur
        model, scaler = model_yukle_veya_olustur(hisse)
        
        # Scaler'ı fit et
        scaler.fit(X.reshape(-1, X.shape[-1]))
        
        # Verileri normalize et
        X_scaled = np.array([scaler.transform(x) for x in X])
        
        # Modeli eğit
        optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)  # Legacy optimizer kullan
        loss = keras.losses.MeanSquaredError()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        
        # Early stopping ekle
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Learning rate azaltma
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        history = model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Modeli kaydet
        models[hisse] = model
        scalers[hisse] = scaler
        
        logger.info(f"{hisse} modeli başarıyla eğitildi")
        return True
        
    except Exception as e:
        logger.error(f"{hisse} modeli eğitilirken hata: {str(e)}")
        return False

def model_yukle_veya_olustur(hisse):
    if hisse not in models:
        input_size = 7  # Fiyat, Hacim, RSI, EMA9, EMA21, FinBERT score, AV score
        model = TimeSeriesPredictor(input_size)
        # Model'i build et
        dummy_input = tf.random.uniform((1, 24, input_size))  # 24 saatlik pencere
        _ = model(dummy_input)
        models[hisse] = model
        scalers[hisse] = MinMaxScaler()
    return models[hisse], scalers[hisse]

def hisse_verisi_al(ticker, gun_sayisi=7):
    try:
        baslangic = (datetime.now() - timedelta(days=gun_sayisi)).strftime('%Y-%m-%d')
        hisse = yf.Ticker(ticker)
        df = hisse.history(start=baslangic, interval="1h")
        
        if df.empty:
            logger.error(f"{ticker} için veri alınamadı")
            return None
            
        # Teknik indikatörleri hesapla
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
        
        # Yüzdelik değişimleri hesapla
        df["Return_1h"] = df["Close"].pct_change(1)
        df["Return_2h"] = df["Close"].pct_change(2)
        df["Return_3h"] = df["Close"].pct_change(3)
        df["Return_4h"] = df["Close"].pct_change(4)
        
        return df
    except Exception as e:
        logger.error(f"{ticker} için veri alınırken hata: {str(e)}")
        return None

def tum_verileri_al():
    veriler = {}
    for ticker in HISSELER:
        df = hisse_verisi_al(ticker)
        if df is not None:
            veriler[ticker] = df
    return veriler

# === 2️⃣ Finans Haberlerini Çekme ===
def gecmis_haberleri_cek(max_deneme=3):
    for deneme in range(max_deneme):
        try:
            # Alpha Vantage News API'sini kullan
            base_url = "https://www.alphavantage.co/query"
            
            # Tickers parametresini hazırla
            tickers = ",".join([h for h in HISSELER if h != "BTC-USD"])  # BTC hariç
            
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": tickers,
                "apikey": ALPHA_VANTAGE_API_KEY,
                "limit": 10,
                "sort": "LATEST"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "feed" not in data:
                raise ValueError(f"API yanıtında haber bulunamadı: {data}")
                
            # Haberleri ve Alpha Vantage sentiment'lerini çek
            haberler = []
            for article in data["feed"][:10]:
                haberler.append({
                    "title": article["title"],
                    "av_sentiment": article.get("overall_sentiment_label", "neutral"),
                    "av_sentiment_score": float(article.get("overall_sentiment_score", "0.0"))
                })
                
            if not haberler:
                raise ValueError("Haber bulunamadı")
                
            return haberler
            
        except Exception as e:
            logger.error(f"Haber çekme denemesi {deneme + 1} başarısız: {str(e)}")
            if deneme == max_deneme - 1:
                return []
            time.sleep(2)  # Yeniden denemeden önce bekle

# === 3️⃣ FinBERT ile Duygu Analizi ===
try:
    finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone")
except Exception as e:
    logger.error(f"FinBERT modeli yüklenirken hata: {str(e)}")
    finbert = None

def haber_duygu_analizi(haber_dict):
    try:
        if finbert is None:
            return haber_dict["av_sentiment"], haber_dict["av_sentiment_score"]
            
        # FinBERT analizi
        sonuc = finbert(haber_dict["title"])
        finbert_duygu = sonuc[0]['label']
        finbert_skor = sonuc[0]['score']
        
        # Alpha Vantage ve FinBERT sonuçlarını karşılaştır
        if haber_dict["av_sentiment"] == finbert_duygu:
            # Eğer iki model de aynı duyguyu tespit ettiyse, skorların ortalamasını al
            final_duygu = finbert_duygu
            final_skor = (finbert_skor + haber_dict["av_sentiment_score"]) / 2
        else:
            # Farklı duygular tespit edildiyse, daha yüksek skora sahip olanı seç
            if finbert_skor > haber_dict["av_sentiment_score"]:
                final_duygu = finbert_duygu
                final_skor = finbert_skor
            else:
                final_duygu = haber_dict["av_sentiment"]
                final_skor = haber_dict["av_sentiment_score"]
        
        return final_duygu, final_skor
        
    except Exception as e:
        logger.error(f"Duygu analizi hatası: {str(e)}")
        return haber_dict["av_sentiment"], haber_dict["av_sentiment_score"]

# === 4️⃣ Veri Setini Hazırlama ===
def tahmin_yap(model, scaler, veriler):
    try:
        # Verileri normalize et
        normalized = scaler.transform(veriler)
        
        # Tensor'a çevir ve reshape et
        x = tf.convert_to_tensor(normalized, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)  # Batch boyutu ekle
        
        # Tahmin yap
        predictions = model(x, training=False)
        
        # Numpy array'e çevir
        return predictions.numpy()[0]
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        return np.zeros(4)  # Hata durumunda sıfır tahmin döndür

def veriyi_hazirla(hisse, haber_dict, df, duygu, skor):
    try:
        son_veri = df.iloc[-1]
        
        # Model için girdi verilerini hazırla
        model_input = np.array([
            [son_veri["Close"], 
             son_veri["Volume"], 
             son_veri["RSI"],
             son_veri["EMA_9"],
             son_veri["EMA_21"],
             skor,  # FinBERT skoru
             haber_dict["av_sentiment_score"]]  # Alpha Vantage skoru
        ])
        
        # Model ve scaler'ı al
        model, scaler = model_yukle_veya_olustur(hisse)
        
        # Tahmin yap
        tahminler = tahmin_yap(model, scaler, model_input)
        
        return pd.DataFrame({
            "Hisse": [hisse],
            "Haber": [haber_dict["title"]],
            "Duygu": [duygu],
            "Duygu_Skor": [skor],
            "AV_Duygu": [haber_dict["av_sentiment"]],
            "AV_Duygu_Skor": [haber_dict["av_sentiment_score"]],
            "Fiyat": [son_veri["Close"]],
            "Hacim": [son_veri["Volume"]],
            "RSI": [son_veri["RSI"]],
            "EMA_9": [son_veri["EMA_9"]],
            "EMA_21": [son_veri["EMA_21"]],
            "Tahmin_1s": [tahminler[0]],
            "Tahmin_2s": [tahminler[1]],
            "Tahmin_3s": [tahminler[2]],
            "Tahmin_4s": [tahminler[3]]
        })
        
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {str(e)}")
        return None

def model_calistir():
    try:
        logger.info("Model çalıştırılıyor...")
        
        # Verileri al
        tum_veriler = tum_verileri_al()
        if not tum_veriler:
            logger.error("Hiçbir hisse için veri alınamadı")
            return None
            
        # Haberleri çek
        haberler = gecmis_haberleri_cek()
        if not haberler:
            logger.error("Haberler çekilemedi")
            return None
            
        duygular = [haber_duygu_analizi(h) for h in haberler]
        df_list = []
        
        for hisse in tum_veriler:
            df = tum_veriler[hisse]
            if df.empty:
                continue
                
            for haber_dict, (duygu, skor) in zip(haberler, duygular):
                try:
                    fiyat = df["Close"].iloc[-1]
                    hacim = df["Volume"].iloc[-1]
                    rsi = df["RSI"].iloc[-1]
                    ema9 = df["EMA_9"].iloc[-1]
                    ema21 = df["EMA_21"].iloc[-1]
                    
                    df_list.append(veriyi_hazirla(hisse, haber_dict, df, duygu, skor))
                except Exception as e:
                    logger.error(f"{hisse} için veri hazırlanırken hata: {str(e)}")
                    continue
        
        if not df_list:
            logger.error("İşlenecek veri bulunamadı")
            return None
            
        return pd.concat(df_list, ignore_index=True)
    except Exception as e:
        logger.error(f"Model çalıştırılırken hata: {str(e)}")
        return None

# === 5️⃣ Long/Short Tavsiyesi ===
def long_short_tavsiye(df):
    if df is None:
        return None
    try:
        # Trend analizi (0 ile 1 arasında)
        df["Trend_Score"] = (df["EMA_9"] > df["EMA_21"]).astype(float)
        
        # Sentiment analizi (0 ile 1 arasında)
        df["Sentiment_Score"] = ((df["Duygu"] == "positive").astype(float) * df["Duygu_Skor"] + 
                               (df["AV_Duygu"] == "positive").astype(float) * df["AV_Duygu_Skor"]) / 2
        
        # Tahmin analizi (tahminlerin ağırlıklı ortalaması)
        weights = [0.4, 0.3, 0.2, 0.1]  # Yakın saatlere daha fazla ağırlık
        df["Tahmin_Score"] = (df["Tahmin_1s"] * weights[0] + 
                             df["Tahmin_2s"] * weights[1] + 
                             df["Tahmin_3s"] * weights[2] + 
                             df["Tahmin_4s"] * weights[3])
        
        # Final skor hesaplama (her faktörün ağırlıklı ortalaması)
        df["Final_Score"] = (df["Trend_Score"] * 0.3 +  # Trend %30
                           df["Sentiment_Score"] * 0.3 +  # Sentiment %30
                           (df["Tahmin_Score"] > 0).astype(float) * 0.4)  # Tahmin %40
        
        # Tavsiye oluşturma
        df["Tavsiye"] = df["Final_Score"].apply(
            lambda x: "LONG" if x >= 0.6 else "SHORT" if x <= 0.4 else "NOTR"
        )
        
        return df
    except Exception as e:
        logger.error(f"Tavsiye oluşturulurken hata: {str(e)}")
        return None

# === 6️⃣ Modeli Çalıştırma ===
def calistir(egitim_modu=False):
    try:
        logger.info("Analiz başlatılıyor...")
        
        if egitim_modu:
            logger.info("Model eğitimi başlatılıyor...")
            for hisse in HISSELER:
                model_egit(hisse)
        
        df = model_calistir()
        if df is not None:
            df = long_short_tavsiye(df)
            if df is not None:
                print("\nSonuçlar:")
                print(df[[
                    "Hisse", "Haber", "Duygu", "Duygu_Skor", 
                    "AV_Duygu", "AV_Duygu_Skor",
                    "Tahmin_1s", "Tahmin_2s", "Tahmin_3s", "Tahmin_4s",
                    "Tavsiye"
                ]])
                df.to_csv("sonuclar.csv", index=False)
                logger.info("Sonuçlar sonuclar.csv dosyasına kaydedildi.")
            else:
                logger.error("Tavsiye oluşturulamadı")
        else:
            logger.error("Model çalıştırılamadı")
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata: {str(e)}")

# === 7️⃣ Otomatik Güncelleme ===
def otomatik_haber_guncelleme(guncelleme_suresi=3600):  # 1 saat
    logger.info("Otomatik güncelleme başlatılıyor...")
    while True:
        try:
            calistir()
            time.sleep(guncelleme_suresi)
        except KeyboardInterrupt:
            logger.info("Program kullanıcı tarafından durduruldu")
            break
        except Exception as e:
            logger.error(f"Otomatik güncelleme hatası: {str(e)}")
            time.sleep(60)  # Hata durumunda 1 dakika bekle ve tekrar dene

if __name__ == "__main__":
    try:
        # İlk çalıştırmada modelleri eğit
        calistir(egitim_modu=True)
        
        # Sonra otomatik güncellemeye geç
        otomatik_haber_guncelleme()
    except Exception as e:
        logger.error(f"Program çalıştırılırken kritik hata: {str(e)}")
