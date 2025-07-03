# =============================================================================
# APLIKASI STREAMLIT UNTUK PREDIKSI PENYAKIT JANTUNG (VERSI KOREKSI)
# =============================================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Konfigurasi Halaman dan Judul ---
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi untuk Memuat Model ---
@st.cache_data
def load_model():
    """Memuat model dari file .pkl"""
    try:
        with open('heart_disease_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("File model 'heart_disease_model.pkl' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
        return None

# Memuat model
model = load_model()

# --- Tampilan Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Data Pasien 👨‍⚕️")
st.sidebar.markdown("Silakan isi semua parameter medis di bawah ini untuk mendapatkan prediksi.")

def user_input_features():
    """Membuat sidebar untuk input dari pengguna."""
    
    with st.sidebar.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Umur (Tahun)', min_value=1, max_value=120, value=50, step=1)
            sex = st.selectbox('Jenis Kelamin', ('Pria', 'Wanita'))
            cp = st.selectbox('Jenis Nyeri Dada (Chest Pain)', (0, 1, 2, 3), help="0: Tipikal Angina, 1: Atipikal Angina, 2: Nyeri Non-angina, 3: Asimtomatik")
            # NAMA KOLOM DIPERBAIKI: trtbps -> trestbps
            trestbps = st.number_input('Tekanan Darah Istirahat (mm Hg)', min_value=50, max_value=250, value=120)
            chol = st.number_input('Kolesterol Serum (mg/dl)', min_value=100, max_value=600, value=200)
            fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', ('Tidak', 'Ya'))
        
        with col2:
            restecg = st.selectbox('Hasil EKG Istirahat', (0, 1, 2), help="0: Normal, 1: Kelainan Gelombang ST-T, 2: Hipertrofi Ventrikel Kiri")
            thalach = st.number_input('Detak Jantung Maksimum', min_value=60, max_value=220, value=150)
            # NAMA KOLOM DIPERBAIKI: exng -> exang
            exang = st.selectbox('Angina Akibat Olahraga', ('Tidak', 'Ya'))
            oldpeak = st.number_input('Oldpeak (Depresi ST)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            # NAMA KOLOM DIPERBAIKI: slp -> slope
            slope = st.selectbox('Slope dari Puncak Latihan ST', (0, 1, 2))
            # NAMA KOLOM DIPERBAIKI: caa -> ca
            ca = st.selectbox('Jumlah Pembuluh Darah Utama (ca)', (0, 1, 2, 3, 4))
            # NAMA KOLOM DIPERBAIKI: thall -> thal
            thal = st.selectbox('Hasil Tes Thallium Stress', (0, 1, 2, 3), help="0: Null, 1: Cacat Tetap, 2: Normal, 3: Cacat Reversibel")

        submitted = st.form_submit_button("Prediksi Sekarang")

    sex_val = 1 if sex == 'Pria' else 0
    fbs_val = 1 if fbs == 'Ya' else 0
    exang_val = 1 if exang == 'Ya' else 0

    # --- BAGIAN PALING PENTING: NAMA KOLOM DI SINI HARUS SAMA PERSIS DENGAN SAAT TRAINING ---
    data = {
        'age': age,
        'sex': sex_val,
        'cp': cp,
        'trestbps': trestbps, # Diperbaiki
        'chol': chol,
        'fbs': fbs_val,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang_val,   # Diperbaiki
        'oldpeak': oldpeak,
        'slope': slope,       # Diperbaiki
        'ca': ca,             # Diperbaiki
        'thal': thal          # Diperbaiki
    }
    features = pd.DataFrame(data, index=[0])
    return features, submitted

input_df, submitted = user_input_features()

# --- Tampilan Utama ---
st.title("Aplikasi Prediksi Penyakit Jantung")
st.markdown("Aplikasi ini dibuat untuk memprediksi risiko penyakit jantung menggunakan model *Machine Learning*. Harap diingat bahwa hasil prediksi ini **bukanlah diagnosis medis**.")
st.markdown("---")

st.subheader("Data Pasien yang Anda Masukkan:")
st.write(input_df)

if submitted and model is not None:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Hasil Prediksi Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediksi Kondisi",
                  value="Berisiko" if prediction[0] == 1 else "Aman",
                  delta="Perlu Perhatian Medis" if prediction[0] == 1 else "Kondisi Baik")
    
    with col2:
        st.metric(label="Tingkat Keyakinan Model",
                  value=f"{prediction_proba[0][prediction[0]]*100:.2f}%")

    if prediction[0] == 1:
        st.error("**Peringatan:** Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien memiliki **RISIKO TINGGI** terkena penyakit jantung.")
    else:
        st.success("**Informasi:** Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien memiliki **RISIKO RENDAH** (aman) dari penyakit jantung.")


