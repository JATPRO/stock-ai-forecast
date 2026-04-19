import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta
import tempfile
import gdown
import glob

# -------------------------------
# KONFIGURASI HALAMAN
# -------------------------------
st.set_page_config(page_title="Google Stock Predictor Pro", layout="wide")
st.title("📈 Google Stock Forecast Dashboard")
st.markdown("Mengunduh model dan data dari Google Drive...")

# -------------------------------
# FUNGSI DOWNLOAD & CACHE (DIPERBAIKI)
# -------------------------------
FOLDER_ID = "1HQIKbA8tkmhOnOlsve3fRuvcZXcD9vzc"
GDRIVE_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

# Simpan referensi direktori sementara di session_state agar tidak dihapus
if 'temp_dir_path' not in st.session_state:
    st.session_state.temp_dir_path = None

@st.cache_resource
def download_assets():
    """Download folder dari Google Drive ke direktori sementara dan kembalikan path-nya."""
    # Buat direktori sementara manual (tidak otomatis dihapus oleh objek)
    temp_path = tempfile.mkdtemp()
    st.session_state.temp_dir_path = temp_path  # simpan agar tetap ada

    with st.spinner("⏳ Mengunduh file dari Google Drive..."):
        try:
            gdown.download_folder(url=GDRIVE_URL, output=temp_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh: {e}")
            return None

    # Tampilkan daftar file yang diunduh (opsional untuk debugging)
    files_list = []
    for root, _, files in os.walk(temp_path):
        for f in files:
            files_list.append(os.path.join(root, f))
    with st.expander("📁 File yang diunduh", expanded=False):
        st.write(files_list)

    return temp_path

SAVE_PATH = download_assets()

# -------------------------------
# FUNGSI PENCARIAN FILE CSV
# -------------------------------
def find_csv_file(folder):
    """Mencari file GOOG.csv terlebih dahulu, lalu fallback ke file .csv pertama."""
    # Verifikasi folder ada
    if not os.path.isdir(folder):
        st.error(f"Folder tidak ditemukan: {folder}")
        return None

    # Cek langsung GOOG.csv
    direct_csv = os.path.join(folder, "GOOG.csv")
    if os.path.isfile(direct_csv):
        return direct_csv

    # Fallback: cari file .csv apa saja secara rekursif
    candidates = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if candidates:
        return candidates[0]

    return None

# -------------------------------
# FUNGSI PREDIKSI UTAMA
# -------------------------------
def forecast_stock(category, model_choice, days, local_path):
    if local_path is None:
        st.error("Folder aset tidak ditemukan. Pastikan unduhan berhasil.")
        return pd.DataFrame(), None

    # 1. Cari CSV
    csv_file = find_csv_file(local_path)
    if csv_file is None:
        st.error("Tidak ditemukan file CSV di folder unduhan.")
        return pd.DataFrame(), None
    st.info(f"📄 Menggunakan CSV: {os.path.basename(csv_file)}")

    # 2. Load Scaler
    scaler_path = os.path.join(local_path, "scaler.pkl")
    if not os.path.exists(scaler_path):
        st.error("scaler.pkl tidak ditemukan.")
        return pd.DataFrame(), None
    sc = joblib.load(scaler_path)

    # 3. Baca dan siapkan data
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    scaled_data = sc.transform(df[['Close']])

    window_size = 60
    temp_input = scaled_data[-window_size:].copy()
    pred_output = []
    blend = 0.3

    try:
        if category == "Model Dasar":
            model_path = os.path.join(local_path, f"model_{model_choice.lower()}.h5")
            if not os.path.exists(model_path):
                st.error(f"File model tidak ditemukan: {model_path}")
                return pd.DataFrame(), None
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='mse')

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                y_pred = model.predict(x_input, verbose=0)[0][0]
                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        else:  # Model Ditingkatkan
            m_gru = load_model(os.path.join(local_path, "model_gru.h5"), compile=False)
            m_lstm = load_model(os.path.join(local_path, "model_lstm.h5"), compile=False)
            m_rnn = load_model(os.path.join(local_path, "model_rnn.h5"), compile=False)
            for m in [m_gru, m_lstm, m_rnn]:
                m.compile(optimizer='adam', loss='mse')

            if model_choice == "Stacking":
                meta_path = os.path.join(local_path, "meta_model.pkl")
                if not os.path.exists(meta_path):
                    st.error("meta_model.pkl tidak ditemukan.")
                    return pd.DataFrame(), None
                meta = joblib.load(meta_path)
            elif model_choice == "Residual":
                resid_path = os.path.join(local_path, "xgb_resid.pkl")
                if not os.path.exists(resid_path):
                    st.error("xgb_resid.pkl tidak ditemukan.")
                    return pd.DataFrame(), None
                resid_mdl = joblib.load(resid_path)

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                g = m_gru.predict(x_input, verbose=0)[0][0]
                l = m_lstm.predict(x_input, verbose=0)[0][0]
                r = m_rnn.predict(x_input, verbose=0)[0][0]

                if model_choice == "Ensemble":
                    y_pred = 0.6*g + 0.3*l + 0.1*r
                elif model_choice == "Stacking":
                    y_pred = meta.predict([[g, l, r]])[0]
                else:  # Residual
                    resid = resid_mdl.predict(x_input.reshape(1, -1))[0]
                    y_pred = g + resid

                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        # Inverse scaling
        pred_array = np.array(pred_output).reshape(-1, 1)
        inv_pred = sc.inverse_transform(pred_array).flatten()

        last_date = df['Date'].iloc[-1]
        date_range = pd.date_range(last_date + timedelta(days=1), periods=len(inv_pred))
        result_df = pd.DataFrame({
            'Tanggal': date_range.strftime('%Y-%m-%d'),
            'Harga (USD)': inv_pred.round(2)
        })

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_range, inv_pred, marker='o', color='#1f77b4', linewidth=2)
        ax.set_title(f"Prediksi Harga Saham Google - {model_choice} ({days} Hari)", fontsize=14)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga Penutupan (USD)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return result_df, fig

    except Exception as e:
        import traceback
        st.error(f"Error saat prediksi:\n```\n{traceback.format_exc()}\n```")
        return pd.DataFrame(), None

# -------------------------------
# ANTARMUKA STREAMLIT
# -------------------------------
if SAVE_PATH is None:
    st.error("❌ Gagal mengunduh aset dari Google Drive. Pastikan folder dibagikan secara publik.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    category = st.radio("Kategori Model", ["Model Dasar", "Model Ditingkatkan"], horizontal=True)
with col2:
    if category == "Model Dasar":
        model_choice = st.selectbox("Pilih Model", ["LSTM", "GRU", "RNN"])
    else:
        model_choice = st.selectbox("Pilih Model", ["Ensemble", "Stacking", "Residual"])
with col3:
    days = st.slider("Jangka Waktu Prediksi (hari)", 1, 365, 30)

predict_btn = st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("🧠 Model sedang memprediksi..."):
        df_result, plot_fig = forecast_stock(category, model_choice, days, SAVE_PATH)

    if not df_result.empty:
        st.success("✅ Prediksi selesai!")
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df_result, use_container_width=True)

        st.subheader("📈 Visualisasi")
        st.pyplot(plot_fig)
    else:
        st.warning("Prediksi tidak dapat dijalankan. Periksa kembali file dan model.")
