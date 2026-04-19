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
import glob # Untuk mencari file CSV secara otomatis

# -------------------------------
# SETUP HALAMAN STREAMLIT
# -------------------------------
st.set_page_config(page_title="Google Stock Predictor Pro", layout="wide")
st.title("📈 Google Stock Forecast Dashboard")
st.markdown("**Status:** Mengunduh model dan data dari Google Drive. Harap tunggu...")

# -------------------------------
# 1. INISIALISASI: UNDUH FILE DARI GOOGLE DRIVE
# -------------------------------
# ID folder Google Drive Anda
FOLDER_ID = "1HQIKbA8tkmhOnOlsve3fRuvcZXcD9vzc"
GDRIVE_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

@st.cache_resource
def download_and_prepare_assets():
    """Mengunduh folder dari Google Drive ke direktori sementara."""
    temp_dir = tempfile.TemporaryDirectory()
    download_path = temp_dir.name

    with st.spinner("Sedang mengunduh file model dan data dari Google Drive..."):
        try:
            gdown.download_folder(url=GDRIVE_URL, output=download_path, quiet=False)
            st.success("✅ Semua file berhasil diunduh!")

            # --- Debug: Tampilkan daftar file yang diunduh ---
            files_found = []
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    files_found.append(os.path.join(root, file))
            
            if files_found:
                with st.expander("📁 Lihat daftar file yang diunduh"):
                    st.write(files_found)
            else:
                st.warning("Tidak ada file yang ditemukan setelah unduhan.")
            # ------------------------------------------------

            return download_path
        except Exception as e:
            st.error(f"Gagal mengunduh file dari Google Drive: {e}")
            st.info("Pastikan folder Google Drive Anda telah dibagikan secara publik (Anyone with the link) dan ID folder benar.")
            return None

SAVE_PATH = download_and_prepare_assets()

def find_csv_file(folder_path):
    """Mencari file CSV pertama dalam folder (rekursif)."""
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    if csv_files:
        return csv_files[0]  # Ambil file CSV pertama yang ditemukan
    return None

# -------------------------------
# 2. FUNGSI PREDIKSI (DENGAN PATH LOKAL)
# -------------------------------
def forecast_stock(category, model_choice, days, local_path):
    try:
        if local_path is None:
            st.error("Path lokal tidak valid. Pastikan unduhan dari Google Drive berhasil.")
            return pd.DataFrame(), None

        # --- Cari file CSV secara otomatis ---
        csv_path = find_csv_file(local_path)
        if csv_path is None:
            st.error(f"Tidak ada file CSV ditemukan di folder unduhan: {local_path}")
            st.info("Silakan periksa kembali isi folder Google Drive Anda. Pastikan ada file .csv di dalamnya.")
            return pd.DataFrame(), None
        
        st.info(f"📄 Menggunakan file CSV: {os.path.basename(csv_path)}")
        # ------------------------------------

        # Muat scaler
        scaler_path = os.path.join(local_path, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            st.error(f"Scaler tidak ditemukan di: {scaler_path}")
            return pd.DataFrame(), None
        sc = joblib.load(scaler_path)

        # Baca CSV
        df_internal = pd.read_csv(csv_path)
        df_internal['Date'] = pd.to_datetime(df_internal['Date'])
        df_internal = df_internal.sort_values('Date')

        # Transformasi data
        scaled_data_internal = sc.transform(df_internal[['Close']])

        window_size = 60
        temp_input = scaled_data_internal[-window_size:].copy()
        pred_output = []
        blend = 0.3

        if category == "Model Dasar":
            model_path = os.path.join(local_path, f'model_{model_choice.lower()}.h5')
            if not os.path.exists(model_path):
                st.error(f"Model tidak ditemukan di: {model_path}")
                return pd.DataFrame(), None
            mdl = load_model(model_path, compile=False)
            mdl.compile(optimizer='adam', loss='mse')

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                y_pred = mdl.predict(x_input, verbose=0)[0][0]
                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        else:  # Model Ditingkatkan
            m_gru = load_model(os.path.join(local_path, 'model_gru.h5'), compile=False)
            m_lstm = load_model(os.path.join(local_path, 'model_lstm.h5'), compile=False)
            m_rnn = load_model(os.path.join(local_path, 'model_rnn.h5'), compile=False)
            for m in [m_gru, m_lstm, m_rnn]:
                m.compile(optimizer='adam', loss='mse')

            if model_choice == 'Stacking':
                meta = joblib.load(os.path.join(local_path, 'meta_model.pkl'))
            elif model_choice == 'Residual':
                resid_mdl = joblib.load(os.path.join(local_path, 'xgb_resid.pkl'))

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                g = m_gru.predict(x_input, verbose=0)[0][0]
                l = m_lstm.predict(x_input, verbose=0)[0][0]
                r = m_rnn.predict(x_input, verbose=0)[0][0]

                if model_choice == 'Ensemble':
                    y_pred = 0.6*g + 0.3*l + 0.1*r
                elif model_choice == 'Stacking':
                    y_pred = meta.predict([[g, l, r]])[0]
                else:  # Residual
                    resid = resid_mdl.predict(x_input.reshape(1, -1))[0]
                    y_pred = g + resid

                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        # Inverse transform
        res_array = np.array(pred_output).reshape(-1, 1)
        inv_res = sc.inverse_transform(res_array).flatten()

        last_date = df_internal['Date'].iloc[-1]
        dates = pd.date_range(last_date + timedelta(days=1), periods=len(inv_res))
        result_df = pd.DataFrame({'Tanggal': dates.strftime('%Y-%m-%d'), 'Harga (USD)': inv_res})

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, inv_res, marker='o', color='blue', label=f'Forecast {model_choice}')
        ax.set_title(f"Prediksi {model_choice} - {days} Hari")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return result_df, fig

    except Exception as e:
        import traceback
        st.error(f"Terjadi kesalahan saat prediksi:\n```\n{traceback.format_exc()}\n```")
        return pd.DataFrame(), None

# -------------------------------
# 3. INTERFACE UTAMA
# -------------------------------
if SAVE_PATH:
    col1, col2, col3 = st.columns(3)
    with col1:
        category_input = st.radio(
            "Kategori Model",
            options=["Model Dasar", "Model Ditingkatkan"],
            index=0,
            horizontal=True
        )
    with col2:
        if category_input == "Model Dasar":
            model_options = ["LSTM", "GRU", "RNN"]
            default_model = "GRU"
        else:
            model_options = ["Ensemble", "Stacking", "Residual"]
            default_model = "Stacking"
        model_input = st.selectbox("Pilih Model", options=model_options, index=model_options.index(default_model))
    with col3:
        days_input = st.slider("Durasi (Hari)", min_value=1, max_value=365, value=30, step=1)

    predict_btn = st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Sedang melakukan prediksi... Mohon tunggu."):
            result_df, fig = forecast_stock(category_input, model_input, days_input, SAVE_PATH)

        if not result_df.empty:
            st.success("Prediksi berhasil!")
            st.subheader("📋 Tabel Prediksi")
            st.dataframe(result_df, use_container_width=True)
            st.subheader("📉 Visualisasi Tren")
            st.pyplot(fig)
        else:
            st.warning("Prediksi gagal. Periksa kembali file yang diperlukan.")
else:
    st.warning("Aplikasi tidak dapat memuat aset dari Google Drive. Silakan periksa kembali koneksi atau ID folder.")
