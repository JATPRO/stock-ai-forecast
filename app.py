import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta

# -------------------------------
# SETUP HALAMAN STREAMLIT
# -------------------------------
st.set_page_config(page_title="Google Stock Predictor Pro", layout="wide")
st.title("📈 Google Stock Forecast Dashboard")
st.markdown("**Status:** Aset dimuat dari path lokal. Pastikan file model, scaler, dan CSV tersedia.")

# -------------------------------
# SIDEBAR: INPUT PATH
# -------------------------------
st.sidebar.header("Konfigurasi Path")
save_path = st.sidebar.text_input(
    "Folder penyimpanan model & scaler",
    value="/content/drive/MyDrive/riset akhir/",  # default dari Colab
    help="Path ke folder yang berisi model_*.h5, scaler.pkl, meta_model.pkl, xgb_resid.pkl"
)
csv_path = st.sidebar.text_input(
    "Path file CSV data saham",
    value="/content/drive/MyDrive/riset akhir/GOOG.csv",
    help="File CSV harus memiliki kolom 'Date' dan 'Close'"
)

st.sidebar.markdown("---")
st.sidebar.info("Pastikan folder dan file tersebut dapat diakses oleh aplikasi.")

# -------------------------------
# FUNGSI PREDIKSI (SAMA SEPERTI SEBELUMNYA)
# -------------------------------
def forecast_stock(category, model_choice, days):
    try:
        # Validasi path
        if not os.path.exists(csv_path):
            st.error(f"File CSV tidak ditemukan: {csv_path}")
            return pd.DataFrame(), None
        scaler_file = os.path.join(save_path, 'scaler.pkl')
        if not os.path.exists(scaler_file):
            st.error(f"Scaler tidak ditemukan: {scaler_file}")
            return pd.DataFrame(), None

        # Muat scaler
        sc = joblib.load(scaler_file)

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
            mdl_path = os.path.join(save_path, f'model_{model_choice.lower()}.h5')
            if not os.path.exists(mdl_path):
                st.error(f"Model file tidak ditemukan: {mdl_path}")
                return pd.DataFrame(), None
            mdl = load_model(mdl_path, compile=False)
            mdl.compile(optimizer='adam', loss='mse')

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                y_pred = mdl.predict(x_input, verbose=0)[0][0]
                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        else:  # Model Ditingkatkan
            # Muat ketiga model dasar
            m_gru = load_model(os.path.join(save_path, 'model_gru.h5'), compile=False)
            m_lstm = load_model(os.path.join(save_path, 'model_lstm.h5'), compile=False)
            m_rnn = load_model(os.path.join(save_path, 'model_rnn.h5'), compile=False)
            for m in [m_gru, m_lstm, m_rnn]:
                m.compile(optimizer='adam', loss='mse')

            # Muat model meta / residual jika diperlukan
            if model_choice == 'Stacking':
                meta_path = os.path.join(save_path, 'meta_model.pkl')
                if not os.path.exists(meta_path):
                    st.error(f"Meta model tidak ditemukan: {meta_path}")
                    return pd.DataFrame(), None
                meta = joblib.load(meta_path)
            elif model_choice == 'Residual':
                resid_path = os.path.join(save_path, 'xgb_resid.pkl')
                if not os.path.exists(resid_path):
                    st.error(f"XGBoost residual model tidak ditemukan: {resid_path}")
                    return pd.DataFrame(), None
                resid_mdl = joblib.load(resid_path)

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

        # Format hasil
        last_date = df_internal['Date'].iloc[-1]
        dates = pd.date_range(last_date + timedelta(days=1), periods=len(inv_res))
        result_df = pd.DataFrame({'Tanggal': dates.strftime('%Y-%m-%d'), 'Harga (USD)': inv_res})

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, inv_res, marker='o', color='blue', label=f'Forecast {model_choice}')
        ax.set_title(f"Prediksi {model_choice} - {days} Hari")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return result_df, fig

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi:\n```\n{traceback.format_exc()}\n```")
        return pd.DataFrame(), None

# -------------------------------
# INTERFACE UTAMA
# -------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    category_input = st.radio(
        "Kategori Model",
        options=["Model Dasar", "Model Ditingkatkan"],
        index=0,
        horizontal=True
    )
with col2:
    # Dropdown dinamis berdasarkan kategori
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

# Tempat hasil
if predict_btn:
    with st.spinner("Sedang melakukan prediksi... Mohon tunggu."):
        result_df, fig = forecast_stock(category_input, model_input, days_input)

    if not result_df.empty:
        st.success("Prediksi berhasil!")
        st.subheader("📋 Tabel Prediksi")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("📉 Visualisasi Tren")
        st.pyplot(fig)
    else:
        st.warning("Prediksi gagal. Periksa kembali path atau file yang diperlukan.")
