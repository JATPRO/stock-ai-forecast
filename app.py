import joblib
import numpy as np
import pandas as pd
import gradio as gr
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta

# --- 1. SETUP PATH ---
save_path = '/content/drive/MyDrive/riset akhir/'
csv_path = '/content/drive/MyDrive/riset akhir/GOOG.csv'

def forecast_stock(category, model_choice, days):
    try:
        print(f"DEBUG: Memulai prediksi untuk {model_choice} ({category}) selama {days} hari...")

        # 1. Validasi Path
        if not os.path.exists(csv_path):
            return pd.DataFrame(), None
        if not os.path.exists(os.path.join(save_path, 'scaler.pkl')):
            return pd.DataFrame(), None

        # 2. Ambil Scaler dari Drive
        sc = joblib.load(os.path.join(save_path, 'scaler.pkl'))

        # 3. Ambil CSV dari Drive
        df_internal = pd.read_csv(csv_path)
        df_internal['Date'] = pd.to_datetime(df_internal['Date'])
        df_internal = df_internal.sort_values('Date')

        # 4. Transformasi data untuk input model
        scaled_data_internal = sc.transform(df_internal[['Close']])

        window_size = 60
        temp_input = scaled_data_internal[-window_size:].copy()
        pred_output = []
        blend = 0.3

        # 5. Load Model & Prediksi
        if category == "Model Dasar":
            mdl_path = os.path.join(save_path, f'model_{model_choice.lower()}.h5')
            print(f"DEBUG: Memuat model dari {mdl_path}")
            # Gunakan compile=False untuk menghindari ValueError deserialization
            mdl = load_model(mdl_path, compile=False)
            mdl.compile(optimizer='adam', loss='mse')

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                y_pred = mdl.predict(x_input, verbose=0)[0][0]
                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))
        else:
            print("DEBUG: Memuat model ditingkatkan (Ensemble/Stacking/Residual)")
            m_gru = load_model(os.path.join(save_path, 'model_gru.h5'), compile=False)
            m_lstm = load_model(os.path.join(save_path, 'model_lstm.h5'), compile=False)
            m_rnn = load_model(os.path.join(save_path, 'model_rnn.h5'), compile=False)

            for m in [m_gru, m_lstm, m_rnn]: m.compile(optimizer='adam', loss='mse')

            for _ in range(int(days)):
                x_input = temp_input[-window_size:].reshape(1, window_size, 1)
                g = m_gru.predict(x_input, verbose=0)[0][0]
                l = m_lstm.predict(x_input, verbose=0)[0][0]
                r = m_rnn.predict(x_input, verbose=0)[0][0]

                if model_choice == 'Ensemble':
                    y_pred = 0.6*g + 0.3*l + 0.1*r
                elif model_choice == 'Stacking':
                    meta = joblib.load(os.path.join(save_path, 'meta_model.pkl'))
                    y_pred = meta.predict([[g, l, r]])[0]
                else: # Residual
                    resid_mdl = joblib.load(os.path.join(save_path, 'xgb_resid.pkl'))
                    resid = resid_mdl.predict(x_input.reshape(1, -1))[0]
                    y_pred = g + resid

                stabilized = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stabilized)
                temp_input = np.vstack((temp_input[1:], [[stabilized]]))

        # 6. Inverse Transform ke USD
        res_array = np.array(pred_output).reshape(-1, 1)
        inv_res = sc.inverse_transform(res_array).flatten()

        # 7. Formatting output
        last_date = df_internal['Date'].iloc[-1]
        dates = pd.date_range(last_date + timedelta(days=1), periods=len(inv_res))
        result_df = pd.DataFrame({'Tanggal': dates.strftime('%Y-%m-%d'), 'Harga (USD)': inv_res})

        fig = plt.figure(figsize=(10, 4))
        plt.plot(dates, inv_res, marker='o', color='blue', label=f'Forecast {model_choice}')
        plt.title(f"Prediksi {model_choice} - {days} Hari")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        print("DEBUG: Prediksi selesai dengan sukses.")
        return result_df, fig

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"DEBUG ERROR:\n{err_msg}")
        # Pastikan output kedua adalah None jika terjadi error agar Plot tidak crash
        return pd.DataFrame(), None

# --- 2. UI GRADIO ---
with gr.Blocks(title="Google Stock Predictor Pro") as demo:
    gr.Markdown("# 📈 Google Stock Forecast Dashboard")
    gr.Markdown("**Status:** Aset dimuat dari Drive. Cek log Colab jika terjadi error.")

    with gr.Row():
        category_input = gr.Radio(choices=["Model Dasar", "Model Ditingkatkan"], value="Model Dasar", label="Kategori Model")
        model_input = gr.Dropdown(choices=["LSTM", "GRU", "RNN"], value="GRU", label="Pilih Model")
        days_input = gr.Slider(minimum=1, maximum=365, value=30, step=1, label="Durasi (Hari)")

    def update_models(cat):
        if cat == "Model Dasar":
            return gr.update(choices=["LSTM", "GRU", "RNN"], value="GRU")
        else:
            return gr.update(choices=["Ensemble", "Stacking", "Residual"], value="Stacking")

    category_input.change(fn=update_models, inputs=category_input, outputs=model_input)

    btn = gr.Button("🚀 Mulai Prediksi", variant="primary")

    with gr.Row():
        out_df = gr.Dataframe(label="Tabel Prediksi")
        out_plot = gr.Plot(label="Visualisasi Tren")

    btn.click(fn=forecast_stock, inputs=[category_input, model_input, days_input], outputs=[out_df, out_plot])

demo.launch(share=True, debug=True)
