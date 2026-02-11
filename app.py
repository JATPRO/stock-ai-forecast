# ============================================
# 🔒 GLOBAL LOCK
# ============================================
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# IMPORT
# ============================================
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide")
st.title("🔒 AI Stock Forecast System")

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model("models/lstm.h5")
    gru = tf.keras.models.load_model("models/gru.h5")
    rnn = tf.keras.models.load_model("models/rnn.h5")
    xgb_stack = joblib.load("models/xgb_stacking.pkl")
    xgb_res = joblib.load("models/xgb_residual.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return lstm, gru, rnn, xgb_stack, xgb_res, scaler

lstm, gru, rnn, xgb_stack, xgb_res, scaler = load_models()

# ============================================
# UPLOAD DATA
# ============================================
st.sidebar.header("📂 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, Close)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("GOOG.csv")

if "Close" not in df.columns:
    st.error("CSV must contain Close column")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ============================================
# PREPARE DATA
# ============================================
scaled = scaler.transform(df[["Close"]])
window = 60

X, y = [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

# ============================================
# PREDICTIONS
# ============================================
y_lstm = lstm.predict(X_test, verbose=0).flatten()
y_gru = gru.predict(X_test, verbose=0).flatten()
y_rnn = rnn.predict(X_test, verbose=0).flatten()

y_ens = 0.4*y_lstm + 0.4*y_gru + 0.2*y_rnn

stack_input = np.column_stack((y_lstm, y_gru, y_rnn))
y_stack = xgb_stack.predict(stack_input)

residual_pred = xgb_res.predict(y_gru.reshape(-1,1))
y_residual = y_gru + residual_pred

# ============================================
# INVERSE TRANSFORM
# ============================================
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

def inverse(arr):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

models = {
    "LSTM": inverse(y_lstm),
    "GRU": inverse(y_gru),
    "RNN": inverse(y_rnn),
    "Ensemble": inverse(y_ens),
    "Stacking": inverse(y_stack),
    "Residual": inverse(y_residual)
}

# ============================================
# METRICS
# ============================================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    da = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    return r2, rmse, mae, mape, da

metrics = []
for name, pred in models.items():
    r2, rmse, mae, mape, da = evaluate(y_test_inv, pred)
    metrics.append([name, r2, rmse, mae, mape, da])

import pandas as pd
metrics_df = pd.DataFrame(metrics, columns=[
    "Model","R2","RMSE","MAE","MAPE (%)","Directional Accuracy"
])

st.subheader("📊 Performance Metrics")
st.dataframe(metrics_df)

best_model = metrics_df.sort_values("R2", ascending=False).iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model}")
