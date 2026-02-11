# ============================================
# 🔒 GLOBAL REPRODUCIBILITY LOCK
# ============================================
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# IMPORT
# ============================================
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("GOOG.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Close"]])

window = 60
X, y = [], []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ============================================
# BUILD MODELS
# ============================================
def build_lstm():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru():
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        GRU(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_rnn():
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        SimpleRNN(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

lstm = build_lstm()
gru = build_gru()
rnn = build_rnn()

print("Training LSTM...")
lstm.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False)

print("Training GRU...")
gru.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False)

print("Training RNN...")
rnn.fit(X_train, y_train, epochs=50, batch_size=32, shuffle=False)

# ============================================
# STACKING MODEL
# ============================================
lstm_pred = lstm.predict(X_train, verbose=0).flatten()
gru_pred = gru.predict(X_train, verbose=0).flatten()
rnn_pred = rnn.predict(X_train, verbose=0).flatten()

stack_input = np.column_stack((lstm_pred, gru_pred, rnn_pred))

xgb_stack = xgb.XGBRegressor(
    n_estimators=200,
    random_state=SEED,
    seed=SEED,
    subsample=1,
    colsample_bytree=1
)

xgb_stack.fit(stack_input, y_train)

# ============================================
# RESIDUAL LEARNING
# ============================================
residual = y_train - gru_pred

xgb_residual = xgb.XGBRegressor(
    n_estimators=200,
    random_state=SEED,
    seed=SEED,
    subsample=1,
    colsample_bytree=1
)

xgb_residual.fit(gru_pred.reshape(-1,1), residual)

# ============================================
# SAVE MODELS
# ============================================
os.makedirs("models", exist_ok=True)

lstm.save("models/lstm.h5")
gru.save("models/gru.h5")
rnn.save("models/rnn.h5")

joblib.dump(xgb_stack, "models/xgb_stacking.pkl")
joblib.dump(xgb_residual, "models/xgb_residual.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ============================================
# VALIDATION CHECK
# ============================================
y_test_pred = lstm.predict(X_test, verbose=0).flatten()
print("LOCK TEST R2 (Scaled):", r2_score(y_test, y_test_pred))

print("ALL MODELS SAVED SUCCESSFULLY")
