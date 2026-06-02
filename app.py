import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tensorflow.keras.models import load_model
from datetime import timedelta
import tempfile
import gdown
import glob

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GOOG Forecast · AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Premium Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0a0d14;
    --surface:   #111520;
    --border:    #1e2538;
    --accent:    #4ade80;
    --accent2:   #22d3ee;
    --muted:     #4a5578;
    --text:      #e2e8f0;
    --text-dim:  #8892a4;
    --danger:    #f87171;
    --warn:      #fbbf24;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── App container ── */
.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1300px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ══════════════════════════════════════
   HERO HEADER
══════════════════════════════════════ */
.hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
}
.hero-left .ticker {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.hero-left h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1;
    color: #fff;
    margin: 0;
}
.hero-left h1 span {
    color: var(--accent);
}
.hero-left .subtitle {
    font-size: 0.82rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
    letter-spacing: 0.03em;
}
.hero-right {
    text-align: right;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.1em;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ══════════════════════════════════════
   STAT CARDS ROW
══════════════════════════════════════ */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0.6;
}
.stat-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
}
.stat-card .sub {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 0.2rem;
}
.stat-card .arrow-up   { color: var(--accent); }
.stat-card .arrow-down { color: var(--danger); }

/* ══════════════════════════════════════
   CONTROL PANEL
══════════════════════════════════════ */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 2rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Radio pills override */
div[role="radiogroup"] {
    display: flex !important;
    gap: 0.6rem;
    flex-wrap: wrap;
}
div[role="radiogroup"] label {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    font-size: 0.8rem !important;
    color: var(--text-dim) !important;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: rgba(74,222,128,0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    padding-top: 1rem;
}
.stSlider [data-baseweb="track-background"] {
    background: var(--border) !important;
}
.stSlider [data-baseweb="track-active"] {
    background: var(--accent) !important;
}
.stSlider [data-baseweb="thumb"] {
    background: var(--accent) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 0 4px rgba(74,222,128,0.2) !important;
}
.stSlider p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text-dim) !important;
}

/* Labels */
.stRadio label p, .stSelectbox label p, .stSlider label p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.6rem !important;
}

/* ── Primary Button ── */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #0a0d14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(74,222,128,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(74,222,128,0.35) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════════════
   RESULT SECTION
══════════════════════════════════════ */
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
}
.result-header h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.badge {
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.3);
    color: var(--accent2);
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── DataFrame table ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
.stDataFrame thead tr th {
    background: var(--bg) !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr td {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr:hover td {
    background: rgba(74,222,128,0.04) !important;
}

/* ── Alert boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    font-size: 0.82rem !important;
}
.stAlert[data-baseweb="notification"] {
    background: rgba(74,222,128,0.06) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info/warning ── */
.stInfo { background: rgba(34,211,238,0.05) !important; border-left: 3px solid var(--accent2) !important; }
.stWarning { background: rgba(251,191,36,0.05) !important; border-left: 3px solid var(--warn) !important; }
.stError { background: rgba(248,113,113,0.05) !important; border-left: 3px solid var(--danger) !important; }

/* ── Separator line ── */
.sep {
    height: 1px;
    background: var(--border);
    margin: 2rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding-top: 2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="ticker">NASDAQ · GOOG · Forecast</div>
    <h1>Google Stock<br><span>Forecast</span> Dashboard</h1>
    <div class="subtitle">Deep learning ensemble — LSTM · GRU · RNN · Stacking · Residual</div>
  </div>
  <div class="hero-right">
    <div class="live-badge">
      <div class="live-dot"></div>
      Model Ready
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DOWNLOAD & CACHE
# ─────────────────────────────────────────────
FOLDER_ID  = "1HQIKbA8tkmhOnOlsve3fRuvcZXcD9vzc"
GDRIVE_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

if 'temp_dir_path' not in st.session_state:
    st.session_state.temp_dir_path = None

@st.cache_resource
def download_assets():
    temp_path = tempfile.mkdtemp()
    st.session_state.temp_dir_path = temp_path
    with st.spinner("⏳ Mengunduh aset dari Google Drive…"):
        try:
            gdown.download_folder(url=GDRIVE_URL, output=temp_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh: {e}")
            return None
    files_list = []
    for root, _, files in os.walk(temp_path):
        for f in files:
            files_list.append(os.path.join(root, f))
    with st.expander("📁 File yang diunduh", expanded=False):
        st.code("\n".join(files_list))
    return temp_path

SAVE_PATH = download_assets()


# ─────────────────────────────────────────────
# STAT CARDS  (populated after CSV load)
# ─────────────────────────────────────────────
def find_csv_file(folder):
    if not os.path.isdir(folder):
        return None
    direct = os.path.join(folder, "GOOG.csv")
    if os.path.isfile(direct):
        return direct
    candidates = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    return candidates[0] if candidates else None

def render_stat_cards(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close_chg = last['Close'] - prev['Close']
        close_pct  = close_chg / prev['Close'] * 100
        vol_b = last['Volume'] / 1e9 if 'Volume' in df.columns else None
        high52 = df['High'].max() if 'High' in df.columns else None
        arrow = "▲" if close_chg >= 0 else "▼"
        arrow_cls = "arrow-up" if close_chg >= 0 else "arrow-down"

        vol_html = f'<div class="value">${vol_b:.2f}B</div><div class="sub">Volume terakhir</div>' if vol_b else '<div class="value">—</div>'
        high_html = f'<div class="value">${high52:,.2f}</div><div class="sub">52-week high</div>' if high52 else '<div class="value">—</div>'

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card">
            <div class="label">Harga Penutupan</div>
            <div class="value">${last['Close']:,.2f}</div>
            <div class="sub"><span class="{arrow_cls}">{arrow} {abs(close_chg):.2f} ({abs(close_pct):.2f}%)</span> hari terakhir</div>
          </div>
          <div class="stat-card">
            <div class="label">Volume</div>
            {vol_html}
          </div>
          <div class="stat-card">
            <div class="label">52-Week High</div>
            {high_html}
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass  # stat cards are decorative; skip silently if CSV unavailable yet

if SAVE_PATH:
    csv_path = find_csv_file(SAVE_PATH)
    if csv_path:
        render_stat_cards(csv_path)


# ─────────────────────────────────────────────
# CONTROL PANEL
# ─────────────────────────────────────────────
st.markdown("""
<div class="panel-title">⚙ &nbsp;Konfigurasi Model</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 3], gap="large")

with col1:
    category = st.radio(
        "Kategori Model",
        ["Model Dasar", "Model Ditingkatkan"],
        horizontal=False,
    )

with col2:
    if category == "Model Dasar":
        model_choice = st.selectbox("Arsitektur", ["LSTM", "GRU", "RNN"])
    else:
        model_choice = st.selectbox("Strategi Ensemble", ["Ensemble", "Stacking", "Residual"])

with col3:
    days = st.slider("Jangka Waktu Prediksi (hari)", 1, 365, 30)
    st.markdown(f"""
    <div style="margin-top:-0.6rem;font-family:'DM Mono',monospace;font-size:0.7rem;color:#4a5578;">
        → Memprediksi <b style="color:#4ade80">{days}</b> hari ke depan
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

col_btn, col_spacer = st.columns([2, 5])
with col_btn:
    predict_btn = st.button(
        f"Jalankan Prediksi  →",
        type="primary",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# FORECAST LOGIC
# ─────────────────────────────────────────────
def forecast_stock(category, model_choice, days, local_path):
    if local_path is None:
        st.error("Folder aset tidak ditemukan. Pastikan unduhan berhasil.")
        return pd.DataFrame(), None

    csv_file = find_csv_file(local_path)
    if csv_file is None:
        st.error("Tidak ditemukan file CSV di folder unduhan.")
        return pd.DataFrame(), None
    st.info(f"📄 Data sumber: **{os.path.basename(csv_file)}**")

    scaler_path = os.path.join(local_path, "scaler.pkl")
    if not os.path.exists(scaler_path):
        st.error("scaler.pkl tidak ditemukan.")
        return pd.DataFrame(), None
    sc = joblib.load(scaler_path)

    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    scaled_data = sc.transform(df[['Close']])

    window_size = 60
    temp_input  = scaled_data[-window_size:].copy()
    pred_output = []
    blend       = 0.3

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
                y_pred  = model.predict(x_input, verbose=0)[0][0]
                stab    = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stab)
                temp_input = np.vstack((temp_input[1:], [[stab]]))

        else:
            m_gru  = load_model(os.path.join(local_path, "model_gru.h5"),  compile=False)
            m_lstm = load_model(os.path.join(local_path, "model_lstm.h5"), compile=False)
            m_rnn  = load_model(os.path.join(local_path, "model_rnn.h5"),  compile=False)
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
                g = m_gru.predict(x_input,  verbose=0)[0][0]
                l = m_lstm.predict(x_input, verbose=0)[0][0]
                r = m_rnn.predict(x_input,  verbose=0)[0][0]
                if model_choice == "Ensemble":
                    y_pred = 0.6*g + 0.3*l + 0.1*r
                elif model_choice == "Stacking":
                    y_pred = meta.predict([[g, l, r]])[0]
                else:
                    resid  = resid_mdl.predict(x_input.reshape(1, -1))[0]
                    y_pred = g + resid
                stab = blend * y_pred + (1 - blend) * temp_input[-1, 0]
                pred_output.append(stab)
                temp_input = np.vstack((temp_input[1:], [[stab]]))

        pred_array = np.array(pred_output).reshape(-1, 1)
        inv_pred   = sc.inverse_transform(pred_array).flatten()
        last_date  = df['Date'].iloc[-1]
        date_range = pd.date_range(last_date + timedelta(days=1), periods=len(inv_pred))

        result_df = pd.DataFrame({
            'Tanggal':    date_range.strftime('%Y-%m-%d'),
            'Harga (USD)': inv_pred.round(2),
        })

        # ── Plot with dark theme ──
        fig = plt.figure(figsize=(12, 5))
        fig.patch.set_facecolor('#111520')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0a0d14')

        # Gradient fill
        ax.fill_between(date_range, inv_pred, alpha=0.12, color='#4ade80')

        # Confidence band (±2% synthetic)
        upper = inv_pred * 1.02
        lower = inv_pred * 0.98
        ax.fill_between(date_range, lower, upper, alpha=0.08, color='#22d3ee', label='±2% band')

        # Main line
        ax.plot(date_range, inv_pred, color='#4ade80', linewidth=2.5, zorder=5, label='Prediksi')

        # Dots at first and last
        ax.scatter([date_range[0], date_range[-1]],
                   [inv_pred[0], inv_pred[-1]],
                   color='#4ade80', s=60, zorder=6)

        # Annotations
        ax.annotate(f"${inv_pred[0]:,.2f}",
                    xy=(date_range[0], inv_pred[0]),
                    xytext=(8, 8), textcoords='offset points',
                    color='#8892a4', fontsize=8, fontfamily='monospace')
        ax.annotate(f"${inv_pred[-1]:,.2f}",
                    xy=(date_range[-1], inv_pred[-1]),
                    xytext=(-70, 8), textcoords='offset points',
                    color='#4ade80', fontsize=9, fontfamily='monospace', fontweight='bold')

        # Styling
        ax.set_title(f"{model_choice}  ·  {days}-Day Forecast", color='#e2e8f0',
                     fontsize=13, fontfamily='sans-serif', pad=14)
        ax.tick_params(colors='#4a5578', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2538')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.grid(True, color='#1e2538', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Tanggal", color='#4a5578', fontsize=9)
        ax.set_ylabel("Harga Penutupan (USD)", color='#4a5578', fontsize=9)

        # Legend
        patch_pred = mpatches.Patch(color='#4ade80', label='Prediksi')
        patch_band = mpatches.Patch(color='#22d3ee', alpha=0.4, label='±2% Band')
        ax.legend(handles=[patch_pred, patch_band], framealpha=0,
                  labelcolor='#8892a4', fontsize=8, loc='upper left')

        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()

        return result_df, fig

    except Exception as e:
        import traceback
        st.error(f"Error saat prediksi:\n```\n{traceback.format_exc()}\n```")
        return pd.DataFrame(), None


# ─────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────
if SAVE_PATH is None:
    st.error("❌ Gagal mengunduh aset. Pastikan folder Google Drive dibagikan secara publik.")
    st.stop()

if predict_btn:
    with st.spinner(f"🧠  {model_choice} sedang menghitung {days} hari prediksi…"):
        df_result, plot_fig = forecast_stock(category, model_choice, days, SAVE_PATH)

    if not df_result.empty:
        st.success("✅  Prediksi berhasil dijalankan!")

        # Summary mini-stats
        min_p = df_result['Harga (USD)'].min()
        max_p = df_result['Harga (USD)'].max()
        avg_p = df_result['Harga (USD)'].mean()
        trend = "▲ Bullish" if df_result['Harga (USD)'].iloc[-1] > df_result['Harga (USD)'].iloc[0] else "▼ Bearish"
        trend_color = "#4ade80" if "Bullish" in trend else "#f87171"

        st.markdown(f"""
        <div class="stat-row" style="margin-top:1rem;">
          <div class="stat-card">
            <div class="label">Harga Minimum</div>
            <div class="value">${min_p:,.2f}</div>
            <div class="sub">Dalam periode prediksi</div>
          </div>
          <div class="stat-card">
            <div class="label">Harga Maksimum</div>
            <div class="value">${max_p:,.2f}</div>
            <div class="sub">Dalam periode prediksi</div>
          </div>
          <div class="stat-card">
            <div class="label">Rata-rata &amp; Tren</div>
            <div class="value">${avg_p:,.2f}</div>
            <div class="sub"><span style="color:{trend_color};">{trend}</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_chart, col_table = st.columns([3, 2], gap="large")

        with col_chart:
            st.markdown("""
            <div class="result-header">
              <h3>📈  Visualisasi Prediksi</h3>
              <span class="badge">Chart</span>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(plot_fig)

        with col_table:
            st.markdown("""
            <div class="result-header">
              <h3>🗓  Data Harian</h3>
              <span class="badge">Exportable</span>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True, height=440)

    else:
        st.warning("⚠️  Prediksi tidak dapat dijalankan. Periksa kembali file model dan data.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="sep"></div>
<div class="footer">
  GOOG Forecast Dashboard · Powered by TensorFlow &amp; Streamlit · Model: Deep Learning Ensemble
</div>
""", unsafe_allow_html=True)
