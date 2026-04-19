# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ── LOAD ──────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     pkg = joblib.load(os.path.join(BASE_DIR, "fraud_detection_model.pkl"))
#     return pkg["model"], pkg["preprocessor"], pkg["threshold"]

# model, preprocessor, threshold = load_model()

# # ── CSS ───────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
# html, body, [class*="css"] {
#     font-family: 'Inter', sans-serif !important;
#     background-color: #0f1117 !important;
#     color: #f0f2f6 !important;
# }
# [data-testid="stSidebar"] {
#     background: #161b22 !important;
#     border-right: 1px solid #21262d !important;
# }
# .metric-box {
#     background: #161b22;
#     border: 1px solid #21262d;
#     border-radius: 10px;
#     padding: 1.2rem 1.4rem;
#     text-align: center;
# }
# .metric-box .label {
#     font-size: 0.7rem;
#     text-transform: uppercase;
#     letter-spacing: 0.1em;
#     color: #8b949e;
#     margin-bottom: 0.4rem;
# }
# .metric-box .value {
#     font-size: 1.7rem;
#     font-weight: 700;
#     color: #58a6ff;
# }
# .fraud-box {
#     background: rgba(248,81,73,0.1);
#     border: 1px solid rgba(248,81,73,0.4);
#     border-radius: 10px;
#     padding: 1.4rem;
#     text-align: center;
#     font-size: 1.2rem;
#     font-weight: 700;
#     color: #f85149;
# }
# .legit-box {
#     background: rgba(63,185,80,0.1);
#     border: 1px solid rgba(63,185,80,0.4);
#     border-radius: 10px;
#     padding: 1.4rem;
#     text-align: center;
#     font-size: 1.2rem;
#     font-weight: 700;
#     color: #3fb950;
# }
# .auto-tag {
#     background: #21262d;
#     border-radius: 4px;
#     padding: 0.1rem 0.4rem;
#     font-size: 0.6rem;
#     color: #8b949e;
#     margin-left: 0.3rem;
# }
# div[data-testid="stButton"] button {
#     width: 100%;
#     background: #238636 !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 8px !important;
#     padding: 0.7rem !important;
#     font-size: 1rem !important;
#     font-weight: 600 !important;
# }
# div[data-testid="stButton"] button:hover {
#     background: #2ea043 !important;
# }
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding-top: 1.5rem !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── PAGE CONFIG ───────────────────────────────────────────────
# st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

# st.markdown("## 🛡️ Fraud Detection System")
# st.caption("XGBoost · Calibrated Probabilities · TRANSFER & CASH_OUT")
# st.divider()

# # ── SIDEBAR ───────────────────────────────────────────────────
# st.sidebar.markdown("### ⚙️ Transaction Input")

# step       = st.sidebar.number_input("Step", value=140, min_value=1)
# type_label = st.sidebar.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
# amount     = st.sidebar.number_input("Amount ($)", value=285086.0, min_value=0.0, step=1000.0)

# st.sidebar.markdown("### 💰 Balance Info")
# oldbalanceOrg  = st.sidebar.number_input("Old Balance (Sender)",   value=229352.0, min_value=0.0)
# newbalanceOrg  = st.sidebar.number_input("New Balance (Sender)",   value=0.0,      min_value=0.0)
# oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", value=0.0,      min_value=0.0)
# newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", value=285086.0, min_value=0.0)

# st.sidebar.divider()
# predict_btn = st.sidebar.button("🔍 Predict")

# # ── ENCODING (0=CASH_OUT, 1=TRANSFER) ────────────────────────
# type_ = 1 if type_label == "TRANSFER" else 0

# # ── FEATURE ENGINEERING ───────────────────────────────────────
# log_amount        = np.log(amount + 1)
# balance_diff_orig = oldbalanceOrg  - newbalanceOrg
# balance_diff_dest = newbalanceDest - oldbalanceDest
# is_drained        = 1 if (newbalanceOrg == 0 and oldbalanceOrg > 0) else 0

# # ── INPUT DATAFRAME ───────────────────────────────────────────
# input_data = pd.DataFrame({
#     "step":              [step],
#     "type":              [type_],
#     "oldbalanceOrg":     [oldbalanceOrg],
#     "oldbalanceDest":    [oldbalanceDest],
#     "log_amount":        [log_amount],
#     "balance_diff_orig": [balance_diff_orig],
#     "balance_diff_dest": [balance_diff_dest],
#     "is_drained":        [is_drained],
# })

# # ── LAYOUT ────────────────────────────────────────────────────
# left, right = st.columns([1.3, 1], gap="large")

# with left:
#     st.markdown("#### 📋 Input Data")
#     st.dataframe(input_data, use_container_width=True, hide_index=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("#### 🔧 Auto-Computed Features")

#     a1, a2, a3 = st.columns(3)
#     a1.markdown(f"""<div class="metric-box">
#         <div class="label">Log Amount <span class="auto-tag">auto</span></div>
#         <div class="value" style="font-size:1.2rem;">{log_amount:.3f}</div>
#     </div>""", unsafe_allow_html=True)
#     a2.markdown(f"""<div class="metric-box">
#         <div class="label">Balance Diff <span class="auto-tag">auto</span></div>
#         <div class="value" style="font-size:1.2rem;">${balance_diff_orig:,.0f}</div>
#     </div>""", unsafe_allow_html=True)
#     a3.markdown(f"""<div class="metric-box">
#         <div class="label">Is Drained <span class="auto-tag">auto</span></div>
#         <div class="value" style="font-size:1.2rem; color:{'#f85149' if is_drained else '#3fb950'}">
#             {'Yes ⚠️' if is_drained else 'No ✓'}
#         </div>
#     </div>""", unsafe_allow_html=True)

# with right:
#     st.markdown("#### 🔍 Result")

#     if not predict_btn:
#         st.info("👈 Fill in the details and click **Predict** to analyze.")
#     else:
#         try:
#             processed  = preprocessor.transform(input_data)
#             proba      = model.predict_proba(processed)[0][1]
#             prediction = int(proba >= threshold)

#             m1, m2 = st.columns(2)
#             m1.markdown(f"""<div class="metric-box">
#                 <div class="label">Fraud Probability</div>
#                 <div class="value" style="color:{'#f85149' if prediction else '#3fb950'}">
#                     {proba:.4f}
#                 </div>
#             </div>""", unsafe_allow_html=True)
#             m2.markdown(f"""<div class="metric-box">
#                 <div class="label">Confidence Gap</div>
#                 <div class="value" style="font-size:1.2rem; color:#8b949e;">
#                     {abs(proba - threshold):.4f}
#                 </div>
#             </div>""", unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)
#             st.progress(float(min(proba, 1.0)))
#             if proba < threshold * 0.5:
#              risk, rc = "🟢 LOW RISK",    "#3fb950"
#             elif proba < threshold:
#                 risk, rc = "🟡 MEDIUM RISK", "#d29922"
#             else:
#                 risk, rc = "🔴 HIGH RISK",   "#f85149"

#             st.markdown(f"""<div class="metric-box" style="margin-top:1rem;">
#                 <div class="label">Risk Level</div>
#                 <div class="value" style="color:{rc}; font-size:1.1rem;">{risk}</div>
#             </div>""", unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)
#             if prediction:
#                 st.markdown('<div class="fraud-box">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="legit-box">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.code(str(input_data.dtypes))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown


# ── LOAD MODEL FROM GOOGLE DRIVE ─────────────────────────────
@st.cache_resource
def load_model():
    model_path = "fraud_detection_model.pkl"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... please wait ⏳"):
            gdown.download(
                "https://drive.google.com/uc?id=1YlIFX9BNlbwf0GFe_vz4nZ0iDzXsGHeE",
                model_path,
                quiet=False
            )
    pkg = joblib.load(model_path)
    return pkg["model"], pkg["preprocessor"], pkg["threshold"]

model, preprocessor, threshold = load_model()

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0f1117 !important;
    color: #f0f2f6 !important;
}
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
.metric-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.4rem;
}
.metric-box .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #58a6ff;
}
.fraud-box {
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.4);
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f85149;
}
.legit-box {
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.4);
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 700;
    color: #3fb950;
}
.auto-tag {
    background: #21262d;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.6rem;
    color: #8b949e;
    margin-left: 0.3rem;
}
div[data-testid="stButton"] button {
    width: 100%;
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] button:hover {
    background: #2ea043 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown("## 🛡️ Fraud Detection System")
st.caption("XGBoost · scale_pos_weight · TRANSFER & CASH_OUT · by Shreyansh Awasthi")
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Transaction Input")

step       = st.sidebar.number_input("Step", value=140, min_value=1)
type_label = st.sidebar.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
amount     = st.sidebar.number_input("Amount ($)", value=285086.0, min_value=0.0, step=1000.0)

st.sidebar.markdown("### 💰 Balance Info")
oldbalanceOrg  = st.sidebar.number_input("Old Balance (Sender)",   value=229352.0, min_value=0.0)
newbalanceOrg  = st.sidebar.number_input("New Balance (Sender)",   value=0.0,      min_value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", value=0.0,      min_value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", value=285086.0, min_value=0.0)

st.sidebar.divider()
predict_btn = st.sidebar.button("🔍 Predict")

# ── ENCODING (0=CASH_OUT, 1=TRANSFER) ────────────────────────
type_ = 1 if type_label == "TRANSFER" else 0

# ── FEATURE ENGINEERING ───────────────────────────────────────
log_amount        = np.log(amount + 1)
balance_diff_orig = oldbalanceOrg  - newbalanceOrg
balance_diff_dest = newbalanceDest - oldbalanceDest
is_drained        = 1 if (newbalanceOrg == 0 and oldbalanceOrg > 0) else 0

# ── INPUT DATAFRAME ───────────────────────────────────────────
input_data = pd.DataFrame({
    "step":              [step],
    "type":              [type_],
    "oldbalanceOrg":     [oldbalanceOrg],
    "oldbalanceDest":    [oldbalanceDest],
    "log_amount":        [log_amount],
    "balance_diff_orig": [balance_diff_orig],
    "balance_diff_dest": [balance_diff_dest],
    "is_drained":        [is_drained],
})

# ── LAYOUT ────────────────────────────────────────────────────
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown("#### 📋 Input Data")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔧 Auto-Computed Features")

    a1, a2, a3 = st.columns(3)
    a1.markdown(f"""<div class="metric-box">
        <div class="label">Log Amount <span class="auto-tag">auto</span></div>
        <div class="value" style="font-size:1.2rem;">{log_amount:.3f}</div>
    </div>""", unsafe_allow_html=True)
    a2.markdown(f"""<div class="metric-box">
        <div class="label">Balance Diff <span class="auto-tag">auto</span></div>
        <div class="value" style="font-size:1.2rem;">${balance_diff_orig:,.0f}</div>
    </div>""", unsafe_allow_html=True)
    a3.markdown(f"""<div class="metric-box">
        <div class="label">Is Drained <span class="auto-tag">auto</span></div>
        <div class="value" style="font-size:1.2rem; color:{'#f85149' if is_drained else '#3fb950'}">
            {'Yes ⚠️' if is_drained else 'No ✓'}
        </div>
    </div>""", unsafe_allow_html=True)

with right:
    st.markdown("#### 🔍 Result")

    if not predict_btn:
        st.info("👈 Fill in the details and click **Predict** to analyze.")
    else:
        try:
            processed  = preprocessor.transform(input_data)
            proba      = model.predict_proba(processed)[0][1]
            prediction = int(proba >= threshold)

            m1, m2 = st.columns(2)
            m1.markdown(f"""<div class="metric-box">
                <div class="label">Fraud Probability</div>
                <div class="value" style="color:{'#f85149' if prediction else '#3fb950'}">
                    {proba:.4f}
                </div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-box">
                <div class="label">Confidence Gap</div>
                <div class="value" style="font-size:1.2rem; color:#8b949e;">
                    {abs(proba - threshold):.4f}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(min(proba, 1.0)))

            if proba < 0.15:
                risk, rc = "🟢 LOW RISK",    "#3fb950"
            elif proba < threshold:
                risk, rc = "🟡 MEDIUM RISK", "#d29922"
            else:
                risk, rc = "🔴 HIGH RISK",   "#f85149"

            st.markdown(f"""<div class="metric-box" style="margin-top:1rem;">
                <div class="label">Risk Level</div>
                <div class="value" style="color:{rc}; font-size:1.1rem;">{risk}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction:
                st.markdown('<div class="fraud-box">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="legit-box">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.code(str(input_data.dtypes))