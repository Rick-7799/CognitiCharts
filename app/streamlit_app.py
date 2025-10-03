import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st, pandas as pd, tempfile
from cogniti_charts.utils import load_config
from cogniti_charts.infer import predict as predict_series
from cogniti_charts.image_infer import predict_image

st.set_page_config(page_title="CognitiCharts v2", layout="wide")
st.title("📈 CognitiCharts v2 — CSV & Chart Image Prediction")
cfg = load_config(); lookback = cfg["data"]["lookback"]

tab1, tab2 = st.tabs(["CSV → Predict", "Chart Image → Predict"])
with tab1:
    fw = st.selectbox("Framework", ["tf","torch"], index=0)
    uploaded = st.file_uploader("Upload CSV (date,open,high,low,close,volume)", type=["csv"])
    if st.button("Run Prediction (CSV)", use_container_width=True):
        try:
            source = "sample_data/sample_prices.csv"
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    uploaded.seek(0); tmp.write(uploaded.read()); source = tmp.name
            preds = predict_series(source, lookback=lookback, framework=fw)
            st.session_state["csv_preds"] = preds
            st.success("CSV predictions ready!")
        except Exception as e:
            st.error(f"CSV prediction failed: {e}")
    if "csv_preds" in st.session_state:
        st.dataframe(st.session_state["csv_preds"].head(20))
        st.bar_chart(st.session_state["csv_preds"]["pred_label"].value_counts())

with tab2:
    fw2 = st.selectbox("Image Framework", ["tf","torch"], index=0, key="imgfw")
    img = st.file_uploader("Upload chart image (png/jpg)", type=["png","jpg","jpeg"])
    if st.button("Run Prediction (Image)", use_container_width=True):
        try:
            if img is None: st.warning("Please upload a chart image first.")
            else:
                res = predict_image(img, framework=fw2, img_size=cfg["image"]["img_size"])
                st.write("Prediction:", res["label"]); st.json(res["probs"])
        except Exception as e:
            st.error(f"Image prediction failed: {e}")


st.divider()
st.subheader("🔍 Explainability (SHAP)")
col1, col2 = st.columns(2)
with col1:
    st.caption("Time-series model SHAP (if present)")
    for p in ["models/shap_summary_tf.png", "models/shap_summary_torch.png"]:
        try:
            st.image(p, caption=p)
        except Exception:
            pass
with col2:
    st.caption("Image model SHAP (if present)")
    for p in ["models/shap_image_summary_tf.png"]:
        try:
            st.image(p, caption=p)
        except Exception:
            pass
