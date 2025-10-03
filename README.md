# CognitiCharts — Financial Pattern Recognition with SHAP

Demo / educational project inspired by Sahamati BuildAAthon 2024.  
Analyzes stock/crypto chart data and images to classify patterns like breakouts, consolidations, and reversals.  
Adds SHAP interpretability so traders can see **why** the model made a decision.  
Not financial advice — educational use only.

---

## Tech
- **Python**: data processing & utilities.  
- **TensorFlow & PyTorch**: dual training backends for flexibility.  
- **SHAP**: interpretability (feature & image attributions).  
- **Streamlit**: interactive dashboard for predictions.

---

## Quickstart

1. **Make a new Python space**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (CSV mode)**
   ```bash
   python -m cogniti_charts.train --framework tf
   ```
   or
   ```bash
   python -m cogniti_charts.train --framework torch
   ```

4. **(Optional) Train on chart images**
   ```bash
   python -m cogniti_charts.image_train --framework tf
   ```

5. **Start the app**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   A browser window opens at `http://localhost:8501`.  
   Upload CSVs or chart screenshots → see predictions and SHAP explanations.

---

## Safety
Outputs are **experimental pattern recognitions**, not trading signals.  
Always double-check with professional financial advice before acting.
