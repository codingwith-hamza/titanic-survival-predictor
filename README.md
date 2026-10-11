# Titanic Passenger Survival Predictor

**Live demo (Streamlit):** _https://codingwith-hamza-titanic-survival-predictor-app-tslyb8.streamlit.app/_

A small Streamlit web app that serves the **Support Vector Classifier** trained in
the instructor's *Titanic Passenger Survival Prediction* notebook
(AIC354 — Assignment 2, Task 1).

## Files

| File | Description |
| ---- | ----------- |
| `app.py` | The Streamlit web app |
| `svc_trained_model.pkl` | The trained Support Vector Classifier (~80% test accuracy) |
| `requirements.txt`, `runtime.txt` | Deployment dependencies (Python 3.11) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a **public** GitHub repo and upload the files above to the repo root.
2. Go to <https://share.streamlit.io> → **Create app** → deploy from GitHub.
3. Pick the repo, branch `main`, main file `app.py`, **Python version = 3.11**.
4. Deploy and copy the resulting `.streamlit.app` URL.
