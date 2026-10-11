"""
Titanic Passenger Survival Predictor
Uses the trained SVC model created by the instructor's notebook.
Keep svc_trained_model.pkl in the same folder as this file.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "svc_trained_model.pkl was not found. Put it in the same folder as app.py."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    # These category lists match the instructor's public example app.
    return {
        "PClass": LabelEncoder().fit(["First", "Second", "Third"]),
        "Gender": LabelEncoder().fit(["Male", "Female"]),
        "Sibling": LabelEncoder().fit(["Zero", "One", "Two", "Three"]),
        "Embarked": LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"]),
    }

st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Choose passenger details and ask the trained Support Vector Classifier "
    "to predict the survival label. This is an educational demo, not a reliable "
    "historical or real-world prediction tool."
)

model = load_model()
encoders = build_label_encoders()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings/Spouses Aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict Survival", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked],
    })
    encoded = pd.DataFrame({
        "PClass": encoders["PClass"].transform(user_input["PClass"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "Sibling": encoders["Sibling"].transform(user_input["Sibling"]),
        "Embarked": encoders["Embarked"].transform(user_input["Embarked"]),
    })
    prediction = model.predict(encoded)[0]
    if prediction == 1:
        st.success("### ✅ SURVIVED")
    else:
        st.error("### ❌ NOT SURVIVED")
    with st.expander("See the encoded inputs sent to the model"):
        st.dataframe(encoded, hide_index=True)
