import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ==============================
# Page Setup
# ==============================
st.set_page_config(
    page_title="Breast Cancer Classifier 💖",
    page_icon="💖",
    layout="wide"
)

st.title("💖 Breast Cancer Classifier")
st.markdown("A patient-friendly app for breast cancer prediction using **SVM + SMOTE**")

# ==============================
# Tabs: Single Prediction | Dataset Upload
# ==============================
tab1, tab2 = st.tabs(["🧑‍⚕️ Single Patient Prediction", "📂 Bulk Prediction (Upload CSV)"])

# ------------------------------
# TAB 1: Single Patient Prediction
# ------------------------------
with tab1:
    st.header("🧑‍⚕️ Enter Patient Data")

    st.markdown("Provide key tumor features for diagnosis:")

    # Example features with sliders
    mean_radius = st.slider("Mean Radius", 0.0, 30.0, 14.0, 0.1)
    mean_texture = st.slider("Mean Texture", 0.0, 40.0, 19.0, 0.1)
    mean_perimeter = st.slider("Mean Perimeter", 0.0, 200.0, 90.0, 0.1)
    mean_area = st.slider("Mean Area", 0.0, 2500.0, 500.0, 1.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.0, 0.2, 0.1, 0.001)

    # Predict button
    if st.button("🔍 Predict Diagnosis"):
        # For now using demo logic, later connect to model
        if mean_radius > 15:
            st.error("🚨 Likely Malignant (High Risk)")
        else:
            st.success("✅ Likely Benign (Low Risk)")

# ------------------------------
# TAB 2: Bulk Dataset Upload
# ------------------------------
with tab2:
    st.header("📂 Upload Dataset for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("👀 Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10))

        # Load sklearn cancer dataset for training
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        X_train, _, y_train, _ = train_test_split(
            cancer.data, cancer.target, test_size=0.2, random_state=42
        )

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_res = scaler.fit_transform(X_res)

        svm_model = SVC(kernel="rbf", probability=True, random_state=42)
        svm_model.fit(X_res, y_res)

        # ✅ Align uploaded data with training features
        required_features = cancer.feature_names
        for col in required_features:
            if col not in df.columns:
                df[col] = 0  # fill missing features with zeros

        df = df[required_features]  # keep only required features

        # Scale + Predict
        X_scaled = scaler.transform(df)
        preds = svm_model.predict(X_scaled)

        # Show results row by row
        st.subheader("🔍 Prediction Results")
        benign_count = 0
        malignant_count = 0
        for i, p in enumerate(preds, start=1):
            if p == 0:
                st.success(f"🧑‍⚕️ Patient {i}: ✅ Likely Benign (Low Risk)")
                benign_count += 1
            else:
                st.error(f"🧑‍⚕️ Patient {i}: 🚨 Likely Malignant (High Risk)")
                malignant_count += 1

        # ✅ Summary
        st.info(f"📊 Summary: {benign_count} Benign, {malignant_count} Malignant")
