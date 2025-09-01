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

        # Case 1: If dataset has diagnosis column → Train & evaluate
        if "diagnosis" in df.columns:
            X = df.drop(columns=["diagnosis"])
            y = df["diagnosis"]

            if y.dtype == "object":
                y = y.map({"M": 1, "B": 0})
            y = y.astype(int)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            # Train SVM
            svm_model = SVC(kernel="rbf", probability=True, random_state=42)
            svm_model.fit(X_res, y_res)

            # Predictions
            y_pred = svm_model.predict(X_test)

            st.subheader("📈 Model Performance")
            st.write(classification_report(y_test, y_pred))

        # Case 2: If dataset has NO diagnosis column → Direct prediction
        else:
            st.warning("No target column found. Running prediction only...")

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)

            # Load or train a simple model (for demo, train fresh here)
            svm_model = SVC(kernel="rbf", probability=True, random_state=42)
            # Train quickly on built-in dataset
            from sklearn.datasets import load_breast_cancer
            cancer = load_breast_cancer()
            X_train, _, y_train, _ = train_test_split(
                cancer.data, cancer.target, test_size=0.2, random_state=42
            )
            X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
            svm_model.fit(X_res, y_res)

            # Predict on uploaded file
            preds = svm_model.predict(X_scaled)

            df_results = df.copy()
            df_results["Prediction"] = ["✅ Likely Benign" if p == 0 else "🚨 Likely Malignant" for p in preds]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df_results.head(20))

            st.success("✅ Predictions generated successfully!")

