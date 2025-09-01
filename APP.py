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
# TAB 2: Bulk Dataset Upload (Simplified with Final Result)
# ------------------------------
with tab2:
    st.header("📂 Upload Dataset for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("👀 Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10))

        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        if target_column:
            # Features & target
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # ✅ Convert target to numeric if needed
            if y.dtype == "object":
                if set(y.unique()) <= {"M", "B"}:
                    y = y.map({"M": 1, "B": 0})
                else:
                    st.error("❌ Target column must be numeric or contain only 'M'/'B'")
                    st.stop()
            y = y.astype(int)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # ✅ Safe SMOTE (only if both classes exist)
            if len(set(y_train)) > 1:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X_train, y_train)
            else:
                X_res, y_res = X_train, y_train

            # Train SVM
            svm_model = SVC(kernel="rbf", random_state=42)
            svm_model.fit(X_res, y_res)

            # Predictions
            y_pred = svm_model.predict(X_test)

            # ✅ Final readable results
            results = ["🚨 Likely Malignant (High Risk)" if pred == 1 
                       else "✅ Likely Benign (Low Risk)" for pred in y_pred]

            st.subheader("✅ Final Predictions")
            st.dataframe(pd.DataFrame({
                "True Diagnosis": ["Malignant" if t == 1 else "Benign" for t in y_test.values],
                "Predicted Result": results
            }).reset_index(drop=True))

