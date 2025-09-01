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
# TAB 2: Bulk Dataset Upload (Prediction on Uploaded File)
# ------------------------------
with tab2:
    st.header("📂 Upload Dataset for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("👀 Preview Uploaded Data", expanded=False):
            st.dataframe(df.head(10))

        # ✅ Auto-detect target if "diagnosis" exists
        if "diagnosis" in df.columns:
            X = df.drop(columns=["diagnosis"])
            y = df["diagnosis"]

            # Convert to numeric if needed
            if y.dtype == "object":
                y = y.map({"M": 1, "B": 0})
            y = y.astype(int)
        else:
            X = df
            y = None

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split if target available
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, None, None

        # Safe SMOTE only if target exists and has both classes
        if y_train is not None and len(set(y_train)) > 1:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        # Train SVM
        svm_model = SVC(kernel="rbf", probability=True, random_state=42)
        svm_model.fit(X_res, y_res if y_res is not None else [0]*len(X_res))

        # Predictions
        y_pred = svm_model.predict(X_test)

        # ✅ Human-readable results
        results = ["🚨 Likely Malignant (High Risk)" if p == 1 
                   else "✅ Likely Benign (Low Risk)" for p in y_pred]

        output_df = pd.DataFrame({"Predicted Result": results})

        if y_test is not None:
            output_df.insert(0, "True Diagnosis", ["Malignant" if t == 1 else "Benign" for t in y_test.values])

        st.subheader("✅ Prediction Results")
        st.dataframe(output_df.reset_index(drop=True))

        # Download option
        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")


