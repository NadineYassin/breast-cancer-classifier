import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("💖 Breast Cancer Prediction App")
st.write("Upload your dataset and predict breast cancer using SVM + SMOTE for better recall.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    # Features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Train SVM
    svm_model = SVC(kernel="rbf", probability=True)
    svm_model.fit(X_res, y_res)

    # Predictions
    y_pred = svm_model.predict(X_test)

    # Metrics
    st.subheader("📈 Model Performance")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.success(f"✅ Recall for class 1: {report['1']['recall'] * 100:.2f}%")
else:
    st.info("👆 Please upload a CSV file to start.")
