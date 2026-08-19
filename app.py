"""
Streamlit Web Application for Wine Quality Classification
M.Tech (AIML/DSE) - Machine Learning Assignment 2

This app allows users to:
1. Upload test data (CSV)
2. Select from 5 trained classification models
3. View evaluation metrics
4. View confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide"
)

# --- Title and Description ---
st.title("Wine Quality Classification App")
st.markdown("""
This application demonstrates **5 Machine Learning classification models** trained on the
**Wine Quality Dataset** (Red + White wines combined). The task is binary classification:
predicting whether a wine is of **Good Quality** (rating >= 7) or **Not Good Quality** (rating < 7).

**Dataset Features (12):** fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, color
""")

st.divider()

# --- Helper Functions ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl'
}


@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk."""
    filepath = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    with open(filepath, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    """Load the fitted scaler."""
    filepath = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(filepath, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_results():
    """Load pre-computed results."""
    filepath = os.path.join(MODEL_DIR, 'results.pkl')
    with open(filepath, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_names():
    """Load feature names."""
    filepath = os.path.join(MODEL_DIR, 'feature_names.pkl')
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix using matplotlib."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Good (0)', 'Good (1)'],
                yticklabels=['Not Good (0)', 'Good (1)'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    return fig


# --- Sidebar: Dataset Upload ---
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV test data file",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data. Must include a 'target' column."
)

# --- Load default test data or uploaded data ---
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({data.shape[0]} rows)")
else:
    default_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.sidebar.info(f"Using default test data ({data.shape[0]} rows)")
    else:
        st.error("No test data available. Please upload a CSV file.")
        st.stop()

# Check if target column exists
if 'target' not in data.columns:
    st.error("The uploaded CSV must contain a 'target' column (0 = Not Good, 1 = Good).")
    st.stop()

# --- Prepare data ---
feature_names = load_feature_names()
X_test = data[feature_names]
y_test = data['target']
scaler = load_scaler()
X_test_scaled = scaler.transform(X_test)

# --- Model Selection ---
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Classification Model:",
    list(MODEL_FILES.keys())
)

# Option to compare all models
compare_all = st.sidebar.checkbox("Compare All Models", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Dataset Info:**")
st.sidebar.write(f"- Instances: {data.shape[0]}")
st.sidebar.write(f"- Features: {len(feature_names)}")
st.sidebar.write(f"- Classes: Good (1), Not Good (0)")
st.sidebar.write(f"- Class 0 count: {(y_test == 0).sum()}")
st.sidebar.write(f"- Class 1 count: {(y_test == 1).sum()}")

# --- Main Content ---

# Tab layout
tab1, tab2, tab3 = st.tabs(["Model Comparison", "Individual Model Analysis", "Dataset Preview"])

# --- Tab 1: Model Comparison ---
with tab1:
    st.header("Model Comparison Table")

    if compare_all:
        all_metrics = {}
        for model_name in MODEL_FILES.keys():
            model = load_model(model_name)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            all_metrics[model_name] = metrics

        # Create comparison dataframe
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.round(4)

        # Style the dataframe - highlight best values
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )

        # Best model
        best_model_name = comparison_df['F1 Score'].idxmax()
        st.success(f"**Best Performing Model (by F1 Score): {best_model_name}** "
                   f"with F1 = {comparison_df.loc[best_model_name, 'F1 Score']:.4f}")

        # Bar chart comparison
        st.subheader("Visual Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        comparison_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig)

# --- Tab 2: Individual Model Analysis ---
with tab2:
    st.header(f"Analysis: {selected_model}")

    # Load and predict
    model = load_model(selected_model)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.divider()

    # Confusion Matrix and Classification Report side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(y_test, y_pred, selected_model)
        st.pyplot(fig)

    with col_right:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred,
                                       target_names=['Not Good (0)', 'Good (1)'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(4), use_container_width=True)

# --- Tab 3: Dataset Preview ---
with tab3:
    st.header("Dataset Preview")
    st.write(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Feature Statistics")
    st.dataframe(data.describe().round(3), use_container_width=True)

    st.subheader("Target Distribution")
    target_counts = y_test.value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['Not Good (0)', 'Good (1)'], [target_counts.get(0, 0), target_counts.get(1, 0)],
           color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Count')
    ax.set_title('Target Class Distribution')
    plt.tight_layout()
    st.pyplot(fig)

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>M.Tech (AIML/DSE) - Machine Learning Assignment 2 | Wine Quality Classification</p>
</div>
""", unsafe_allow_html=True)

