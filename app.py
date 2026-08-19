"""
Streamlit Web Application for Wine Quality Classification
M.Tech (AIML/DSE) - Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Page Configuration ---
st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")

# --- Title ---
st.title("Wine Quality Classification App")
st.markdown("""
This application demonstrates **5 Machine Learning classification models** trained on the
**Wine Quality Dataset** (Red + White wines combined). The task is binary classification:
predicting whether a wine is of **Good Quality** (rating >= 7) or **Not Good Quality** (rating < 7).

**Dataset Features (12):** fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, color
""")
st.divider()


# --- Train models once and cache ---
@st.cache_resource
def load_and_train():
    """Load dataset, train all models, return everything needed."""
    # Load datasets
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red_wine = pd.read_csv(red_wine_url, sep=';')
    white_wine = pd.read_csv(white_wine_url, sep=';')

    red_wine['color'] = 1
    white_wine['color'] = 0

    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    wine_data['target'] = (wine_data['quality'] >= 7).astype(int)
    wine_data = wine_data.drop('quality', axis=1)

    X = wine_data.drop('target', axis=1)
    y = wine_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    # Build test dataframe
    test_data = X_test.copy()
    test_data['target'] = y_test.values

    return trained_models, scaler, list(X.columns), test_data, X_test_scaled, y_test


# --- Load models ---
trained_models, scaler, feature_names, default_test_data, default_X_scaled, default_y_test = load_and_train()


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
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
    "Upload your CSV test data file", type=['csv'],
    help="Upload a CSV with same features as training data. Must include a 'target' column."
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name} ({data.shape[0]} rows)")
    if 'target' not in data.columns:
        st.error("The uploaded CSV must contain a 'target' column (0 = Not Good, 1 = Good).")
        st.stop()
    X_test = data[feature_names]
    y_test = data['target']
    X_test_scaled = scaler.transform(X_test)
else:
    data = default_test_data
    X_test_scaled = default_X_scaled
    y_test = default_y_test
    st.sidebar.info(f"Using default test data ({data.shape[0]} rows)")

# --- Model Selection ---
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a Classification Model:", list(trained_models.keys()))
compare_all = st.sidebar.checkbox("Compare All Models", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Dataset Info:**")
st.sidebar.write(f"- Instances: {data.shape[0]}")
st.sidebar.write(f"- Features: {len(feature_names)}")
st.sidebar.write(f"- Class 0 count: {int((y_test == 0).sum())}")
st.sidebar.write(f"- Class 1 count: {int((y_test == 1).sum())}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Model Comparison", "Individual Model Analysis", "Dataset Preview"])

with tab1:
    st.header("Model Comparison Table")
    if compare_all:
        all_metrics = {}
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            all_metrics[model_name] = calculate_metrics(y_test, y_pred, y_prob)

        comparison_df = pd.DataFrame(all_metrics).T.round(4)
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

        best_model_name = comparison_df['F1 Score'].idxmax()
        st.success(f"**Best Model (by F1 Score): {best_model_name}** with F1 = {comparison_df.loc[best_model_name, 'F1 Score']:.4f}")

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

with tab2:
    st.header(f"Analysis: {selected_model}")
    model = trained_models[selected_model]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
    col3.metric("Precision", f"{metrics['Precision']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{metrics['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    col6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.divider()
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
        st.dataframe(report_df.round(4))

with tab3:
    st.header("Dataset Preview")
    st.write(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    st.dataframe(data.head(20))

    st.subheader("Feature Statistics")
    st.dataframe(data.describe().round(3))

    st.subheader("Target Distribution")
    target_counts = y_test.value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['Not Good (0)', 'Good (1)'], [target_counts.get(0, 0), target_counts.get(1, 0)],
           color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Count')
    ax.set_title('Target Class Distribution')
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.markdown("<div style='text-align: center; color: gray;'><p>M.Tech (AIML/DSE) - Machine Learning Assignment 2 | Wine Quality Classification</p></div>", unsafe_allow_html=True)
