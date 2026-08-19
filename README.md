# Wine Quality Classification - ML Assignment 2

## a. Problem Statement

Predict whether a wine is of **Good Quality** (rating >= 7) or **Not Good Quality** (rating < 7) based on its physicochemical properties. This is a **binary classification** problem where we combine the UCI Red and White Wine Quality datasets and use 12 input features to classify wine quality.

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) |
| **Type** | Binary Classification |
| **Total Instances** | 6,497 (1,599 red + 4,898 white) |
| **Number of Features** | 12 |
| **Target Variable** | Quality (0 = Not Good, 1 = Good) |
| **Class Distribution** | ~83% Not Good, ~17% Good |

### Features:
1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Color (1 = Red, 0 = White)

## c. GitHub Repository Link

[GitHub Repository](https://github.com/keerthi2109/ml-classification-app)

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8508 | 0.8019 | 0.6220 | 0.6172 | 0.6196 | 0.5268 |
| KNN | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| Naive Bayes | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Random Forest (Ensemble) | 0.8869 | 0.9123 | 0.8079 | 0.5586 | 0.6605 | 0.6100 |

### Model Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Accuracy of 82.2% with the highest precision (61.5%) among simpler models but very low recall (26.2%), indicating highly conservative predictions. The linear decision boundary captures the primary separable patterns but misses many good-quality wines. F1 score of 0.367 reflects the precision-recall imbalance. Suitable when false positives are costly. |
| Decision Tree | Best balance of precision (62.2%) and recall (61.7%) among non-ensemble models, yielding F1=0.6196. With max_depth=10 constraint, it avoids severe overfitting while capturing non-linear feature interactions (e.g., alcohol + volatile acidity thresholds). MCC of 0.5268 shows strong correlation between predictions and actuals. |
| KNN | Accuracy of 83.2% with moderate precision (59.2%) and recall (47.7%). As a distance-based learner, it benefits from StandardScaler preprocessing. AUC of 0.8264 (second highest after Random Forest) indicates good ranking ability. The k=5 setting smooths out noisy neighbors but limits recall on the minority class. |
| Naive Bayes | Lowest accuracy (73.5%) and precision (39.0%) due to the strong feature independence assumption, which is violated by correlated chemical properties (density-alcohol, acidity-pH). However, achieves recall of 61.7% (tied with Decision Tree), catching more good-quality wines at the cost of many false positives. MCC of 0.3268 is the lowest, confirming weaker overall predictive power. |
| Random Forest (Ensemble) | **Best overall performer** across all metrics. Accuracy: 88.7%, AUC: 0.9123 (significantly ahead), Precision: 80.8% (far exceeds others), F1: 0.6605, MCC: 0.6100. The ensemble of 100 bagged decision trees reduces variance and captures complex feature interactions. Random feature subsampling decorrelates trees, making predictions robust to noise and collinearity. |
| **Overall Winner** | **Random Forest** - Dominates across every metric, especially AUC (0.9123) and Precision (0.8079). Its ensemble averaging produces well-calibrated probabilities and the highest discriminative power for separating good vs. not-good quality wines. |

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python model/train_models.py
```

### 3. Run Streamlit App Locally
```bash
streamlit run app.py
```

### 4. Deployed App
[Live Streamlit App](https://ml-classification-app-hvtt3qsfsist7avuphghq3.streamlit.app/)

## Project Structure
```
ml-classification-app/
│-- app.py                  # Streamlit web application
│-- requirements.txt        # Python dependencies
│-- README.md               # Project documentation
│-- test_data.csv           # Test dataset for evaluation
│-- model/
│   │-- train_models.py     # Model training script
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- scaler.pkl
│   │-- results.pkl
│   └-- feature_names.pkl
```

## Technologies Used
- Python 3.12
- Scikit-learn (Machine Learning models)
- Streamlit (Web Application)
- Pandas & NumPy (Data manipulation)
- Matplotlib & Seaborn (Visualization)
