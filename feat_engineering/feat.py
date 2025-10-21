'''
Heart Disease Prediction (Classification)

Dataset: Heart Disease UCI
Why: Health dataset with both categorical & numeric data, good for binary classification.
What to do:
Predict if a patient has heart disease (target).
Handle missing values.
Feature scaling for Logistic Regression / SVM.
Compare model performance with ROC-AUC.
Skills practiced: Data Cleaning, Classification, Feature Scaling, Model Comparison.

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np


uci_data = pd.read_csv('./../../data/heart_cleveland_upload.csv')
print(uci_data.head(10))
# print(uci_data.shape)
# print(uci_data.columns)
# print(uci_data.info())
# print(uci_data.describe())
# print(uci_data.isnull().sum())
# print(uci_data['dataset'].unique())
# categorical_cols = [cname for cname in uci_data.columns if uci_data[cname].dtype == "object"]
# print(categorical_cols)
# numerical_cols = [cname for cname in uci_data.columns if uci_data[cname].dtype in ['int64','float64']]
# print(numerical_cols)

'''
Target variable distribution
'''
# Rename the target column to 'target' for consistency
uci_data = uci_data.rename(columns={'condition': 'target'})

'''
Split features and target
'''
y = uci_data["target"]
X = uci_data.drop("target", axis=1)

'''
SPLIT TRAIN/VALIDATION DATA 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


'''
Feature scaling
Scale numeric features since the only existing ones
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


'''
Feature Engineering
Here, we focus on selecting and transforming features that may improve model performance.
For simplicity, we’ll keep the features as-is but could further explore:
    -Polynomial features
    -Domain-driven ratios (e.g., cholesterol/BP)

Our target is discrete(categorical : 0/1) thus use mutual_info_classif
'''
print("Number of features:", X_train.shape[1])
from sklearn.feature_selection import mutual_info_classif
discrete_features = X.dtypes == 'int64'
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores) # show features with their MI scores


# Now a bar plot to make comparisons
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()