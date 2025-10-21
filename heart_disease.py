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


uci_data = pd.read_csv('./../data/heart_cleveland_upload.csv')
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
sns.countplot(data=uci_data,x='target',palette='Set2')
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")

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
'''
print("Number of features:", X_train.shape[1])

'''
MODEL TRAINING AND EVALUATION
'''
#LOGISTIC REGRESSION
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:,1]

# SVM
svm_clf = SVC(kernel='rbf',probability=True)
svm_clf.fit(X_train_scaled,y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)
y_prob_svm = svm_clf.predict_proba(X_test_scaled)[:,1]

# Random Forest is non-linear so no need for scaling
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


'''
Comparing their performance on accuracy, ROC-AUC, and classification report.
'''
print("🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))


print('_'*30)
print("🔹 SVM Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

print('_'*30)
print("🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
'''

EDA : Plot ROC Curves
''' 
plt.figure(figsize=(6,5))
RocCurveDisplay.from_estimator(log_reg, X_test_scaled, y_test, name="Logistic Regression")
RocCurveDisplay.from_estimator(svm_clf, X_test_scaled, y_test, name="Support Vector Machine")
RocCurveDisplay.from_estimator(rf, X_test, y_test, name="Random Forest")
plt.title("ROC Curves Comparison")


'''
FEATURE IMPORTANCE
LR derives it from the magnitude of coefficients (coef_) — larger absolute values indicate stronger influence on the prediction.
Interpretation:
Positive coefficient → increases probability of class 1
Negative coefficient → decreases probability of class 1
Magnitude → how strongly it affects the prediction
'''
coefficients = log_reg.coef_[0]
feat_imp_lr = pd.Series(coefficients, index=X_train.columns).sort_values(key=np.abs, ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp_lr.values, y=feat_imp_lr.index, palette='coolwarm',hue=feat_imp_lr.index,legend=False)
plt.title("Feature Importance (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")

'''
Random Forest exposes it i.e. rf.feature_importances
'''
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp, y=feat_imp.index, palette='Set2',hue=feat_imp.index,legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")

'''
SVM has limited interpretability
If you use an RBF kernel, there are no direct coefficients — feature relationships are nonlinear and mapped into higher dimensions.
In that case, you need model-agnostic explainers, such as:
    -Permutation Importance (sklearn.inspection.permutation_importance)
    -SHAP (SHapley Additive exPlanations) — for detailed interpretability


'''
# Permutation Importance for RBF SVM
from sklearn.inspection import permutation_importance
result = permutation_importance(svm_clf, X_test_scaled, y_test, n_repeats=10, random_state=42)
feat_imp_svm = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp_svm.values, y=feat_imp_svm.index, palette='viridis',hue=feat_imp_svm.index,legend=False)
plt.title("Permutation Importance (SVM - RBF Kernel)")
plt.xlabel("Mean Importance")
plt.ylabel("Feature")
plt.show()


'''
SUMMARY
Model	How to Get Importance	Notes
Random Forest	model.feature_importances_	Based on splits / Gini importance
Logistic Regression	model.coef_	Direction (sign) and magnitude matter
SVM (Linear)	model.coef_	Similar to logistic regression
SVM (RBF / Nonlinear)	permutation_importance() or SHAP	No direct coefficients
'''

'''
Future work could explore:
Hyperparameter tuning with GridSearchCV
Using ensemble methods (e.g., XGBoost)
Model interpretability tools (e.g., SHAP, LIME)
'''