'''
Heart Disease Prediction EDA
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


uci_data = pd.read_csv('./../data/heart_cleveland_upload.csv')
# print(uci_data.head(10))
# print(uci_data.shape)
# print(uci_data.columns)
# print(uci_data.describe())
print(uci_data.info())


'''
EDA ANALYSIS
'''
# checking TARGET values and count
print(uci_data['condition'].unique())
print(uci_data['condition'].nunique())

'''
Summarize object-type columns so : O
Since none, nothing
'''
# print(uci_data.describe(include = "O"))

'''
Missing values : none of the columns
'''
print(uci_data.isnull().sum())

'''
Duplicates : NONE
'''
print(uci_data.duplicated().sum())

'''
Aggregation of specific columns
'''
print(uci_data["chol"].agg(["max","min","mean"]).to_frame())
# print(uci_data["chol"].agg(["max","min","mean"]))
print(uci_data["age"].agg(["max","min","mean"]).to_frame())
print(uci_data["trestbps"].agg(["max","min","mean"]).to_frame())
print(uci_data["thalach"].agg(["max","min","mean"]).to_frame())
print(uci_data["oldpeak"].agg(["max","min","mean"]).to_frame())
print(uci_data["slope"].agg(["max","min","mean"]).to_frame())
print(uci_data["ca"].agg(["max","min","mean"]).to_frame())
print(uci_data["cp"].agg(["max","min","mean"]).to_frame())
print(uci_data["thal"].agg(["max","min","mean"]).to_frame())
print(uci_data["restecg"].agg(["max","min","mean"]).to_frame())
print(uci_data[uci_data["chol"] == 126.000000])


'''
Cholestrol distribution
'''
plt.figure(figsize=(6,4))
plt.hist(uci_data["chol"], bins=20)
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.show()


'''
Aggregate mean of fbs by age
'''
# print(uci_data['age'].unique())
# print(uci_data['age'].nunique())
print(uci_data.groupby("fbs")["age"].mean().to_frame())

'''
Aggregate mean of heart disease by chol
'''
print(uci_data.groupby("condition")["chol"].mean().to_frame())

'''
Break sex column and make Gender column where 0 : female and 1 : male
'''
print(uci_data['sex'].unique())
print(uci_data['sex'].nunique())
uci_data["Gender"] = uci_data["sex"].map({0:"female" , 1: "male"})
print(uci_data.head(10))
print(uci_data['Gender'].value_counts())

'''
Aggregate count heart disease by Gender and its distribution
crosstab creates a frequency table that shows how two categorical variables interact.
Rows → Sex
Columns → condition
Values → counts
'''
print(uci_data.groupby("condition")["Gender"].value_counts().to_frame())
gender_disease = pd.crosstab(uci_data["sex"], uci_data["condition"])

gender_disease.plot(kind="bar", figsize=(6,4))
plt.title("Heart Disease by Gender")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

'''
Heart disease distribution
'''
target_counts = uci_data['condition'].value_counts()

plt.figure(figsize=(5,4))
plt.bar(target_counts.index,target_counts.values)
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease (0 = No Disease, 1 = Disease)')
plt.ylabel('Count')
plt.show()

'''
Age distribution
'''
plt.figure(figsize=(6,4))
plt.hist(uci_data["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()

'''
Maximum Heart Rate distribution
'''
plt.figure(figsize=(6,4))
plt.hist(uci_data["thalach"], bins=20)
plt.title("Maximum Heart Rate distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Count")
plt.show()

