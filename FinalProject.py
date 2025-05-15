# Natalia Guzman
# DSC 540
# Final Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE 

df = pd.read_csv("train.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().any())
print(df.shape)

# Data preprocessing
# Distribution of numerical features

plt.figure(figsize=(15,5))

plt.subplot(1,5,1)
sns.histplot(x=df['Age'])

plt.subplot(1,5,2)
sns.histplot(x=df['Region_Code'])

plt.subplot(1,5,3)
sns.histplot(x=df['Annual_Premium'])

plt.subplot(1,5,4)
sns.histplot(x=df['Policy_Sales_Channel'])

plt.subplot(1,5,5)
sns.histplot(x=df['Vintage'])

plt.show()

# Distribution of categorical features

plt.figure(figsize=(15,6))

plt.subplot(2,3,1)
sns.countplot(x=df['Gender'])

plt.subplot(2,3,2)
sns.countplot(x=df['Driving_License'])

plt.subplot(2,3,3)
sns.countplot(x=df['Previously_Insured'])

plt.subplot(2,3,4)
sns.countplot(x=df['Vehicle_Age'])

plt.subplot(2,3,5)
sns.countplot(x=df['Vehicle_Damage'])

plt.subplot(2,3,6)
sns.countplot(x=df['Response'])

plt.show()

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Gender', 'Vehicle_Age', 'Vehicle_Damage'], drop_first=True)

# Define Features and Target and Scale
X = df_encoded.drop(['id', 'Response'], axis=1)
y = df_encoded['Response']
scaler = MinMaxScaler()
X[['Age', 'Annual_Premium', 'Vintage', 'Policy_Sales_Channel','Region_Code']] = scaler.fit_transform(X[['Age', 'Annual_Premium', 'Vintage', 'Policy_Sales_Channel','Region_Code']])

# SMOTE for imbalanced target

oversample = SMOTE(random_state=42)
balanced_X, balanced_y = oversample.fit_resample(X, y)
print(balanced_y.value_counts())

# Modeling
# Random Forest + feature selection

clf = RandomForestClassifier(random_state= 42)             
sel = SelectFromModel(clf, threshold='mean', max_features=None)                   
print ('Wrapper Select: ')

selected_X = sel.fit_transform(balanced_X, balanced_y)

selected_features = balanced_X.columns[sel.get_support()]
print("Selected features:", selected_features)

data_train, data_test, target_train, target_test = train_test_split(balanced_X, balanced_y, test_size=0.35)

scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
scores = cross_validate(clf, data_train, target_train, scoring=scorers, cv=5)
scores_Acc = scores['test_Accuracy']
print("Random Forest Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                         #Only works with binary classes, not multiclass
scores_AUC= scores['test_roc_auc']
print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))

# Prediction and Performance metrics

clf.fit(data_train, target_train)
target_pred = clf.predict(data_test)

cm = confusion_matrix(target_test, target_pred, labels=clf.classes_)
ConfusionMatrixDisplay.from_predictions(y_pred=target_pred, y_true=target_test, display_labels=clf.classes_, cmap='Blues')
plt.show()

# Gradient boosting + feature selection

clf_2 = GradientBoostingClassifier(random_state= 42)             
sel = SelectFromModel(clf_2, threshold='mean', max_features=None)                   
print ('Wrapper Select: ')

selected_X = sel.fit_transform(balanced_X, balanced_y)

selected_features = balanced_X.columns[sel.get_support()]
print("Selected features:", selected_features)

scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
scores = cross_validate(clf_2, data_train, target_train, scoring=scorers, cv=5)
scores_Acc = scores['test_Accuracy']
print("Gradient Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                         #Only works with binary classes, not multiclass
scores_AUC= scores['test_roc_auc']
print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))

# Prediction and Performance metrics

clf_2.fit(data_train, target_train)
target_pred = clf_2.predict(data_test)

cm = confusion_matrix(target_test, target_pred, labels=clf_2.classes_)
ConfusionMatrixDisplay.from_predictions(y_pred=target_pred, y_true=target_test, display_labels=clf_2.classes_, cmap='Blues')
plt.show()
