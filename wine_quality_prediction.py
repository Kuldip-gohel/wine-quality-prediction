import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# 1.load dataset
wine = pd.read_csv("wine_quality_merged.csv")

# 2.split data based on quality
x = wine.drop("quality", axis=1)
y = wine["quality"]

split = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
for test_index, train_index in split.split(x, y):
    train_set = wine.loc[train_index]
    test_set = wine.loc[test_index]

wine = test_set.copy()

# 3.now clean train dataset
# handle catagorical data
wine["type"] = wine["type"].map({"red": 0, "white": 1})

# 4.seprate feature and label
wine_label = wine["quality"].copy()
wine = wine.drop("quality", axis=1)

# 5.seprate numerical and catagorical column
num_attribs = wine.drop("type", axis=1).columns.tolist()
cat_attribs = ["type"]

# 6.now create pipline
# Numerical pipeline: handle missing, scale values
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 7.Transform the training data
wine_prepared = full_pipeline.fit_transform(wine)



# # Train Logistic Regression model
# log_reg_clf = LogisticRegression(random_state=42)
# log_reg_clf.fit(wine_prepared, wine_label)

# # Predictions on training set
# lr_preds = log_reg_clf.predict(wine_prepared)

# # Training accuracy
# train_acc = accuracy_score(wine_label, lr_preds)
# print(f"Training Accuracy: {train_acc:.4f}")

# # Cross-validation accuracy
# lr_cv_scores = cross_val_score(log_reg_clf, wine_prepared, wine_label, cv=10, scoring="accuracy")
# print("\nCross-validation accuracy stats:")
# print(pd.Series(lr_cv_scores).describe())

# # Classification report
# print("\nClassification Report:")
# print(classification_report(wine_label, lr_preds))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(wine_label, lr_preds))

# -----------------------------------------------------------------------

# # 10. Train Decision Tree Classifier
# decision_tree_clf = DecisionTreeClassifier(random_state=42)
# decision_tree_clf.fit(wine_prepared, wine_label)

# # Predictions on training set
# dt_preds = decision_tree_clf.predict(wine_prepared)

# # Training accuracy
# train_acc = accuracy_score(wine_label, dt_preds)
# print(f"Training Accuracy: {train_acc:.4f}")

# # Cross-validation accuracy
# dt_cv_scores = cross_val_score(decision_tree_clf, wine_prepared, wine_label, cv=10, scoring="accuracy")
# print("\nCross-validation accuracy stats:")
# print(pd.Series(dt_cv_scores).describe())

# # Classification report
# print("\nClassification Report:")
# print(classification_report(wine_label, dt_preds))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(wine_label, dt_preds))

# ---------------------------------------------------------------------------------------------------------

# # 10. Train Random Forest Classifier
random_forest_clf = RandomForestClassifier(random_state=42)
random_forest_clf.fit(wine_prepared, wine_label)

# Predictions on training set
rf_preds = random_forest_clf.predict(wine_prepared)

# Training accuracy
train_acc = accuracy_score(wine_label, rf_preds)
print(f"Training Accuracy: {train_acc:.4f}")

# Cross-validation accuracy
rf_cv_scores = cross_val_score(random_forest_clf, wine_prepared, wine_label, cv=10, scoring="accuracy")
print("\nCross-validation accuracy stats:")
print(pd.Series(rf_cv_scores).describe())

# Classification report
print("\nClassification Report:")
print(classification_report(wine_label, rf_preds))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(wine_label, rf_preds))

# ---------------------------------------------------------------------------------------------------------






# 10. Train Random Forest Classifier on TRAIN DATA
# (changed from test_set to train_set above)

wine = train_set.copy()   # ✅ Use train set instead of test set

# handle categorical data
wine["type"] = wine["type"].map({"red": 0, "white": 1})

# separate feature and label
wine_label = wine["quality"].copy()
wine = wine.drop("quality", axis=1)

# separate numerical and categorical column
num_attribs = wine.drop("type", axis=1).columns.tolist()
cat_attribs = ["type"]

# pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# transform training data
wine_prepared = full_pipeline.fit_transform(wine)

# Train Random Forest
random_forest_clf = RandomForestClassifier(random_state=42)
random_forest_clf.fit(wine_prepared, wine_label)

# Predictions on training set
rf_preds = random_forest_clf.predict(wine_prepared)

# Training accuracy
train_acc = accuracy_score(wine_label, rf_preds)
print(f"Training Accuracy: {train_acc:.4f}")

# Cross-validation accuracy
rf_cv_scores = cross_val_score(random_forest_clf, wine_prepared, wine_label, cv=10, scoring="accuracy")
print("\nCross-validation accuracy stats:")
print(pd.Series(rf_cv_scores).describe())

# Classification report
print("\nClassification Report (Train Data):")
print(classification_report(wine_label, rf_preds))

# Confusion matrix
print("\nConfusion Matrix (Train Data):")
print(confusion_matrix(wine_label, rf_preds))




