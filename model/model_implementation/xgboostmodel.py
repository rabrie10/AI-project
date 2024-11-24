# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:02:48 2024

@author: mbnas
"""
###66%accuracy
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt

# Load the datasets (ensure correct file paths)
train_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

#%%
# Prepare the data: Extract features and labels
X_train = train_df['sentence']  
y_train = train_df['label']

X_dev = dev_df['sentence']
y_dev = dev_df['label']

X_dev_test = dev_test_df['sentence']
y_dev_test = dev_test_df['label']

X_test = test_df['sentence']
y_test = test_df['label']

#%%
# Encode the labels into numeric format (if necessary)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_dev = label_encoder.transform(y_dev)
y_dev_test = label_encoder.transform(y_dev_test)
y_test = label_encoder.transform(y_test)

#%%
# Feature extraction with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_dev_test_tfidf = vectorizer.transform(X_dev_test)
X_test_tfidf = vectorizer.transform(X_test)

#%%
# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False)

#%%
# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'learning_rate': [0.05, 0.1],  # Step size shrinking
    'max_depth': [3, 5, 7],  # Depth of the tree
    'min_child_weight': [1, 3],  # Minimum weight required for a child node
    'subsample': [0.8, 1.0],  # Fraction of samples used for training
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

#%%
# Get the best parameters
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

#%%
# Use the best estimator from grid search
best_xgb_model = grid_search.best_estimator_

#%%
# Evaluate on the development set
y_dev_pred = best_xgb_model.predict(X_dev_tfidf)
print("Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred, target_names=label_encoder.classes_))

#%%
# Fine-tune and evaluate on the development-test set
best_xgb_model.fit(X_dev_test_tfidf, y_dev_test)
y_dev_test_pred = best_xgb_model.predict(X_dev_test_tfidf)
print("Development-Test Set Evaluation:")
print(classification_report(y_dev_test, y_dev_test_pred, target_names=label_encoder.classes_))

#%%
# Final evaluation on the test set
y_test_pred = best_xgb_model.predict(X_test_tfidf)
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

#%%
# Confusion Matrix for the test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(conf_matrix_test, display_labels=label_encoder.classes_)
disp_test.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set")
plt.show()

#%%
# Save the final predictions on the test dataset
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
output = pd.DataFrame({'sentence_id': test_df['sentence_id'], 'label': y_test_pred_labels})
output.to_csv("C:/Users/mbnas/.spyder-py3/final_predictions.tsv", sep='\t', index=False)

