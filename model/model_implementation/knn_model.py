# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:20:07 2024

@author: mbnas
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # For balancing the dataset

# Step 1: Load the datasets (ensure correct file paths)
train_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

#%% Prepare the data: Extract features and labels
X_train = train_df['sentence']  # The text data (sentences)
y_train = train_df['label']  # The target column (labels)

X_dev = dev_df['sentence']
y_dev = dev_df['label']

X_dev_test = dev_test_df['sentence']
y_dev_test = dev_test_df['label']

X_test = test_df['sentence']
y_test = test_df['label']

#%% Encode the labels into numeric format (if necessary)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_dev = label_encoder.transform(y_dev)
y_dev_test = label_encoder.transform(y_dev_test)
y_test = label_encoder.transform(y_test)

#%% Step 2: Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_dev_test_tfidf = vectorizer.transform(X_dev_test)
X_test_tfidf = vectorizer.transform(X_test)

# Balance the training dataset using SMOTE (for class imbalance)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

#%% Step 3: Initialize the KNN classifier
knn_model = KNeighborsClassifier()

#%% Step 4: Enhanced grid search with KNN-specific hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 7, 11],  # Number of neighbors to use
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'p': [1, 2],  # Power parameter for the Minkowski distance metric (1 for Manhattan, 2 for Euclidean)
}

grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model using GridSearchCV
grid_search.fit(X_train_tfidf, y_train)

# Get the best parameters from the grid search
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Use the best estimator found by GridSearchCV
best_knn_model = grid_search.best_estimator_

#%% Step 5: Evaluate on the development set
y_dev_pred = best_knn_model.predict(X_dev_tfidf)
print("Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred, target_names=label_encoder.classes_))

# Confusion matrix
print("Confusion Matrix on Development Set:")
print(confusion_matrix(y_dev, y_dev_pred))

#%% Step 6: Fine-tune and evaluate on the development-test set
best_knn_model.fit(X_dev_test_tfidf, y_dev_test)
y_dev_test_pred = best_knn_model.predict(X_dev_test_tfidf)
print("Development-Test Set Evaluation:")
print(classification_report(y_dev_test, y_dev_test_pred, target_names=label_encoder.classes_))

# Confusion matrix
print("Confusion Matrix on Development-Test Set:")
print(confusion_matrix(y_dev_test, y_dev_test_pred))

#%% Step 7: Final evaluation on the test set
y_test_pred = best_knn_model.predict(X_test_tfidf)
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# Confusion matrix
print("Confusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_test_pred))

#%% Step 8: Plot ROC Curve for the test set
fpr, tpr, thresholds = roc_curve(y_test, best_knn_model.predict_proba(X_test_tfidf)[:, 1])
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.2f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# After making predictions on the test set, convert numeric labels back to class names
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

#%% Step 9: Save the final predictions on the test dataset in the required TSV format
output = pd.DataFrame({'sentence_id': test_df['sentence_id'], 'label': y_test_pred_labels})
output.to_csv("C:/Users/mbnas/.spyder-py3/AI-project/model_outputs/final_predictions_knn.tsv", sep='\t', index=False)
