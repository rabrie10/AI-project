# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:25:45 2024

@author: mbnas
"""
#0.73 accuracy
#%% Import libraries
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB  # Import Naive Bayes
from imblearn.over_sampling import SMOTE  # For balancing the dataset

#%% Load datasets
train_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

#%% Prepare data
X_train, y_train = train_data['sentence'], train_data['label']
X_dev, y_dev = dev_data['sentence'], dev_data['label']
X_dev_test, y_dev_test = dev_test_data['sentence'], dev_test_data['label']
X_test, y_test = test_data['sentence'], test_data['label']

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_dev = label_encoder.transform(y_dev)
y_dev_test = label_encoder.transform(y_dev_test)
y_test = label_encoder.transform(y_test)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_dev_test_tfidf = vectorizer.transform(X_dev_test)
X_test_tfidf = vectorizer.transform(X_test)

#%% Balance the training dataset using SMOTE (for class imbalance)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

#%% Train Objective Model (XGBoost)
xgb_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=(len(y_train_balanced) / (2 * np.bincount(y_train_balanced)[1])))
param_grid_obj = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
}
grid_search_obj = GridSearchCV(estimator=xgb_model, param_grid=param_grid_obj, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search_obj.fit(X_train_balanced, y_train_balanced)
best_obj_model = grid_search_obj.best_estimator_

#%% Train Subjective Model (Naive Bayes)
nb_model = MultinomialNB()
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0],  # Regularization parameter for Naive Bayes
}
grid_search_nb = GridSearchCV(estimator=nb_model, param_grid=param_grid_nb, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search_nb.fit(X_train_balanced, y_train_balanced)
best_nb_model = grid_search_nb.best_estimator_

#%% Confidence-based Prediction Function
def confidence_based_prediction(obj_model, nb_model, X, threshold=0.6):
    obj_probs = obj_model.predict_proba(X)
    nb_probs = nb_model.predict_proba(X)
    
    obj_confidence = obj_probs.max(axis=1)
    nb_confidence = nb_probs.max(axis=1)
    
    obj_preds = obj_probs.argmax(axis=1)
    nb_preds = nb_probs.argmax(axis=1)
    
    final_preds = []
    for i in range(len(obj_confidence)):
        if obj_confidence[i] > threshold:
            final_preds.append(obj_preds[i])  # XGBoost for objective classification
        else:
            final_preds.append(nb_preds[i])  # Naive Bayes for subjective classification
    
    return final_preds

#%% Evaluate on Dev Set (for early validation)
y_dev_pred = confidence_based_prediction(best_obj_model, best_nb_model, X_dev_tfidf)

print(classification_report(y_test, y_dev_pred, target_names=label_encoder.classes_))
#%% Evaluate on Dev Test Set (final evaluation before Test)
y_dev_test_pred = confidence_based_prediction(best_obj_model, best_nb_model, X_dev_test_tfidf)
print(classification_report(y_test, y_dev_test_pred, target_names=label_encoder.classes_))

#%% Evaluate on Test Set and Save Results
y_test_pred = confidence_based_prediction(best_obj_model, best_nb_model, X_test_tfidf)

# Decode predictions and save results
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
output_test = pd.DataFrame({'sentence_id': test_data['sentence_id'], 'label': y_test_pred_labels})
output_test.to_csv("C:/Users/mbnas/.spyder-py3/final_predictions_with_nb_and_xgb.tsv", sep='\t', index=False)

#%% Metrics and Confusion Matrix for Test Set
print("Test Set Evaluation with XGBoost and Naive Bayes:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set with XGBoost and Naive Bayes")
plt.show()

#%% Metrics and Confusion Matrix for Dev Set (for early validation)
print("Dev Set Evaluation with XGBoost and Naive Bayes:")
print(classification_report(y_dev, y_dev_pred, target_names=label_encoder.classes_))

conf_matrix_dev = confusion_matrix(y_dev, y_dev_pred)
disp_dev = ConfusionMatrixDisplay(conf_matrix_dev, display_labels=label_encoder.classes_)
disp_dev.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Dev Set with XGBoost and Naive Bayes")
plt.show()

#%% Metrics and Confusion Matrix for Dev Test Set (final evaluation before Test)
print("Dev Test Set Evaluation with XGBoost and Naive Bayes:")
print(classification_report(y_dev_test, y_dev_test_pred, target_names=label_encoder.classes_))

conf_matrix_dev_test = confusion_matrix(y_dev_test, y_dev_test_pred)
disp_dev_test = ConfusionMatrixDisplay(conf_matrix_dev_test, display_labels=label_encoder.classes_)
disp_dev_test.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Dev Test Set with XGBoost and Naive Bayes")
plt.show()

