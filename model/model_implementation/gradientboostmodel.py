# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:52:24 2024

@author: mbnas
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
#%% Load the datasets (ensure correct file paths)
train_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

#%% Prepare the data: Extract features and labels
X_train = train_df['sentence']  
y_train = train_df['label']
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

#%% Feature extraction with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_dev_test_tfidf = vectorizer.transform(X_dev_test)
X_test_tfidf = vectorizer.transform(X_test)

#%% Initialize the Gradient Boosting classifier
gbm_model = GradientBoostingClassifier(random_state=42)

#%% Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'learning_rate': [0.05, 0.1],  # Step size shrinking
    'max_depth': [3, 5, 7],  # Depth of the tree
    'min_samples_split': [2, 5],  # Minimum samples required to split a node
    'subsample': [0.8, 1.0],  # Fraction of samples used for training
}

grid_search = GridSearchCV(estimator=gbm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

#%% Get the best parameters
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

#%% Use the best estimator from grid search
best_gbm_model = grid_search.best_estimator_

#%% Evaluate on the development set
y_dev_pred = best_gbm_model.predict(X_dev_tfidf)
print("Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred, target_names=label_encoder.classes_))

#%% Fine-tune and evaluate on the development-test set
best_gbm_model.fit(X_dev_test_tfidf, y_dev_test)
y_dev_test_pred = best_gbm_model.predict(X_dev_test_tfidf)
print("Development-Test Set Evaluation:")
print(classification_report(y_dev_test, y_dev_test_pred, target_names=label_encoder.classes_))

#%% Final evaluation on the test set
y_test_pred = best_gbm_model.predict(X_test_tfidf)
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

#%% Save the final predictions on the test dataset
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
output = pd.DataFrame({'sentence_id': test_df['sentence_id'], 'label': y_test_pred_labels})
output.to_csv("C:/Users/mbnas/.spyder-py3/AI-project/model_outputs/final_predictions_gb.tsv", sep='\t', index=False)

#%% Confusion Matrix Visualization
# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix with appropriate labels
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_).plot()

# Show the plot
plt.show()


#%% ROC Curve and AUC for the test set
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, best_gbm_model.predict_proba(X_test_tfidf)[:, 1])
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
