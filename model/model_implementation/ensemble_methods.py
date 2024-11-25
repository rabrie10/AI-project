# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:42:25 2024

@author: mbnas
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier

# Load data
train_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_df = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

# Prepare data
X_train = train_df['sentence']
y_train = train_df['label']
X_dev = dev_df['sentence']
y_dev = dev_df['label']
X_test = test_df['sentence']
y_test = test_df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTEENN to balance the training data
smoteenn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smoteenn.fit_resample(X_train_tfidf, y_train)
#%%
# Define parameter grid for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid Search for RandomForest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring='f1_weighted')
grid_search_rf.fit(X_train_res, y_train_res)
best_rf = grid_search_rf.best_estimator_
#%%
# Define parameter grid for GradientBoosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Grid Search for GradientBoosting
grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid_gb, cv=5, scoring='f1_weighted')
grid_search_gb.fit(X_train_res, y_train_res)
best_gb = grid_search_gb.best_estimator_
#%%
# Define base models with tuned hyperparameters and add SVM
base_models = [
    ('rf', best_rf),
    ('gb', best_gb),
    ('knn', Pipeline([('scaler', StandardScaler(with_mean=False)), 
                      ('knn', KNeighborsClassifier(n_neighbors=5))])),
    ('svm', SVC(probability=True, random_state=42))
]

# Define meta-learner for stacking
meta_learner = XGBClassifier(scale_pos_weight=len(y_train_res[y_train_res == 'OBJ']) / len(y_train_res[y_train_res == 'SUBJ']),
                             n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

#%%
# Create stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5)

# Bagging with Random Forest using the best estimator from Grid Search
bagging_clf = BaggingClassifier(estimator=meta_learner, n_estimators=10, random_state=42)

# Boosting with AdaBoost using the best estimator from Grid Search
boosting_clf = AdaBoostClassifier(estimator=RandomForestClassifier(class_weight={'OBJ': 1, 'SUBJ': 3}),
                                  n_estimators=50, random_state=42)

# Voting Classifier
voting_clf = VotingClassifier(estimators=base_models, voting='soft', weights=[2, 1, 1, 1])

#%%
# Train and evaluate each ensemble method
ensemble_methods = {
    'Stacking': stacking_clf,
    'Bagging': bagging_clf,
    'Boosting': boosting_clf,
    'Voting': voting_clf
}

for name, clf in ensemble_methods.items():
    print(f"\nTraining {name} model...")
    clf.fit(X_train_res, y_train_res)
    
    print(f"\nEvaluating {name} model on test set...")
    y_pred_test = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred_test))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    
    # Save predictions to file
    output_test = pd.DataFrame({
        'sentence_id': test_df['sentence_id'],  # Ensure 'sentence_id' exists in your test_df
        'label': y_pred_test
    })
    output_file = f"C:/Users/mbnas/.spyder-py3/{name.lower()}_predictions.tsv"
    output_test.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions for {name} saved to {output_file}")
    