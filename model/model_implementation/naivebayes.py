# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:28:06 2024

@author: mbnas
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_curve,
    RocCurveDisplay,
    accuracy_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
# Balancing the Training Dataset with Undersampling
from imblearn.under_sampling import RandomUnderSampler
# Balancing the Training Dataset
from imblearn.over_sampling import SMOTE

#%% 
# Define an identity tokenizer for pre-tokenized input
def identity_tokenizer(text):
    return text

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    tokenizer=identity_tokenizer,
    lowercase=False,
    stop_words=None
)

#%%
# Step 1: Load the datasets
train_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_train_en.tsv", sep='\t')
dev_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_en.tsv", sep='\t')
dev_test_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_data = pd.read_csv("C:/Users/mbnas/.spyder-py3/AI-project/model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

# Assuming datasets have 'sentence_id', 'sentence', and 'label' columns
train_texts = train_data['sentence']
train_labels = train_data['label']
dev_texts = dev_data['sentence']
dev_labels = dev_data['label']
dev_test_texts = dev_test_data['sentence']
dev_test_labels = dev_test_data['label']
test_texts = test_data['sentence']
test_labels = test_data['label']

#%%
# Step 2: Feature extraction using TF-IDF
X_train_tfidf = tfidf.fit_transform(train_texts)
X_dev_tfidf = tfidf.transform(dev_texts)
X_dev_test_tfidf = tfidf.transform(dev_test_texts)
X_test_tfidf = tfidf.transform(test_texts)

#%%
# Perform SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_tfidf_balanced, train_labels_balanced = smote.fit_resample(X_train_tfidf, train_labels)
# Perform undersampling on the majority class
undersampler = RandomUnderSampler(random_state=42)
X_train_tfidf_balanced, train_labels_balanced = undersampler.fit_resample(X_train_tfidf, train_labels)


#%%
# Step 3: Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf_balanced, train_labels_balanced)

#%%
# Step 4: Hyperparameter tuning using GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_dev_tfidf, dev_labels)
best_model = grid_search.best_estimator_

#%%
# Step 5: Fine-tuning the model
best_model.fit(X_dev_test_tfidf, dev_test_labels)

#%%
# Step 6: Evaluate the model on the development-test dataset
dev_test_pred = best_model.predict(X_dev_test_tfidf)
print("Development-Test Evaluation:")
print(classification_report(dev_test_labels, dev_test_pred))

#%%
# Step 7: Final evaluation on the test dataset
test_pred = best_model.predict(X_test_tfidf)
print("Final Test Evaluation:")
print(classification_report(test_labels, test_pred))

#%%
# Confusion Matrix
conf_matrix_test = confusion_matrix(test_labels, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test)
disp.plot()
plt.title("Confusion Matrix - Test Set")
plt.show()

#%%
# Step 8: ROC Curve
y_test_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = roc_curve(test_labels, y_test_proba, pos_label=best_model.classes_[1])
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_disp.plot()
plt.title("ROC Curve - Test Set")
plt.show()

#%%
# Step 9: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_labels, y_test_proba, pos_label=best_model.classes_[1])
pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_disp.plot()
plt.title("Precision-Recall Curve - Test Set")
plt.show()

#%%
# Step 10: Save final predictions
output = pd.DataFrame({'sentence_id': test_data['sentence_id'], 'label': test_pred})
output.to_csv("C:/Users/mbnas/.spyder-py3/AI-project/model_outputs/final_predictions_nb.tsv", sep='\t', index=False)

