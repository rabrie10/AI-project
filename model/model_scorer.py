# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:12:54 2024

@author: mbnas
"""

import pandas as pd
from sklearn.metrics import (
classification_report, 
confusion_matrix, 
precision_recall_curve, 
precision_recall_fscore_support, 
roc_curve, 
roc_auc_score, 
RocCurveDisplay,
ConfusionMatrixDisplay,
PrecisionRecallDisplay,
matthews_corrcoef
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def evaluate_model(test_data_file, prediction_file, output_dir="evaluation_output"):
    # Load the testing data and model predictions
    test_df = pd.read_csv(test_data_file, sep='\t')
    pred_df = pd.read_csv(prediction_file, sep='\t')

    # Merge the true labels and predicted labels based on 'sentence_id'
    merged_df = pd.merge(test_df[['sentence_id', 'label']], pred_df[['sentence_id', 'label']], on='sentence_id', suffixes=('_true', '_pred'))

    # Extract the true and predicted labels
    y_true = merged_df['label_true']
    y_pred = merged_df['label_pred']

    # Encode labels (if they are categorical, e.g., 'OBJ' and 'SUBJ')
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    #matthews correlation coefficient
    mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Plot Precision-Recall Curves for both classes (OBJ and SUBJ)
    for i, label in enumerate(label_encoder.classes_):
        precision_values, recall_values, _ = precision_recall_curve(y_true_encoded, y_pred_encoded, pos_label=i)
        pr_display = PrecisionRecallDisplay(precision=precision_values, recall=recall_values)
        pr_display.plot()
        plt.title(f"Precision-Recall Curve for {label}")
        plt.savefig(f"{output_dir}/precision_recall_curve_{label}.png")
        plt.close()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_encoded)
    auc = roc_auc_score(y_true_encoded, y_pred_encoded)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Model")
    roc_display.plot()
    plt.title("ROC Curve")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    # Output AUC score
    print(f"AUC Score: {auc:.4f}")

    # Return final evaluation metrics in a dictionary
    return {
        "classification_report": classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_),
        "confusion_matrix": conf_matrix,
        "precision_recall_curve": {label: (precision_values, recall_values) for i, label in enumerate(label_encoder.classes_)},
        "roc_auc": auc
    }


# Usage
test_data_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/test_en_gold.tsv"  # Your testing data file path
prediction_file = "C:/Users/mbnas/.spyder-py3/final_predictions.tsv"  # Your model output file path

evaluation_results = evaluate_model(test_data_file, prediction_file, "C:/Users/mbnas/.spyder-py3")
