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
    matthews_corrcoef,
    roc_curve, 
    roc_auc_score, 
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def save_text_as_image(text, output_path):
    """Save a block of text as an image using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
    ax.text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True, transform=ax.transAxes)
    ax.axis('off')  # Hide axes
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def evaluate_model(test_data_file, prediction_file, output_dir="evaluation_output", model_name="model_name"):
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

    # Generate Classification Report
    class_report = classification_report(y_true_encoded, y_pred_encoded, target_names=label_encoder.classes_)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true_encoded, y_pred_encoded)

    # ROC Curve and AUC
    auc = roc_auc_score(y_true_encoded, y_pred_encoded)

    # Combine classification report, MCC, and AUC into one string
    combined_text = (
        "Classification Report:\n"
        f"{class_report}\n"
        f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n"
        f"AUC Score: {auc:.4f}"
    )
    print(combined_text)

    # Save the combined text as an image
    save_text_as_image(combined_text, f"{output_dir}/classification_report_with_mcc_auc_{model_name}.png")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.png")
    plt.close()
    
    # Plot Precision-Recall Curves for both classes (OBJ and SUBJ)
    for i, label in enumerate(label_encoder.classes_):
        precision_values, recall_values, _ = precision_recall_curve(y_true_encoded, y_pred_encoded, pos_label=i)
        pr_display = PrecisionRecallDisplay(precision=precision_values, recall=recall_values)
        pr_display.plot()
        plt.title(f"Precision-Recall Curve for {label}")
        plt.savefig(f"{output_dir}/precision_recall_curve_{label}_{model_name}.png")
        plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_encoded, y_pred_encoded)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name="Model")
    roc_display.plot()
    plt.title("ROC Curve")
    plt.savefig(f"{output_dir}/roc_curve_{model_name}.png")
    plt.close()

    # Return final evaluation metrics
    return {
        "classification_report": class_report,
        "mcc": mcc,
        "roc_auc": auc,
        "confusion_matrix": conf_matrix,
    }

# Usage
test_data_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/test_en_gold.tsv"  # Your testing data file path
prediction_file = "C:/Users/mbnas/.spyder-py3/AI-project/model_outputs/voting_predictions.tsv"  # Your model output file path

evaluation_results = evaluate_model(
    test_data_file, 
    prediction_file, 
    "C:/Users/mbnas/.spyder-py3/AI-project/evaluation_scores_data", 
    "voting"
)

