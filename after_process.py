import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


try:
    df = pd.read_csv(r"C:\Users\welde\Documents\GitHub\AI-project\processed_subjectivity_results.tsv", sep='\t')
    print("File loaded successfully!")
    print("Columns:", df.columns)
    print("Number of rows:", len(df))
except Exception as e:
    print("Error:", e)


# Step 1: Map Predicted Labels to Classes
# Ensure "LABEL_0" = "OBJ" and "LABEL_1" = "SUBJ"
df['predicted_label'] = df['predicted_label'].replace({'LABEL_0': 'OBJ', 'LABEL_1': 'SUBJ'})

# Step 2: Generate the Confusion Matrix
true_labels = df['label']  # Actual labels
predicted_labels = df['predicted_label']  # Predicted labels

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['OBJ', 'SUBJ'])

# Step 3: Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['OBJ', 'SUBJ'], yticklabels=['OBJ', 'SUBJ'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 4: Print Classification Report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['OBJ', 'SUBJ']))

df.top()