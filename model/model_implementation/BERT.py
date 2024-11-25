from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the datasets with untokenized sentences
train_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\subjectivity_determiner\data\subtask-2-english\train_en.tsv"  # Replace with your training file path
test_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\subjectivity_determiner\data\subtask-2-english\test_en_gold.tsv"    # Replace with your testing file path

# Load the data
train_df = pd.read_csv(train_file_path, sep="\t")
test_df = pd.read_csv(test_file_path, sep="\t")

# Map labels to binary values
train_df['label'] = train_df['label'].map({'SUBJ': 1, 'OBJ': 0})
test_df['label'] = test_df['label'].map({'SUBJ': 1, 'OBJ': 0})

# Step 2: Load BERT tokenizer and model
model_name = "bert-base-uncased"  # Replace with another BERT model if needed
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 3: Tokenize the data
max_length = 50  # Maximum sequence length for BERT input

def tokenize_data(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

train_dataset = tokenize_data(train_df['sentence'], train_df['label'], tokenizer, max_length).shuffle(len(train_df)).batch(32)
test_dataset = tokenize_data(test_df['sentence'], test_df['label'], tokenizer, max_length).batch(32)

# Step 4: Create the optimizer using Transformers' create_optimizer
num_train_steps = len(train_dataset) * 3  # 3 is the number of epochs
num_warmup_steps = int(0.1 * num_train_steps)  # 10% of total steps as warmup

optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    weight_decay_rate=0.01
)


# Compile the model
model.compile(optimizer=optimizer, metrics=['accuracy'])

# Step 5: Train the BERT model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Step 6: Evaluate the model
y_pred_probs = model.predict(test_dataset).logits
y_pred = tf.argmax(y_pred_probs, axis=1).numpy()

# Classification report
print("Classification Report:\n", classification_report(test_df['label'], y_pred, target_names=['OBJ', 'SUBJ']))

# Step 7: Plot confusion matrix
cm = confusion_matrix(test_df['label'], y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['OBJ', 'SUBJ'], yticklabels=['OBJ', 'SUBJ'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
