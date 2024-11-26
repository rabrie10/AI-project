import pandas as pd
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, create_optimizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Preprocess the Data
train_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_train_en.tsv"
test_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_test_en_gold.tsv"

train_df = pd.read_csv(train_file_path, sep="\t")
test_df = pd.read_csv(test_file_path, sep="\t")

# Map labels to binary values
train_df['label'] = train_df['label'].map({'SUBJ': 1, 'OBJ': 0})
test_df['label'] = test_df['label'].map({'SUBJ': 1, 'OBJ': 0})

# Drop NaN values and ensure sentences are strings
train_df = train_df.dropna(subset=['sentence'])
test_df = test_df.dropna(subset=['sentence'])
train_df['sentence'] = train_df['sentence'].astype(str)
test_df['sentence'] = test_df['sentence'].astype(str)

# Debugging: Check label distribution
print("Train Labels Distribution:\n", train_df['label'].value_counts())
print("Test Labels Distribution:\n", test_df['label'].value_counts())

# Step 2: Load RoBERTa Tokenizer and Model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = TFRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 3: Tokenize Data
max_length = 50

def tokenize_data(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    labels = tf.convert_to_tensor(labels.values, dtype=tf.int32)  # Ensure labels are tensors
    print("Sample Labels:", labels.numpy())  # Debugging
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

train_dataset = tokenize_data(train_df['sentence'], train_df['label'], tokenizer, max_length).shuffle(len(train_df)).batch(32)
test_dataset = tokenize_data(test_df['sentence'], test_df['label'], tokenizer, max_length).batch(32)

# Step 4: Define Optimizer and Compile the Model
num_train_steps = len(train_dataset) * 3  # 3 epochs
num_warmup_steps = int(0.1 * num_train_steps)

optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    weight_decay_rate=0.01
)

model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Step 6: Evaluate the Model
y_pred_probs = model.predict(test_dataset).logits
y_pred = tf.argmax(y_pred_probs, axis=1).numpy()
y_true = test_df['label'].values

# Classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['OBJ', 'SUBJ']))

# Step 7: Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['OBJ', 'SUBJ'], yticklabels=['OBJ', 'SUBJ'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
