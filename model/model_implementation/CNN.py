import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Step 1: Load the training data
train_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_train_en.tsv"  # Replace with your training file path
test_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_test_en_gold.tsv"    # Replace with your testing file path

# Load and preprocess training data
train_df = pd.read_csv(train_file_path, sep="\t")
train_df['label'] = train_df['label'].map({'SUBJ': 1, 'OBJ': 0})
train_df['sentence'] = train_df['sentence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Load and preprocess testing data
test_df = pd.read_csv(test_file_path, sep="\t")
test_df['label'] = test_df['label'].map({'SUBJ': 1, 'OBJ': 0})
test_df['sentence'] = test_df['sentence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Step 2: Create a tokenizer vocabulary from the training data
tokenizer_vocab = {word: idx + 1 for idx, word in enumerate(set(token for tokens in train_df['sentence'] for token in tokens))}

# Convert sentences to sequences using the tokenizer vocabulary
train_df['sentence_seq'] = train_df['sentence'].apply(lambda tokens: [tokenizer_vocab.get(token, 0) for token in tokens])
test_df['sentence_seq'] = test_df['sentence'].apply(lambda tokens: [tokenizer_vocab.get(token, 0) for token in tokens])

# Pad the sequences
max_len = 50  # Adjust based on the longest tokenized sentence
X_train = pad_sequences(train_df['sentence_seq'], maxlen=max_len, padding='post')
X_test = pad_sequences(test_df['sentence_seq'], maxlen=max_len, padding='post')

# Extract labels
y_train = train_df['label']
y_test = test_df['label']

# Step 3: Build the CNN model
embedding_dim = 50  # Size of the word embedding vectors

model = Sequential([
    Embedding(input_dim=len(tokenizer_vocab) + 1, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (SUBJ vs OBJ)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Evaluate on the test dataset
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['OBJ', 'SUBJ']))

# Step 6: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['OBJ', 'SUBJ'], yticklabels=['OBJ', 'SUBJ'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
