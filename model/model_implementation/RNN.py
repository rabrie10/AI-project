from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import ast

# Step 1: Load and preprocess the training data
train_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_train_en.tsv"
test_file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_test_en_gold.tsv"

# Load and preprocess training data
train_df = pd.read_csv(train_file_path, sep="\t")
train_df['label'] = train_df['label'].map({'SUBJ': 1, 'OBJ': 0})
train_df['sentence'] = train_df['sentence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Load and preprocess testing data
test_df = pd.read_csv(test_file_path, sep="\t")
test_df['label'] = test_df['label'].map({'SUBJ': 1, 'OBJ': 0})
test_df['sentence'] = test_df['sentence'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Create a tokenizer vocabulary from the training data
tokenizer_vocab = {word: idx + 1 for idx, word in enumerate(set(token for tokens in train_df['sentence'] for token in tokens))}

# Convert sentences to sequences using the tokenizer vocabulary
train_df['sentence_seq'] = train_df['sentence'].apply(lambda tokens: [tokenizer_vocab.get(token, 0) for token in tokens])
test_df['sentence_seq'] = test_df['sentence'].apply(lambda tokens: [tokenizer_vocab.get(token, 0) for token in tokens])

# Pad the sequences
max_len = 50  # Maximum sequence length
X_train = pad_sequences(train_df['sentence_seq'], maxlen=max_len, padding='post')
X_test = pad_sequences(test_df['sentence_seq'], maxlen=max_len, padding='post')

# Extract labels
y_train = train_df['label']
y_test = test_df['label']

# Step 2: Build the adjusted RNN model
embedding_dim = 50  # Size of the word embedding vectors

model = Sequential([
    Embedding(input_dim=len(tokenizer_vocab) + 1, output_dim=embedding_dim, input_length=max_len),
    LSTM(256, return_sequences=True),  # First LSTM layer with return_sequences=True
    Dropout(0.5),
    LSTM(128, return_sequences=False),  # Second LSTM layer
    Dense(64, activation='relu'),  # Fully connected layer with 64 units
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification (SUBJ vs OBJ)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the model
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Step 4: Evaluate on the test dataset
y_pred = (model.predict(X_test) > 0.5).astype("int32")
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = y_pred.ravel()
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['OBJ', 'SUBJ']))
print(test_df['sentence_id'].shape)
print(test_df['sentence_id'].head())
print(y_pred.shape)
print(y_pred[:50])

# Step 5: Plot confusion matrix
output = pd.DataFrame({'sentence_id': test_df['sentence_id'], 'label': y_pred})
output.to_csv(r"C:\Users\welde\Documents\GitHub\AI-project\model_outputs\RNN_prediction.tsv", sep='\t', index=False)

