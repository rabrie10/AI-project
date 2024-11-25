import pandas as pd
from transformers import pipeline

# Step 1: Load the TSV File
file_path = r"C:\Users\welde\Documents\GitHub\AI-project\model\preprocessed_data\processed_test_en_gold.tsv"  # Replace with the actual file path
df = pd.read_csv(file_path, sep='\t')

# Step 2: Convert Tokenized Sentences to Plain Text
# Join tokenized words in 'sentence' column to form plain text sentences
df['sentence_text'] = df['sentence'].apply(lambda tokens: " ".join(eval(tokens)) if isinstance(tokens, str) else " ".join(tokens))

# Step 3: Load the Pre-Trained Transformer Model
classifier = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english")

# Step 4: Apply the Model to Classify Subjectivity
df['predicted_label'] = df['sentence_text'].apply(lambda x: classifier(x)[0]['label'])
df['confidence_score'] = df['sentence_text'].apply(lambda x: classifier(x)[0]['score'])

# Step 5: Save Results (Optional)
output_path = "processed_subjectivity_results.tsv"
df.to_csv(output_path, sep='\t', index=False)

# Print the DataFrame with Predictions
print(df[['sentence_id', 'sentence_text', 'label', 'predicted_label', 'confidence_score']].head())
