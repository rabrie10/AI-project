import pandas as pd
import re

# Step 1: Load the Data
filepath = r"C:\Users\welde\Documents\GitHub\AI-project\model_outputs\RNN_prediction.tsv"

# Load the .tsv file into a pandas DataFrame
df = pd.read_csv(filepath, sep='\t')

# Step 2: Ensure 'label' is numeric
# If 'label' is textual, map it to numeric first
if df['label'].dtype == 'object':  # Check if labels are strings
    df['label'] = df['label'].map({'OBJ': 0, 'SUBJ': 1})

# Step 3: Map numeric values back to strings (if needed)
df['label'] = df['label'].map({0: 'OBJ', 1: 'SUBJ'})

# Step 4: Save the updated DataFrame back to a file
output_path = r"C:\Users\welde\Documents\GitHub\AI-project\model_outputs\RNN_prediction.tsv"  # Replace with the desired output file name
df.to_csv(output_path, sep="\t", index=False)

print("File updated and saved as:", output_path)
