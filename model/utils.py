import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Step 1: Load the Data
filepath = r"C:\Users\welde\Documents\GitHub\AI-project\subjectivity_determiner\data\subtask-2-english\train_en.tsv"

#Load the .tsv file into a pandas DataFrame.

df = pd.read_csv(filepath, sep='\t')
print(f"Loaded data from {filepath} with shape {df.shape}")
print(df.describe())
#Convert the text column to lowercase."""
#df[text_column] = df[text_column].str.lower()
