import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Step 1: Load the Data
filepath = "LB"
#Load the .tsv file into a pandas DataFrame.

df = pd.read_csv(file_path, sep='\t')
print(f"Loaded data from {file_path} with shape {df.shape}")

#Convert the text column to lowercase."""
df[text_column] = df[text_column].str.lower()
