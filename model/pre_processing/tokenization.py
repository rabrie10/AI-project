# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:37:22 2024

@author: mbnas
"""

import spacy
import pandas as pd

# Load the pre-trained model
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    # Process the text using spaCy's pipeline
    doc = nlp(text)
    # Return a list of tokens (words), filtering out empty, whitespace, or single-character tokens
    return [token.text for token in doc if token.text.strip() != '' and len(token.text) > 1]


def tokenize_column(file_path, column_name, output_path=None):
    
    # Read the .tsv file
    df = pd.read_csv(file_path, sep='\t')
    
    # Apply tokenization to the specified column
    if column_name in df.columns:
        df[column_name + '_tokens'] = df[column_name].apply(lambda x: tokenize_text(x))
    else:
        raise ValueError(f"Column '{column_name}' not found in the file.")
        
    # Drop the original text column if you don't want it in the output
    df = df.drop(columns=[column_name])
        
    # Save the processed file if output_path is provided
    if output_path:
        df.to_csv(output_path, sep='\t', index=False)
    return df


# # Example usage:

# # Path to the .tsv file
# input_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/test_en.tsv"

# # Column to process
# text_column = "sentence"

# # Path for the output file
# output_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/processed_data.tsv"

# # Preprocess the data
# processed_df = tokenize_column(input_file, text_column, output_file)

# print(processed_df.head())


