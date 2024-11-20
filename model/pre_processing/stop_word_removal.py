# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:14:07 2024

@author: mbnas
"""

import spacy
import pandas as pd
import ast

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def remove_stop_words(tokens):
    """
    Remove stop words from a list of tokens.
    """
    return [token for token in tokens if not nlp.vocab[token].is_stop]

def remove_stop_words_from_column(file_path, column_name, output_path=None):
    """
    Read the .tsv file, remove stop words from the tokenized sentence column, 
    and save the result to a new file.
    """
    # Read the .tsv file
    df = pd.read_csv(file_path, sep='\t')

    # Check if the column exists
    if column_name in df.columns:
        # Convert the string representation of lists back into actual lists (if necessary)
        df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Apply stop word removal to the specified tokenized sentence column
        df[column_name + '_no_stop'] = df[column_name].apply(lambda x: remove_stop_words(x))
    else:
        raise ValueError(f"Column '{column_name}' not found in the file.")

    # Drop the original text column if you don't want it in the output
    df = df.drop(columns=[column_name])

    # Save the processed file if output_path is provided
    if output_path:
        df.to_csv(output_path, sep='\t', index=False)

    return df



"""
Remember that this function only works on a already tokenized column.
"""

# # Example usage:
# input_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/processed_data.tsv"
# text_column = "sentence_tokens"  # Name of the tokenized sentence column
# output_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/processed_data_no_stop.tsv"

# # Preprocess the data (remove stop words)
# processed_df = remove_stop_words_from_column(input_file, text_column, output_file)

# # Check the result
# print(processed_df.head())
