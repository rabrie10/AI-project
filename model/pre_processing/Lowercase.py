# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:10:32 2024

@author: mbnas
"""

import pandas as pd

def lowercase_text(text):
    """
    Converts input text to lowercase.
    
    Args:
        text (str): The text to be converted to lowercase.
        
    Returns:
        str: The text in lowercase.
    """
    return text.lower() #

def preprocess_lowercase(file_path, column_name, output_path=None):
    """
    Reads a .tsv file, converts a specific column's text to lowercase, and saves the processed file.
    
    Args:
        file_path (str): Path to the input .tsv file.
        column_name (str): Column name to apply the lowercase transformation.
        output_path (str): Path to save the processed .tsv file. If None, file won't be saved.
        
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Read the .tsv file
    df = pd.read_csv(file_path, sep='\t')
    
    # Apply the lowercase function to the specified column
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(lowercase_text)
    else:
        raise ValueError(f"Column '{column_name}' not found in the file.")
    
    # Save the processed file if output_path is provided
    if output_path:
        df.to_csv(output_path, sep='\t', index=False)
    
    
    return df



# # Path to the .tsv file
# input_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/test_en.tsv"

# # Column to process
# text_column = "sentence"

# # Path for the output file
# output_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/processed_data.tsv"

# # Preprocess the data
# processed_df = preprocess_lowercase(input_file, text_column, output_file)

# print(processed_df.head())
