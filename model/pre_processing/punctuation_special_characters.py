# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:26:50 2024

@author: mbnas
"""

#This is a character class that matches any character not in the set a-z, A-Z, 0-9, or whitespace (\s). The ^ at the beginning of the character class negates it, so it matches any character that is not a letter, digit, or whitespace.
import re
import pandas as pd

def remove_punctuation_special_chars(text):
    """
    Removes punctuation and special characters from the input text.
    
    Args:
        text (str): The text to process.
        
    Returns:
        str: The cleaned text.
    """
    # Use regex to remove characters that are not letters, numbers, or spaces
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)#This is a character class that matches any character not in the set a-z, A-Z, 0-9, or whitespace (\s). The ^ at the beginning of the character class negates it, so it matches any character that is not a letter, digit, or whitespace.


def preprocess_remove_special_chars(file_path, column_name, output_path=None):
    """
    Reads a .tsv file, removes punctuation and special characters from a specific column, 
    and saves the processed file.
    
    Args:
        file_path (str): Path to the input .tsv file.
        column_name (str): Column name to apply the cleaning.
        output_path (str): Path to save the processed .tsv file. If None, file won't be saved.
        
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    # Read the .tsv file
    df = pd.read_csv(file_path, sep='\t')
    
    # Apply the cleaning function to the specified column
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(remove_punctuation_special_chars)
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
# processed_df = preprocess_remove_special_chars(input_file, text_column, output_file)

# print(processed_df.head())

