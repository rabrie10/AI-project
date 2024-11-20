# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:27:54 2024

@author: mbnas
"""

import os
from Lowercase import preprocess_lowercase
from punctuation_special_characters import preprocess_remove_special_chars
from tokenization import tokenize_column
from stop_word_removal import remove_stop_words_from_column
from lemmatization import lemmatize_column

def delete_file(file_path):
    """Helper function to delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File not found: {file_path}")



def preprocessing(input_file, text_column, output_file=None):
    """
    Preprocess the text data by applying all preprocessing steps.
    """
    # Step 1: Lowercasing
    print("Starting Lowercasing...")
    lowercased_df = preprocess_lowercase(input_file, text_column, output_path="temp_lowercase.tsv")
    lowercased_file = "temp_lowercase.tsv"
    
    # Step 2: Removing punctuation and special characters
    print("Starting Removing punctuation and special characters...")
    punctuation_special_char_removed_df = preprocess_remove_special_chars(lowercased_file, text_column, output_path="temp_no_punctuation_special_char.tsv")
    punctuation_special_char_removed_file = "temp_no_punctuation_special_char.tsv"
    
    # Step 3: Tokenization
    print("Starting Tokenization...")
    tokenized_df = tokenize_column(punctuation_special_char_removed_file, text_column, output_path="temp_tokenized.tsv")
    tokenized_file = "temp_tokenized.tsv"
    
    # Step 4: Remove stop words
    print("Removing Stop Words...")
    stop_words_removed_df = remove_stop_words_from_column(tokenized_file, 'sentence_tokens', output_path="temp_no_stop.tsv")
    stop_words_removed_file = "temp_no_stop.tsv"
    
    # Step 5: Lemmatization
    print("Lemmatizing the tokens...")
    lemmatized_df = lemmatize_column(stop_words_removed_file, 'sentence_tokens_no_stop', output_path="final_processed_data.tsv")
    
    # Rename the processed column back to the original text_column
    print(f"Renaming processed column to '{text_column}'...")
    lemmatized_df.rename(columns={'sentence_tokens_no_stop_lemmatized': text_column}, inplace=True)
    
    # Final file output
    final_file = "final_processed_data.tsv"
    
    # Save the final processed DataFrame to the output file
    if output_file:
        lemmatized_df.to_csv(output_file, sep='\t', index=False)
        print(f"Processed data saved to {output_file}")
    else:
        lemmatized_df.to_csv(final_file, sep='\t', index=False)
        print(f"Processed data saved to {final_file}")
        
    
    # Delete temporary files
    delete_file(lowercased_file)
    delete_file(punctuation_special_char_removed_file)
    delete_file(tokenized_file)
    delete_file(stop_words_removed_file)
    
    return lemmatized_df



# Path to the input .tsv file
input_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/test_en.tsv"
text_column = "sentence"  # Column to process
# output_file = "C:/Users/mbnas/.spyder-py3/AI-project/subjectivity_determiner/data/subtask-2-english/processed_data_final.tsv"

# Run preprocessing
processed_df = preprocessing(input_file, text_column)

# Print a sample of the processed DataFrame
print(processed_df.head())


