# AI project
ask 2: Multilingual Subjectivity in News Articles
Mentor to be contacted: Shaghayegh Vafi (s.vafi@stud.uis.no)
Goal: The task is for systems to distinguish whether a sentence from a news article expresses the
subjective view of the author behind it or presents an objective view on the covered topic instead.
This is a binary classification task where the model has to identify whether a text sequence (a
sentence or paragraph) is subjective or objective.
Dataset: https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task2
Instructions
1. Data Preprocessing:
o Read the dataset
o Lowercasing: Convert all text to lowercase to avoid treating words like "News" and
"news" as different tokens.
o Remove Punctuation and Special Characters: Stripping unnecessary characters
that don't carry meaning (commas, exclamation points, etc.).
o Tokenization: Split text into individual words or tokens. Libraries like nltk, spaCy, or
transformers can handle this.
o Stop Word Removal: Remove common words like "the", "is", and "and" that don't
contribute much to the subjectivity of a sentence.
o Lemmatization or Stemming: Reduce words to their base form (e.g., "running"
becomes "run").
2. Exploratory Analysis
o To begin this exploratory analysis, first use matplotlib to import libraries and define
functions for plotting the data. Depending on the data, not all plots will be made.
3. Feature Extraction
o You can either use one of the methods to convert the text data into numerical
features that a model can process.
o Bag-of-Words (BoW)
o TF-IDF (Term Frequency-Inverse Document Frequency)
o Word Embeddings (Word2Vec, GloVe, FastText
o Pretrained transformer Transformers (e.g., BERT, GPT)
4. Model selection:
o Different machine learning and deep learning models can be used:
▪ Logistic Regression, Support Vector Machines (SVM), Random Forest or
XGBoost Can be used with BoW or TF-IDF features.
▪ Deep Learning (RNNs, CNNs)
▪ Transformers (BERT, RoBERTa, DistilBERT)
▪ Also, you must apply different models and compare the results.
5. Evaluation Metrics:
o Extend the evaluation metrics to include:
▪ Confusion matrix
▪ Precision, Recall, and F1-score for a more detailed analysis of the model
performance, especially since this is a binary classification task.
o Analyze the results across multiple languages and report the findings, focusing on
how well the model generalizes to multilingual data.
6. Model Training:
o Train the model on the training split and validate on the development test split.
o Plot the training and validation accuracy/loss curves over the training epochs.
o Experimenting with different models like BERT or SVMs to see which performs best
on your dataset
