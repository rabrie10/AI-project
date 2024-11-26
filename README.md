# Subjectivity Detection in Sentences

## Project Overview
This project aims to classify sentences as either **subjective** (expressing opinions, feelings, or personal views) or **objective** (stating facts or impartial information). By applying advanced natural language processing techniques, the goal is to identify distinguishing features of subjectivity and achieve high classification performance.

## Table of Contents
1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Model Selection](#model-selection)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)

---

## Dataset
The dataset contains labeled sentences, where:
- **SUBJ** indicates subjective sentences.
- **OBJ** indicates objective sentences.

The data was split into training and testing sets for model training and evaluation.

---

## Preprocessing
Several preprocessing steps were applied to clean and prepare the data:
1. **Lowercasing**: Converted all text to lowercase for uniformity.
2. **Removing Punctuation and Special Characters**: Eliminated unnecessary symbols.
3. **Tokenization**: Split sentences into individual tokens (words).
4. **Stop-Word Removal**: Removed common stop words while retaining relevant ones for subjectivity detection.
5. **Lemmatization**: Converted words to their base forms.

---

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the data better:
- **Class Distribution**: Analyzed the balance between subjective and objective sentences.
- **Most Frequent Words**: Identified common words in both categories.
- **Word Pair Analysis**: Explored co-occurrences of words.
- **Sentiment Distribution**: Examined sentiment variations across labels.
- **Part of Speech Analysis**: Analyzed the grammatical patterns in sentences.

---

## Model Selection
Various models were implemented and compared:
1. **Traditional Machine Learning**:
   - Naive Bayes
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors
2. **Deep Learning**:
   - Convolutional Neural Networks (CNNs)
   - Recurrent Neural Networks (RNNs)
3. **Transformer Models**:
   - BERT
   - RoBERTa

BERT achieved the highest performance due to its contextual understanding and pre-trained features.

---

## Evaluation Metrics
The models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Area Under Curve (AUC)**
- **Precision-Recall Curve**
- **Matthews Correlation Coefficient (MCC)**

---

## Results
- **BERT** outperformed other models with a balanced accuracy across subjective and objective sentences.
- The data imbalance (objective sentences dominating the dataset) significantly impacted model performance, particularly for traditional and deep learning models.