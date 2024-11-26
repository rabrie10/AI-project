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
9. [References](#references)

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

## How to Run

### Preprocessing the Data
1. Navigate to the `model/preprocessing/` directory.
2. Open the `preprocessing.py` file and update the `input_file` variable with the path to your input dataset. For example:
   ```python
   input_file = "path/to/your/input_file.tsv"
   ```
3. Run the script using the following command:
   ```bash
   python preprocessing.py
   ```
4. The preprocessed data will be saved in the output location specified inside the script.

### Running the Models
1. Locate the model you want to run in the `model/model_implementation/` directory.
2. Open the script for the model you wish to test (e.g., `CNN.py`, `RNN.py`, or `BERT.py`).
3. Update the `test_data` or `test_df` line to point to the path of your test dataset. For example:
   ```python
   test_df = pd.read_csv("path/to/your/test_data.tsv", sep="\t")
   ```
4. Run the model script using:
   ```bash
   python <model_name>.py
   ```
5. The predictions will be saved in the `model_outputs/` directory, with the file named after the model you ran.

### Evaluating the Models
1. Navigate to the `model/` directory.
2. Open the `model_scorer.py` file and update the following variables:
   - **`gold_file_path`**: Path to the gold standard test dataset.
   - **`pred_file_path`**: Path to the output predictions from your model.
   - **`output_figures`**: Desired name for the result figures.

   Example:
   ```python
   gold_file_path = "path/to/gold_file.tsv"
   pred_file_path = "path/to/predictions.tsv"
   output_figures = "model_evaluation_results"
   ```
3. Run the evaluation script:
   ```bash
   python model_scorer.py
   ```
4. The evaluation results, including figures and metrics, will be saved in the `evaluation_scores_data/` directory.



## Requirements
To run the project, install the following libraries:
- `transformers`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `spacy`
- `numpy`
- `xgboost`
- `imbalanced-learn`

Install the dependencies using:
```bash
pip install -r requirements.txt



