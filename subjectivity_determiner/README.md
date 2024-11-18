# Task 2: Subjectivity in News Articles

Systems are challenged to distinguish whether a sentence from a news article expresses the subjective view of the author behind it or presents an objective view on the covered topic instead. This is a binary classification tasks in which systems have to identify whether a text sequence (a sentence or a paragraph) is **subjective** or **objective**.

The task is offered in Arabic, Bulgarian, English, German, and Italian. We also offer the task for a multilingual setting that mixes all the previous languages.

Information regarding the annotation guidelines can be found in the following papers:

> Federico Ruggeri, Francesco Antici, Andrea Galassi, aikaterini Korre, Arianna Muti, Alberto Barron,  _[On the Definition of Prescriptive Annotation Guidelines for Language-Agnostic Subjectivity Detection](https://ceur-ws.org/Vol-3370/paper10.pdf)_, in: Proceedings of Text2Story — Sixth Workshop on Narrative Extraction From Texts, CEUR-WS.org, 2023, Vol 3370, pp. 103 - 111

> Francesco Antici, Andrea Galassi, Federico Ruggeri, Katerina Korre, Arianna Muti, Alessandra Bardi, Alice Fedotova, Alberto Barrón-Cedeño, _[A Corpus for Sentence-level Subjectivity Detection on English News Articles](https://arxiv.org/abs/2305.18034)_, in: Proceedings of Joint International Conference on Computational Linguistics, Language Resources and Evaluation (COLING-LREC), 2024



__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 2 Directory](#contents-of-the-repository)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)

<!-- ## Evaluation Results

TBA -->

## List of Versions
- [29/04/2024] test data released.
- [25/01/2024] training data released.

<!-- * **subtask-2A-english**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] previously released training data contained also validation data, they are now split in two separate files.
  - [30/01/2023] training data are released.
* **subtask-2A-arabic**
  - [03/05/2023] (unlabeled) test data are released.
  - [10/03/2023] training and validation data are released.
* **subtask-2A-dutch**
  - [03/05/2023] (unlabeled) test data are released.
  - [16/03/2023] training and validation data are released.
* **subtask-2A-german**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-italian**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] validation data are released.
  - [30/01/2023] training data are released.
* **subtask-2A-turkish**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-multilingual**
  - [03/05/2023] (unlabeled) test data are released.
  - [23/03/2023] training and validation data are released. -->

## Contents of the Task 2 Directory

- Main folder: [data](./data)
  - Contains a subfolder for each language which contain the data as TSV format with .tsv extension (train_LANG.tsv, dev_LANG.tsv, test_LANG.tsv).
  As LANG we used standard language code for each language.
- Main folder: [baseline](./baseline)<br/>
  - Contains a file, baseline.py, used to train a baseline and provide predictions.
  - Contains a .tsv file for each language, consisting of the predictions of the baseline over the dev sets.
  - Contains a README.md file that reports the scores obtained by the baseline on the dev sets.
- Main folder: [scorer](./scorer)<br/>
  - Contains a single file, evaluate.py, that checks the format of a submission and evaluate the various metrics.
- [README.md](./README.md) <br/>

## Datasets statistics

<!-- * **subtask-2A-arabic**
  - train: 1185 sentences, 905 OBJ, 280 SUBJ
  - dev: 297 sentences, 227 OBJ, 70 SUBJ
  - test - 445 sentences, 363 OBJ, 82 SUBJ
* **subtask-2A-dutch**
  - train: 800 sentences, 489 OBJ, 311 SUBJ
  - dev: 200 sentences, 107 OBJ, 93 SUBJ
  - test - 500 sentences, 263 OBJ, 237 SUBJ
* **subtask-2A-english**
  - train: 830 sentences, 352 OBJ, 298 SUBJ
  - dev: 219 sentences, 106 OBJ, 113 SUBJ
  - test - 243 sentences, 116 OBJ, 127 SUBJ
* **subtask-2A-german**
  - train: 800 sentences, 492 OBJ, 308 SUBJ
  - dev: 200 sentences, 123 OBJ, 77 SUBJ
  - test - 291 sentences, 194 OBJ, 97 SUBJ
* **subtask-2A-italian**
  - train: 1613 sentences, 1231 OBJ, 382 SUBJ
  - dev: 227 sentences, 167 OBJ, 60 SUBJ
  - test - 440 sentences, 323 OBJ, 117 SUBJ
* **subtask-2A-turkish**
  - train: 800 sentences, 422 OBJ, 378 SUBJ
  - dev: 200 sentences, 100 OBJ, 100 SUBJ
  - test - 240 sentences, 129 OBJ, 111 SUBJ
* **subtask-2A-multilingual**
  - train: 6628 sentences, 4,371 OBJ, 2,257 SUBJ
  - dev: 600 sentences, 300 OBJ, 300 SUBJ
  - test - 600 sentences, 300 OBJ, 300 SUBJ -->

* **subtask-2A-english**
  - train: 830 sentences, 532 OBJ, 298 SUBJ
  - dev: 219 sentences, 106 OBJ, 113 SUBJ
  - dev-test: 243 sentences, 116 OBJ, 127 SUBJ 
  - test: 490 sentences
* **subtask-2A-italian**
  - train: 1613 sentences, 1231 OBJ, 382 SUBJ
  - dev: 227 sentences, 167 OBJ, 60 SUBJ
  - dev-test - 440 sentences, 323 OBJ, 117 SUBJ
  - test: 513 sentences
* **subtask-2A-german**
  - train: 800 sentences, 492 OBJ, 308 SUBJ
  - dev: 200 sentences, 123 OBJ, 77 SUBJ
  - dev-test - 291 sentences, 194 OBJ, 97 SUBJ
  - test: 337 sentences
* **subtask-2A-bulgarian**
  - train: 729 sentences, 406 OBJ, 323 SUBJ
  - dev: 106 sentences, 59 OBJ, 47 SUBJ
  - dev-test - 208 sentences, 116 OBJ, 92 SUBJ
  - test: 250 sentences
* **subtask-2A-arabic**
  - train: 1185 sentences, 905 OBJ, 280 SUBJ
  - dev: 297 sentences, 227 OBJ, 70 SUBJ
  - dev-test - 445 sentences, 363 OBJ, 82 SUBJ
  - test: 748 sentences
* **subtask-2A-multilingual**
  - train (ml_only_2024_languages): 5159 sentences, 3568 OBJ, 1591 SUBJ
  - dev (ml_only_2024_languages): 500 sentences, 250 OBJ, 250 SUBJ
  - dev-test (ml_only_2024_languages) - 500 sentences, 250 OBJ, 250 SUBJ
  - test: 500 sentences

## Input Data Format

The data will be provided as a TSV file with three columns:
> sentence_id <TAB> sentence <TAB> label

Where: <br>
* sentence_id: sentence id for a given sentence in a news article<br/>
* sentence: sentence's text <br/>
* label: *OBJ* and *SUBJ*

<!-- **Note:** For English, the training and development (validation) sets will also include a fourth column, "solved_conflict", whose boolean value reflects whether the annotators had a strong disagreement. -->

**Examples:**

> b9e1635a-72aa-467f-86d6-f56ef09f62c3  Gone are the days when they led the world in recession-busting SUBJ  True
>
> f99b5143-70d2-494a-a2f5-c68f10d09d0a  The trend is expected to reverse as soon as next month.  OBJ  False

## Output Data Format

The output must be a TSV format with two columns: sentence_id and label.

## Evaluation Metrics

This task is evaluated as a classification task. We will use the F1-macro measure for the ranking of teams.

We will also measure Precision, Recall, and F1 of the SUBJ class and the macro-averaged scores.
<!--
There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt -->

## Scorers

To evaluate the output of your model which should be in the output format required, please run the script below:

> python evaluate.py -g dev_truth.tsv -p dev_predicted.tsv

where dev_predicted.tsv is the output of your model on the dev set, and dev_truth.tsv is the golden label file provided by us.

The file can be used also to validate the format of the submission, simply use the provided test file as gold data.
The evaluation will not be performed, but the format of your input will be checked.


## Baselines

The script to train the baseline is provided in the related directory.
The script can be run as follow:

> python baseline.py -trp train_data.tsv -ttp dev_data.tsv

where train_data.tsv is the file to be used for training and dev_data.tsv is the file on which doing the prediction.

The baseline is a logistic regressor trained on a Sentence-BERT multilingual representation of the data.

<!-- ### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498
 -->

## Submission

Submission is done through the Codalab platform at: https://codalab.lisn.upsaclay.fr/competitions/18809

-   Make sure that you create one account for each team, and submit it through one account only.
-   The last file submitted to the leaderboard will be considered as the final submission.
-   For subtask 2A, there are 5 languages (Arabic, Bulgarian, English, German and Italian). Moreover, we define a multi-lingual evaluation scenario where we use a balanced sub-sample of all 5 languages to define multi-lingual evaluation splits.
-   Name of the output file has to be `subtask2A_LANG.tsv` where LANG can be arabic, bulgarian, english, german, italian, multilingual.
-   Get sure to set `.tsv` as the file extension; otherwise, you will get an error on the leaderboard.
-   Examples of submission file names should be `subtask2A_arabic.tsv`, `subtask2A_bulgarian.tsv`, `subtask2A_english.tsv`, `subtask2A_german.tsv`, `subtask2A_italian.tsv`, `subtask2A_multilingual.tsv`.
-   You have to zip the tsv into a file with the same name, e.g., `subtask2A_arabic.zip`, and submit it through the codalab page.
-   If you participate in the task for more than one language, for each language you must do a different submission.
-   for each submission, it is required to submit the team name. **Your team name must EXACTLY match the one used during the CLEF registration.**
-   You are allowed to submit max 200 submissions per day for each subtask.
-	Additionally, we ask you to fill out a questionnaire (link will be provided, once the evaluation cycle started started) to provide some details on your approach as we need that information for the overview paper.
-   We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period. -->

<!-- Each participant must submit their results as a single .ZIP file. -->


<!--
The file must contain .TSV files, one file for each result.

Participants are allowed to submit up to 2 results for each language.

For each language, one will be the "main" one and considered for the final ranking,
the other will be evaluated but not considered for the final ranking.

The main file must be name as "result_LN_TYPE_TEAMNAME.csv";
where LAN are two letters that identify the language
(must be either EN, AR, NL, DE, IT, TR, or ML for multi-lingual),
TYPE are four characters that indicate the type of submission
(must be either "MAIN", "ALT1"),
TEAMNAME is the identifier name of the team.
-->


## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2024/task2/
