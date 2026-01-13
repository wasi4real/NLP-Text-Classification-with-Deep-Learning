# Question-Answer Text Classification

## Overview

This project implements a comprehensive framework for classifying question-answer (QA) pairs into ten distinct categories using both traditional machine learning and modern deep learning architectures. The objective is to evaluate the effectiveness of various text representation techniques—ranging from frequency-based methods to pre-trained contextual embeddings—in the context of automated text categorization.

## Dataset

The project utilizes a dataset comprising approximately 280,000 QA entries. The data is balanced across the following ten classes:

* Business & Finance
* Computers & Internet
* Education & Reference
* Entertainment & Music
* Family & Relationships
* Health
* Politics & Government
* Science & Mathematics
* Society & Culture
* Sports

## Technical Pipeline

### 1. Data Preprocessing

Textual data undergoes standard NLP cleaning, including lowercasing and punctuation removal, followed by:

* **Tokenization**: Converting text into sequences of integers.
* 
**Padding**: Ensuring uniform sequence length (maxlen=100) for neural network input.


* 
**Label Encoding**: Converting categorical labels into numerical format for model training.



### 2. Feature Extraction

Four primary feature representation techniques were compared:

* **Frequency-based**: Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).
* **Word Embeddings**: Pre-trained GloVe vectors and custom-trained Skip-gram models.

### 3. Model Architectures

The project evaluates a suite of models including:

* **Classical ML**: Multinomial Naive Bayes, Logistic Regression, and Random Forest.
* **Neural Networks**: Deep Neural Networks (DNN), Simple RNNs, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).
* **Advanced Architectures**: Bidirectional LSTM and Bidirectional GRU variants.

## Performance Summary

The models were evaluated based on Accuracy and F1-Score (Macro and Weighted).

| Model Category | Top Performing Architecture | Best Feature Type | Highest Accuracy |
| --- | --- | --- | --- |
| Classical ML | Logistic Regression | TF-IDF | ~61% |
| Deep Learning | Bidirectional GRU | GloVe Embeddings | **71.32%** |

### Key Findings

* 
**Embedding Superiority**: Pre-trained GloVe embeddings consistently outperformed frequency-based methods in deep learning contexts by capturing semantic relationships.


* 
**Bidirectionality**: Models capable of processing sequences in both forward and backward directions (Bi-LSTM, Bi-GRU) significantly improved classification performance by capturing comprehensive context.


* 
**Top Performer**: The Bidirectional GRU utilizing GloVe embeddings achieved the highest overall accuracy of 71.32%.



## Repository Structure

* 
`CSE440_Project_Final.ipynb`: Complete implementation including EDA, model training, and evaluation.


* 
`NLP QA Classification- Project Report.pdf`: Detailed research paper documenting methodology, experimental setup, and results.


* 
`model_results.pkl`: Persisted metrics for all experimental runs.

