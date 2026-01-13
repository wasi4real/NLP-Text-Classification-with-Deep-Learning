# Question-Answer Classification: Architecture and Performance Analysis

## Overview

This repository contains a comprehensive framework for the classification of question-answer (QA) pairs into ten distinct categories. The project evaluates the efficacy of traditional machine learning (ML) versus deep neural architectures, utilizing diverse feature representation strategies to categorize approximately 280,000 QA entries.

## Data Preprocessing Pipeline

Input processing follows a deterministic sequence to ensure data integrity and model compatibility:

* **Text Normalization**: All QA text is converted to lowercase and stripped of punctuation.


* **Categorical Encoding**: The ten target classes (e.g., Science & Mathematics, Sports, Health) are transformed into discrete integer labels using `LabelEncoder`.


* **Tokenization**: Textual data is converted into integer sequences based on a vocabulary mapping.


* **Sequence Padding**: To maintain uniform tensor dimensions for neural input, sequences are padded or truncated to a fixed length of 100 tokens using 'post' padding.



## Feature Engineering

The project implements and compares four vectorization methods:

* **Bag of Words (BoW)**: Frequency-based count matrix.
* **TF-IDF**: Statistical weighting for term importance, utilizing unigrams and bigrams with a maximum of 10,000 features.


* **GloVe Embeddings**: 100-dimensional pre-trained contextual vectors.


* **Skip-gram (Word2Vec)**: Custom embeddings trained specifically on the project corpus.



## Model Architectures

### Classical Machine Learning

* **Multinomial Naive Bayes**: A probabilistic baseline for text classification.
* **Logistic Regression**: Utilized with TF-IDF features to provide a strong linear baseline.


* **Random Forest**: An ensemble method used to capture non-linear relationships in frequency-based features.



### Deep Neural Networks

Neural architectures are constructed using the Sequential API with the following layers:

* **Embedding Layer**: Maps integer sequences to dense vectors (using GloVe, Skip-gram, or custom training).


* **Recurrent Layers**: SimpleRNN, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) are employed to capture temporal dependencies.


* **Bidirectional Wrappers**: Applied to RNN, LSTM, and GRU layers to process information from both the beginning and end of the sequence simultaneously.


* **Dense and Dropout**: Fully connected layers with softmax activation for classification, integrated with Dropout layers to prevent overfitting.



## Results and Analytical Insights

### Performance Comparison

| Model Strategy | Feature Type | Accuracy | Weighted F1-Score |
| --- | --- | --- | --- |
| Logistic Regression | TF-IDF | 61.02% | 0.61 |
| SimpleRNN | Skip-gram | 52.82% | 0.52 |
| LSTM | GloVe | 60.93% | 0.61 |
| **Bidirectional GRU** | **GloVe** | **71.32%** | **0.71** |

### Key Insights

* **Pre-trained Advantage**: GloVe embeddings consistently outperformed custom Skip-gram and frequency-based models in neural contexts, providing superior semantic density.


* **Bidirectionality**: The transition from unidirectional to bidirectional architectures provided a significant performance boost, as it allows the model to capture the full context of a question regardless of where key keywords are located.


* **Baseline Efficiency**: Logistic Regression with TF-IDF remains a viable, low-compute baseline, nearly matching the performance of standard unidirectional LSTMs.


* **Optimal SOTA**: The Bidirectional GRU with GloVe embeddings is the state-of-the-art configuration for this dataset, achieving the highest accuracy of 71.32%.
