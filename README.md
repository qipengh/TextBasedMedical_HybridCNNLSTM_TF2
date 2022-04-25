# Text-based Classification of Medical Conditions using Hybrid CNN-LSTM model with TF2

This is Fork from [harshita219/Predicting_medical_condition](https://github.com/harshita219/Predicting_medical_condition).

Using UCI ML repository's Drug Reviews dataset to predict condition based on medical reviews. 
Structure of directory:

- raw_data: contains raw train and test data from https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
- cleaning.py: Script containing functions to pre-process texts
- preprocessing.ipynb : Notebook contain splitting and processing raw data
- modelling.pynb : Notebook using pre-trained word embeddings " GoogleNews-vectors-negative300.bin.gz." from https://code.google.com/archive/p/word2vec/ for predicting top10 conditions selected from raw data. Used sequential CNN-LSTM model.
- testData.csv: Processed data for testing
- trainData.csv: Processed data for training

# 00. Prepare Env

## Download " GoogleNews-vectors-negative300.bin.gz."

from git repo:[word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors), download from [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

get `*.bin` file by `$ gzip -d GoogleNews-vectors-negative300.bin.gz`

## install deps lib

`$ pip install -r requirements.txt`

# 01. Training Model

`$ python 01_train_model.py`
# Eval or Test Model

`$ python 02_eval_test_model.py`
