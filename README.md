# Text-based Classification of Medical Conditions using Hybrid CNN-LSTM model 
Using UCI ML repository's Drug Reviews dataset to predict condition based on medical reviews. 
Structure of directory:

- raw_data: contains raw train and test data from https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29
- cleaning.py: Script containing functions to pre-process texts
- preprocessing.ipynb : Notebook contain splitting and processing raw data
- modelling.pynb : Notebook using pre-trained word embeddings " GoogleNews-vectors-negative300.bin.gz." from https://code.google.com/archive/p/word2vec/ for predicting top10 conditions selected from raw data. Used sequential CNN-LSTM model.
- testData.csv: Processed data for testing
- trainData.csv: Processed data for training
