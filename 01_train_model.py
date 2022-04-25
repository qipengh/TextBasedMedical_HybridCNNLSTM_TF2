
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cleaning import text_prepare # self-created library

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from gensim.models import KeyedVectors

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout, Activation, LSTM, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
emb = KeyedVectors.load_word2vec_format('datasets/GoogleNews-vectors-negative300.bin', binary=True)

data_train = pd.read_csv("datasets/trainData.csv")
print("data_train:\n {}".format(data_train.tail(5)))

X = data_train.processed_text.values
y = data_train.condition.values

# One-hot encoding the classes (alphabetically by default)
le = LabelBinarizer()
y = le.fit_transform(y)
print("le.classes_:\n {}".format(le.classes_))

print("y:\n {}".format(y))

# ### Basic Logistic Regression

# Splitting data into train:test by 80:20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

# Creating the vectorizer
tfidf = TfidfVectorizer(max_features = 10000, ngram_range=(1,3), min_df=5, max_df=0.9)
tfidf = tfidf.fit(X_train)

X_train = tfidf.transform(X_train).toarray()
print('\nTraining features shape: ',X_train.shape)

X_val = tfidf.transform(X_val).toarray()
print('Test features shape:     ',X_val.shape)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
clf = clf.fit(X_train, y_train)
prediction = clf.predict(X_val)

print("\nClassifier results: \n")
print(classification_report(prediction, y_val, target_names = list(le.classes_)))

# ### CNN-LSTM

# Splitting data into train:test by 80:20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 42)

NUM_WORDS=24000
tokenizer = Tokenizer(num_words=NUM_WORDS)

# Converting text to tokens
tokenizer.fit_on_texts(X_train)

# Converting into sequential data
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_valid = tokenizer.texts_to_sequences(X_val)

# 
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])

print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)

SEQUENCE_LEN = X_train.shape[1]
EMBEDDING_DIM = 300

vocabulary_size = min(len(word_index)+1, NUM_WORDS)

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= NUM_WORDS:
        continue
    try:
        embedding_vector = emb[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i] = np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

print("embedding_matrix.shape: {}".format(embedding_matrix.shape))

print("SEQUENCE_LEN: {}".format(SEQUENCE_LEN))

model= Sequential()
model.add(Embedding(NUM_WORDS, EMBEDDING_DIM, weights = [embedding_matrix], input_length = SEQUENCE_LEN))

model.add(Conv1D(32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.3))

model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.3))

model.add(Conv1D(128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.4))

model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])    
print("model.summary: \n{}".format(model.summary()))

print("\n=================== Training Model ===================\n")
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

# ## data visualization
# history_df = pd.DataFrame(model.history.history).rename(columns={"loss":"train_loss", "accuracy":"train_accuracy"})
# history_df.plot(figsize=(8,8))
# plt.grid(True)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

model.save('models/model_cnn3_lstm1.h5')
print("model saved in {}".format('models/model_cnn3_lstm1.h5'))

