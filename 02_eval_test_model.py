from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

from record_time import TimeHistoryRecord, write_json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Test an HybridCNNLSTM model')

parser.add_argument('--batch_size', metavar='<batch size>', default=32, type=int, help='batch size to use (default 128)')

args = parser.parse_args()

# Load Model
load_model = keras.models.load_model('models/model_cnn3_lstm1.h5')

# Test Model
from sklearn.metrics import confusion_matrix
import seaborn as sns

data_test = pd.read_csv("datasets/testData.csv")
print("data_test.tail:\n {}".format(data_test.tail(5)))

X_test = data_test.processed_text.values
y_test = data_test.condition.values

le = LabelBinarizer()
# One-hot encoding the classes (alphabetically by default)
y_test = le.fit_transform(y_test)

NUM_WORDS=24000
tokenizer = Tokenizer(num_words=NUM_WORDS)
sequences_test = tokenizer.texts_to_sequences(X_test)

SEQUENCE_LEN=440
X_test = pad_sequences(sequences_test, maxlen=SEQUENCE_LEN)
print("X_test.shape: {}".format(X_test.shape))

## eval model
print("\n=================== Evaluating Model ===================\n")
test_loss, test_acc = load_model.evaluate(X_test, y_test)
print('Test Loss : ',test_loss,' Test Accuracy : ',test_acc)


print("\n=================== Testing Model ===================\n")
predictions = load_model.predict(X_test, steps=100, batch_size=args.batch_size, verbose=1).argmax(axis=1)


# ## data visualization
# cm = confusion_matrix(y_test.argmax(axis=1), predictions, normalize='true')
# fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(cm, cmap='Blues', annot=True, ax = ax); 

# # Labels, title and ticks
# ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(list(le.classes_)); 
# ax.yaxis.set_ticklabels(list(le.classes_));
