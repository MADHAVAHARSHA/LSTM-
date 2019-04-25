from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import sequence

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from keras.layers import Dense, Embedding
from keras.layers import LSTM

from sklearn.datasets import fetch_20newsgroups
from keras.models import Sequential
max_features = 20000

maxlen = 80
batch_size = 32


def load_20ng_dataset_bow():
   
    k=['alt.atheism','rec.autos']
    newsgroups_train = fetch_20newsgroups(subset='train',categories=k)
    newsgroups_train.target_names
   
    newsgroups_test = fetch_20newsgroups(subset='test',categories=k)
    newsgroups_test.target_names

   

    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.95)
    train_data = vectorizer.fit_transform(newsgroups_train.data)
    test_data = vectorizer.transform(newsgroups_test.data)
    train_data = train_data.todense()
    test_data = test_data.todense()
    train_labels = newsgroups_train.target
    test_labels = newsgroups_test.target

    return train_data, train_labels, test_data, test_labels

   


np.random.seed(1)
n_train = 715
   
train_data, train_labels, test_data, test_labels = load_20ng_dataset_bow()
train_data = train_data[:n_train, :]
train_labels = train_labels[:n_train]
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='softmax'))



model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
              metrics=['accuracy'])

print('Train...')
model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=1,
validation_data=(test_data, train_labels))
score, acc = model.evaluate(test_data, train_labels,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
   

