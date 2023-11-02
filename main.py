# importing packages
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# opening file and getting data
fields = ['hashtags', 'full_text']
fields2 = ['party_id']
train_x = pd.read_csv(r'\\kc.umkc.edu\kc-users\home\l\lerpfp\Desktop\pythonProject\congressional_tweet_training_data'
                       r'.csv',usecols= fields, nrows= 60000)
train_party = pd.read_csv(r'\\kc.umkc.edu\kc-users\home\l\lerpfp\Desktop\pythonProject\congressional_tweet_training_data'
                       r'.csv',usecols= fields2, nrows= 60000)
train_y = np.where(train_party['party_id']=='D', 1, 0)

#getting the hashtags as tokens in a dictionary FIX HERE
max_words = 500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x['hashtags'])
dictionary = tokenizer.word_index
dictionary = dict(list(dictionary.items())[:499])

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)



def convert_text_to_index_array(text):
    # making all text the same length and turing into an index array
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
    return wordIndices


allWordIndices = []
# chaning each token to its id
for text in train_x['hashtags']:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# making an array
allWordIndices = np.asarray(allWordIndices)
# create matrices out of the indexed tweets FIX HERE
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = to_categorical(train_y, 2)


# making model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# compiling network
model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])


# training my network
model.fit(train_x, train_y,
  batch_size=25,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)
