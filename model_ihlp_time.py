import os
import re
import time
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tcn import TCN
from tensorflow.python.keras.layers import SpatialDropout1D
from classes.ihlp import IHLP
from sklearn.model_selection import KFold
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPool1D, Flatten, Dense, Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

tf.config.list_physical_devices('GPU')
os.environ["SM_FRAMEWORK"] = "tf.keras"


def get_has_error(x):
    return (
        isinstance(x['solutionDate'], str)
    )


def get_process(x):
    return int(x.solutionDate.timestamp()) - int(x.receivedDate.timestamp())


def get_cleanup(x):
    text = BeautifulSoup(x.description, "lxml").text
    text = text.lower()
    text = re.sub('[\n.]', ' ', text)
    return text


data = IHLP().get()
sentences, labels = list(data.processText), list(data.processTimeDense)


def get_max_length(sequences):
    max_length = 0
    for idx, seq in enumerate(sequences):
        length = len(seq)
        if max_length < length:
            max_length = length
    return max_length


trunc_type = 'post'
padding_type = 'post'
oov_tok = "<UNK>"

print("Example of sentence: ", sentences[2])

# Cleaning and Tokenization
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

# Turn the text into sequence
training_sequences = tokenizer.texts_to_sequences(sentences)
max_len = get_max_length(training_sequences)

print('Into a sequence of int:', training_sequences[2])

# Pad the sequence to have the same size
training_padded = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
print('Into a padded sequence:', training_padded[2])


def define_model(_kernel_size=3, _activation='relu', _input_dim=None, _output_dim=300, _max_length=None):

    inp = Input(shape=(_max_length,))
    x = Embedding(input_dim=_input_dim, output_dim=_output_dim, input_length=_max_length)(inp)
    x = SpatialDropout1D(0.2)(x)

    x = TCN(128, dilations=[1, 2, 4], return_sequences=True, activation=_activation, name='tcn1')(x)
    x = TCN(64, dilations=[1, 2, 4], return_sequences=True, activation=_activation, name='tcn2')(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.2)(conc)
    outp = Dense(1)(conc)

    _model = Model(inputs=inp, outputs=outp)
    _model.compile(loss='mae', optimizer='adam', metrics=tf.keras.metrics.MeanAbsoluteError())

    return _model

model_0 = define_model(_input_dim=1000, _max_length=100)
model_0.summary()

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                             patience=7, verbose=2,
                                             mode='auto', restore_best_weights=True)

# Parameter Initialization
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<UNK>"
activations = ['relu']
filters = 100
kernel_sizes = [1, 2, 3, 4, 5, 6]

columns = ['Activation', 'Filters', 'acc1', 'acc2', 'acc3', 'acc4', 'acc5', 'acc6', 'acc7', 'acc8', 'acc9', 'acc10', 'AVG']
record = pd.DataFrame(columns=columns)

# prepare cross validation with 10 splits and shuffle = True
kfold = KFold(10, shuffle=True)

# Separate the sentences and the labels
sentences, labels = list(data.processText), list(data.processTimeDense)

for train, test in kfold.split(sentences):

    train_x, test_x = [], []
    train_y, test_y = [], []

    for i in train:
        train_x.append(sentences[i])
        train_y.append(labels[i])

    for i in test:
        test_x.append(sentences[i])
        test_y.append(labels[i])

    # Turn the labels into a numpy array
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    # encode data using
    # Cleaning and Tokenization
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(train_x)

    # Turn the text into sequence
    training_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    max_len = get_max_length(training_sequences)

    # Pad the sequence to have the same size
    Xtrain = pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
    Xtest = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    # Define the input shape
    model = define_model(_input_dim=vocab_size, _max_length=max_len)

    print('Training ...')

    # Train the model
    model.fit(Xtrain, train_y, batch_size=50, epochs=25, callbacks=[callbacks], validation_data=(Xtest, test_y))

    # evaluate the model
    loss, acc = model.evaluate(Xtest, test_y, verbose=0)
    print('Test Accuracy: {}'.format(acc * 100))

    model.save('saved_model/model_ihlp_time/1.h5')

    with open('saved_model/model_ihlp_time/1.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dictionary = {
        "max_len": max_len,
        "trunc_type": trunc_type,
        "padding_type": padding_type
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("saved_model/model_ihlp_time/1.json", "w") as outfile:
        outfile.write(json_object)

    break
