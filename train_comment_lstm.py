import pandas as pd
import numpy as np
import os, sys
from bpe import BPE
from utils import *

SEQ_LENGTH = 24
MAX_EPOCHS = 100
BATCH_SIZE = 200
DATA_DIR = 'training_data'
EMB_SIZE = 200

if not os.path.exists('sequences.txt'):

    comments = load_text(DATA_DIR+'/comments.txt')
    comments = comments.replace('\n', ' endofcomment ')
    tokens = comments.split()

    print('Total Tokens: ', len(tokens))
    print('Unique Tokens: ', len(set(tokens)))

    length = SEQ_LENGTH + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        line = ' '.join(seq)
        sequences.append(line)

    #print(sequences[0])

    save_text(sequences, 'sequences.txt')

    print('Total Sequences: ', len(sequences))

#========TOKENIZE SEQUENCES========

doc = load_text('sequences.txt')
lines = doc.split('\n')

from keras.preprocessing.text import Tokenizer
from pickle import dump, load

if not os.path.exists('tokenizer.pkl'):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

tokenizer = load(open('tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

#========CREATE MODEL========

from keras.models import Sequential
from keras.layers import Dropout, Dense, CuDNNGRU, Embedding, TimeDistributed, BatchNormalization, Input

model = Sequential()
#model.add(Input(shape=(SEQ_LENGTH,)))
model.add(Embedding(vocab_size, EMB_SIZE, input_length=SEQ_LENGTH))
model.add(CuDNNGRU(128))
model.add(Dropout(0.2))
model.add(BatchNormalization(momentum=0.99))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_crossentropy'])

model.summary()

#========ASSEMBLING TRAINING DATA========
from keras.utils import to_categorical
sequences = np.array(sequences)
print(sequences.shape)
X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)

print(X.shape)
print(y.shape)

if os.path.exists('model.h5'):
    from keras.models import load_model
    model = load_model('model.h5')

model.fit(X, y, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)
model.save('model.h5')
