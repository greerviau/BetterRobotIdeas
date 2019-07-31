from utils import *
import random
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

SEQ_LENGTH = 24
BATCH_SIZE = 200
CONF_THRESH = 0.6

doc = load_text('sequences.txt')
lines = doc.split('\n')

from pickle import load

tokenizer = load(open('tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

model = load_model('model.h5')

for i in range(5):
    result = []
    in_text = lines[random.randint(0,len(lines))].split()
    in_text[len(in_text)-1] = 'endofcomment'
    in_text = ' '.join(in_text)
    print('\n--------SAMPLE {}-------'.format(i+1))
    print('----------Seed---------\n',in_text)
    print('-------Generated-------')
    for _ in range(10):
        new_comment = ''
        while True:
            if len(new_comment.split()) >= SEQ_LENGTH:
                in_text += ' endofcomment'
                break
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            encoded = pad_sequences([encoded], maxlen=SEQ_LENGTH, truncating='pre')
            yhat_probs = model.predict(encoded, verbose=0)[0]
            yhat = np.random.choice(len(yhat_probs), 1, p=yhat_probs)
            #yhat = model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word,index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            in_text += ' ' + out_word
            if out_word == 'endofcomment':
                break
            else:
                new_comment += ' ' + out_word
        print('-'+new_comment)
        result.append(new_comment)
    print('----------Done---------')
