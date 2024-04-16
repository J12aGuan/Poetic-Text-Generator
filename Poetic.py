import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')    #Get the file from this path

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()     #Open path and put it in read binary mode

text = text[300000:800000]

characters = sorted(set(text))      #Get all the unique characters in the text and sort them

char_to_index = dict((c, i) for i, c in enumerate(characters))      #Create a dictionary that stores all the c(characters) and i(index)
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40     #Each sequence contains 40 characters
STEP_SIZE = 3       #Each time, sequence will add 3 new characters and delete 3 old characters

sentences = []
next_characters = []

# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])       #Store from i to i + SEQ_LENGTH in text into sentences
#     next_characters.append(text[i + SEQ_LENGTH])    #Store in the next correct character

# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)


# for i, sentence in enumerate(sentences):
#     for t, character in enumerate(sentence):
#         x[i, t, char_to_index[character]] = 1

#     y[i, char_to_index[next_characters[i]]] = 1


# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

# model.fit(x, y, batch_size = 256, epochs = 4)
# model.save('textgenerator.model')
    
model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose = 0)[0]
        next_index = sample(predictions, temperature)
        next_characters = index_to_char[next_index]

        generated += next_characters
        sentence = sentence[1:] + next_characters
    return generated

print(generate_text(4000, 0.6))