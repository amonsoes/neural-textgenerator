import os
import numpy as np
import pickle
import random
import sys

from keras.preprocessing import sequence
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D


# ========= macros ============


MAXLEN = 40 # we're grabbing 40-character-blocks as we're going through the corpus, and the model has to guess the 41st character
STEP = 3 # move in 3-steps as you're going through the character block
BATCH_SIZE = 128
EPOCHS = 60


# ========= i/o ===============


def import_dir(path="./files/"):
    ls = []
    for _,_, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                ls.append(import_file(path+file))
    return ls
                
def import_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def export_model(model, path="./model"):
    with open(path, "wb") as b:
        pickle.dump(model)

def import_model(path="./model"):
    with open(path, "rb") as b:
        model = pickle.load(b)
    return model

def split_data(ls):
    train_size = int(len(ls) * 0.7)
    dev_size = int(len(ls) * 0.8)
    train = ls[:train_size]
    dev = ls[train_size:dev_size]
    test = ls[dev_size:]
    return train, dev, test


# ======== preprocessing =======
    

def vocabulary_for_text(text):
    return sorted(list(set(text)))

def vocabulary_for_corpus(ls):
    vocab = []
    for text in ls:
        vocab.extend(vocabulary_for_text(text))
    return sorted(vocab)

def make_dics(vocab):
    return {ind:i for ind, i in enumerate(vocab)}, {i:ind for ind, i in enumerate(vocab)}

def make_sequences(text):
    sents = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, STEP):
        sents.append(text[i:i+MAXLEN])
        next_chars.append(text[i+MAXLEN])
    return sents, next_chars

def make_all_sequences(corpus):
    ls_sents = []
    ls_chars = []
    for text in corpus:
        sents, chars = make_sequences(text)
        ls_sents.extend(sents)
        ls_chars.extend(chars)
    return ls_sents, ls_chars

def vectorize(sents, next_chars, vocab, char_ind):
    x = np.zeros((len(sents), MAXLEN, len(vocab)), dtype=np.bool) # sents = wieviele Sätze, MAXLEN = Wieviele character in den Sätzen, VOCAB = für One-Hot encoding der character
    y = np.zeros((len(sents), len(vocab)), dtype=np.bool)
    for ind, sent in enumerate(sents):
        for sub, char in enumerate(sent):
            x[ind, sub, char_ind[char]] = 1 # for each sent, each char, make a 1 at the char's indice of the vocabulry
            y[ind, char_ind[next_chars[ind]]] # for each sent, mark in the one hot vector the indice of the char that follows on the sent
    return x, y


# ====== model ==========


def make_model(vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape=(MAXLEN, len(vocab))))
    model.add(Dense(len(vocab), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    return model
    

def sample(preds, temperature=1.0):
    """
    preds: output neurons of NN
    temperature : introduce randomness to model, to make the tg less conservative
    """
    preds = np.log(np.asarray(preds).astype("float64")) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/ np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def on_epoch_end(epoch, _, text, vocab, model, indices_char):
    # Function invoked at end of each epoch. Prints generated text.
    print("****************************************************************************")
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - MAXLEN - 1)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('----- temperature:', temperature)

        generated = ''
        sentence = text[start_index: start_index + MAXLEN] # generate sequence from text with MAXLEN
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, MAXLEN, len(vocab))) # dim 1 x 40 X 60
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1. # for each char in the sampled sent, mark the vectors

            preds = model.predict(x_pred, verbose=0)[0] # pass the vector to the NN make a prediction
            next_index = sample(preds, temperature) # pass the predictions to the sample function
            next_char = indices_char[next_index] # to reverse the one-hot-encoding process

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def fit_model(model, x, y):
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])