from lstm import *


MAXLEN = 40
STEP = 3 
BATCH_SIZE = 128
EPOCHS = 60


corpus = import_dir()
train, dev, test = split_data(corpus)
vocab = vocabulary_for_corpus(train)
print(len(vocab))
ind_char, char_ind = make_dics(vocab)
"""
sequences, next_chars = make_all_sequences(train)
trainx,trainy = vectorize(sequences, next_chars, vocab, char_ind)

model = make_model(vocab)
fit_model(model)

"""
