__author__ = 'teemu kanstren'

#loads a pre-trained LSTM model and uses it to predict the component for the bug reports from [Qt](http://bugreports.qt.io) bug database.

import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Bidirectional
from keras.layers import Embedding, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, CuDNNLSTM

print(os.listdir("lstm"))

df_2019 = pd.read_csv("bugs2019/reduced.csv", parse_dates=["Created", "Due Date", "Resolved"])

counts = df_2019["comp1"].value_counts()

df_2019 = df_2019[df_2019['comp1'].isin(counts[counts > 3].index)]

df_2019 = df_2019.reset_index()

values = set(df_2019["comp1"].unique())
values.update(df_2019["comp2"].unique())
len(values)

df_vals = pd.DataFrame({
  "value": list(values),
})

df_vals = df_vals.dropna()

from sklearn.preprocessing import LabelEncoder

# encode class values as integers so they work as targets for the prediction algorithm
encoder = LabelEncoder()
encoder.fit(df_vals["value"])

df_2019.dropna(subset=['comp1', "Description"], inplace=True)
df_2019 = df_2019.reset_index()

df_2019["comp1_label"] = encoder.transform(df_2019["comp1"])

features = df_2019["Description"]

def load_word_vectors(glove_dir):
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def tokenize_text(vocab_size, texts, labels, seq_length):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = pad_sequences(sequences, maxlen=seq_length)
    #to_categorical converst vector of class labels (0 to N ints) to binary matrix
    #see keras docs for more info
    y = to_categorical(labels)
    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', y.shape)

    return data, X, y, tokenizer

def train_val_test_split(X, y):

    X_train, X_test_val, y_train,  y_test_val = train_test_split(X, y,
                                                                 test_size=0.2,
                                                                 random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val,
                                                    test_size=0.25,
                                                    random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

def embedding_index_to_matrix(embeddings_index, vocab_size, embedding_dim, word_index):
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(vocab_size, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_model_lstm(vocab_size, embedding_dim, sequence_length, cat_count):
    input = Input(shape=(sequence_length,), name="Input")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length,
                          name="embedding")(input)
    lstm1_bi1 = Bidirectional(LSTM(128, return_sequences=True, name='lstm1'), name="lstm-bi1")(embedding)
    drop1 = Dropout(0.2, name="drop1")(lstm1_bi1)
    lstm2_bi2 = Bidirectional(LSTM(64, return_sequences=False, name='lstm2'), name="lstm-bi2")(drop1)
    drop2 = Dropout(0.2, name="drop2")(lstm2_bi2)
    output = Dense(cat_count, activation='sigmoid', name='sigmoid')(drop2)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_lstm_cuda(vocab_size, embedding_dim, sequence_length, cat_count):
    input = Input(shape=(sequence_length,), name="Input")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length,
                          name="embedding")(input)
    lstm1_bi1 = Bidirectional(CuDNNLSTM(128, return_sequences=True, name='lstm1'), name="lstm-bi1")(embedding)
    drop1 = Dropout(0.2, name="drop1")(lstm1_bi1)
    lstm2_bi2 = Bidirectional(CuDNNLSTM(64, return_sequences=False, name='lstm2'), name="lstm-bi2")(drop1)
    drop2 = Dropout(0.2, name="drop2")(lstm2_bi2)
    output = Dense(cat_count, activation='sigmoid', name='sigmoid')(drop2)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

glove_dir = "lstm"
embeddings_index = load_word_vectors(glove_dir)

data = df_2019["Description"]
vocab_size = 20000
seq_length = 1000
data, X, y, tokenizer = tokenize_text(vocab_size, data, df_2019["comp1_label"], seq_length)

#https://raw.githubusercontent.com/PacktPublishing/Deep-Learning-Quick-Reference/master/Chapter10/newsgroup_classifier_pretrained_word_embeddings.py
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
embedding_dim = 300
embedding_matrix = embedding_index_to_matrix(embeddings_index=embeddings_index,
                                                     vocab_size=vocab_size,
                                                     embedding_dim=embedding_dim,
                                                     word_index=tokenizer.word_index)

model = build_model_lstm(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    sequence_length=seq_length,
#                    embedding_matrix=embedding_matrix,
                   cat_count=168) #TODO: get this number from encoder

model.load_weights('lstm/model-lstm-11-epoch.hdf5')

#model = build_model(vocab_size=vocab_size,
#                    embedding_dim=embedding_dim,
#                    sequence_length=seq_length,
#                    embedding_matrix=embedding_matrix,
#                   cat_count=168) #TODO: get this number from encoder

#model.fit(X_train, y_train,
#          batch_size=128,
#          epochs=15,
#          validation_data=(X_val, y_val))

#model.save("newsgroup_model_word_embedding.h5")

#score, acc = model.evaluate(x=X_test,
#                            y=y_test,
#                            batch_size=128)
#print('Test loss:', score)
#print('Test accuracy:', acc)

#which integer matches which textual label/component name
le_id_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

def predict(bug_description, seq_length):
    #texts_to_sequences vs text_to_word_sequence?
    sequences = tokenizer.texts_to_sequences([bug_description])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = pad_sequences(sequences, maxlen=seq_length)

    probs = model.predict(X)
    result = []
    for idx in range(probs.shape[1]):
        name = le_id_mapping[idx]
        prob = (probs[0, idx]*100)
        prob_str = "%.2f%%" % prob
        #print(name, ":", prob_str)
        result.append((name, prob))
    return result

def predict_file():
    with open('example_bug.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')

    result = predict(data, seq_length)
    sorted_by_second = sorted(result, key=lambda x: x[1])
    return sorted_by_second

probs = predict_file()
#print(probs)
for prob in probs:
    line = "{}: {:3.4f}".format(prob[0], prob[1])
    print(line)


