# Step 1 - creating utility functions
import csv
import pandas as pd
import emoji 
from sklearn.metrics import confusion_matrix

# 1.1 - softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 1.2 - func. for reading training data
def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []
    
    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])
    
    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)
    
    return X, Y

# 1.3 - func. to load pre-trained word embeddings
# which were downloaded from this link: 
# https://www.kaggle.com/datasets/watts2/glove6b50dtxt?resource=download
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
        
    return words_to_index, index_to_words, word_to_vec_map

# 1.4 - func. which converts Y outputs to one-hot vector
# Y - output vector, C - number of possible outcomes
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# 1.5 - the dictionary of the emojis that we'll use in 
# our softmax layer
emoji_dictionary = {
    "0": "\u2764\uFE0F",
    "1": ":baseball:",
    "2": ":smile:",
    "3": ":disappointed:",
    "4": ":fork_and_knife:"
}

# 1.6 - func. that converts a string label to a real emoji
# with emoji package
def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], language='alias')

# 2 - Importing necessary libraries & layers from Keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform

# 3 - Importing sets & weights

# 3.1 - loading the training & testing sets
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tests.csv')

max_len = len(max(X_train, key=len).split())

# 3.2 - loading the pretrained weights
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# 4 - Creating a model
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):
        sentence_words = X[i].lower().split()
        
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j = j + 1
                
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_size = len(word_to_index) + 1
    any_word =list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]
    
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]
        
    embedding_layer = Embedding(vocab_size, emb_dim, trainable=False)
    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def NetworkModel(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentence_indices)
    
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    
    model = Model(sentence_indices, X)
    
    return model

model = NetworkModel((max_len, ), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)
Y_train_onehot = convert_to_one_hot(Y_train, C = 5)

model.fit(X_train_indices, Y_train_onehot, epochs=50, batch_size=32, shuffle=True)


# Testing the model 
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
Y_test_onehot = convert_to_one_hot(Y_test, C = 5)

loss, acc = model.evaluate(X_test_indices, Y_test_onehot)
print('\nTest accuracy = ', acc)

# Showing mislabeled examples
y_test_onehot = np.eye(5)[Y_test.reshape(-1)]
x_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
pred = model.predict(x_test_indices)

for i in range(len(X_test)):
    x = x_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:' + label_to_emoji(Y_test[i]) + ' prediction: ' + X_test[i] + label_to_emoji(num).strip())

# Testing the model on custom sentence
x_test = np.array(["not feeling happy"])
x_test_indices = sentences_to_indices(x_test, word_to_index, max_len)
print(x_test[0] + ' ' +  label_to_emoji(np.argmax(model.predict(x_test_indices))))