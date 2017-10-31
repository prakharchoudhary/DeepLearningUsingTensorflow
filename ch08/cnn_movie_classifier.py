import os
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Conv1D, GlobalMaxPooling1D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train,
                                 maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test,
                                maxlen=max_review_length)

# transforming data via word embedding technique
embedding_vector_length = 32
model = Sequential()

# stacking layers
model.add(Embedding(top_words, embedding_vector_length,
                    input_length=max_review_length))
model.add(Conv1D(padding="same", activation="relu", kernel_size=3, filters=32))
# model.add(GlobalMaxPooling1D())
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# fitting the model
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=3,
          batch_size=64)

# model performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))