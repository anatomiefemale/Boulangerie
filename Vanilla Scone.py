# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation

# Provide sample text to train the model
"""
The model will learn from this input text and try to 
generate text after learning.
"""
text = "hello world"
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# Prepare dataset
"""
Convert characters to integers and 
create sequences that the model can learn from.
"""
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

X = np.reshape(dataX, (len(dataX), seq_length, 1))
X = X / float(len(chars))
y = tf.keras.utils.to_categorical(dataY)

# Define a simple LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile and train the model on the prepared data
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=300, batch_size=64)

# Define function to generate text
"""
Use the model to generate text based on a seed input.
"""
def generate_text(model, seed_text, n_vocab, char_to_int, 
                  int_to_char, length=100):
    pattern = [char_to_int[char] for char in seed_text]
    text = seed_text
    for i in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        text += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return text

# Generate some text
int_to_char = dict((i, c) for i, c in enumerate(chars))
print(generate_text(model, "hel", len(chars), char_to_int, int_to_char))