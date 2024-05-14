# Import modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Data generation parameters
num_patients = 1000
max_seq_length = 10
vocab_size = 50  # Number of possible symptoms

np.random.seed(0)
patient_data = np.random.randint(1, vocab_size, size=(num_patients, max_seq_length))
print(patient_data)
"""
[[45 48  1 ... 20 22 37]
 [24  7 25 ... 40 24 47]
 [25 18 38 ... 21 17  6]
 ...
 [26 46  4 ... 11  8  5]
 [ 7  5 41 ... 36 16 23]
 [47 11 37 ...  5 41 45]]
"""
patient_labels = np.random.randint(2, size=(num_patients, 1))
print(patient_labels)
"""
 [1]
 [0]
 [1]
 [1]
 [1]
 [0]...etc.
"""
# Train-test split
split = int(num_patients * 0.8)
train_data, test_data = patient_data[:split], patient_data[split:]
train_labels, test_labels = patient_labels[:split], patient_labels[split:]

# Define the positional encoding function
def get_positional_encoding(max_seq_length, d_model):
    positional_enc = np.zeros((max_seq_length, d_model))
    position = np.arange(0, max_seq_length, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    positional_enc[:, 0::2] = np.sin(position * div_term)
    positional_enc[:, 1::2] = np.cos(position * div_term)

    positional_enc = tf.cast(positional_enc, dtype=tf.float32)
    return positional_enc

# Define the scaled dot-product attention function
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

class TransformerEncoder(Model):
    def __init__(self, vocab_size, num_heads, d_model):
        super(TransformerEncoder, self).__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(100, d_model)  # Adjust max sequence length if necessary
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.pooling = GlobalAveragePooling1D()
        self.final = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embed(x)
        x += self.pos_encoding[:tf.shape(x)[1], :]
        x = self.attention(x, x, x)  # Mask is omitted unless you specifically need it
        x = self.pooling(x)
        return self.final(x)

# Model setup parameters
num_heads = 4
d_model = 128  # Dimensionality of the embedding

# Initialize and compile the model
model = TransformerEncoder(vocab_size=vocab_size, num_heads=num_heads, d_model=d_model)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_data=(test_data, test_labels))

# Make predictions
predictions = model.predict(test_data)
print("Sample predictions:", predictions[:5])
"""
Sample predictions: [[0.59326303]
 [0.51481295]
 [0.8828845 ]
 [0.34554717]
 [0.35884818]]
"""