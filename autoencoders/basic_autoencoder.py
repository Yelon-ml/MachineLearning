import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random

class data:
    def __init__(self, input, name):
        self.name = name
        self.normalized = input.astype("float32") / 255.
        self.final = self.normalized.reshape(input.shape[0], np.prod(input.shape[1:]))

    def train_val_split(self, val_size):
        if val_size > 1 or val_size <= 0:
            raise SyntaxError("Validation size should be a float between 0 and 1")
        perm = list(np.random.permutation(self.final.shape[0]))
        split_index = int(self.final.shape[0] * val_size)
        train_idxs = perm[:split_index]
        test_idxs = perm[split_index:]
        self.train = self.final[train_idxs]
        self.val = self.final[test_idxs]

class Model:
    def __init__(self, input):
        self.input = tf.keras.Input(shape=[784], dtype=tf.float32, name='input')
        self.layer1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(), dtype=tf.float32, name="layer1")(self.input)
        self.layer2 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu, activity_regularizer=tf.keras.regularizers.l1(10e-5), kernel_initializer=tf.keras.initializers.he_normal(), dtype=tf.float32, name="layer2")(self.layer1)
        self.layer3 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_normal(), dtype=tf.float32, name="layer3")(self.layer2)
        self.outputs = tf.keras.layers.Dense(units=784, activation='sigmoid', name='output')(self.layer3)

(x_train, _ ), (x_test, _ ) = mnist.load_data()

tf.keras.backend.clear_session()

input = data(x_train, "mnist")

model = Model(input.train)

autoencoder = tf.keras.Model(encod.input, decoder_model(encoder_model(encod.input)), name="Autoencoder")

autoencoder.summary()

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True)

history = autoencoder.fit(x=input.train, y=input.train, epochs=5, shuffle=True, batch_size=256, validation_data=(input.val, input.val), callbacks=[es])

test = data(x_test, "mnist_test")
pred_imgs = autoencoder.predict(test.final)

def reals_and_predictions(n):
    len = test.final.shape[0]
    perm = np.random.permutation(len)
    perm = perm[:n]

    ax_s = []
    for i in range(n):
        ax_s.append(str("ax") + str(i))

    f, ax_s = plt.subplots(1, n, figsize=(20,2))
    f.suptitle("Test Images", fontsize=16)
    for idx, ax in enumerate(ax_s):
        ax.imshow(test.final[perm[idx]].reshape(28, 28), cmap='gray')
        ax.axis('off')

    g, ax_s = plt.subplots(1, n, figsize=(20,2))
    g.suptitle("Reconstructed Images", fontsize=16)
    for idx, ax in enumerate(ax_s):
        ax.imshow(pred_imgs[perm[idx]].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

reals_and_predictions(10)
