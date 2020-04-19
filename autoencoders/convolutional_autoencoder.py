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
        self.final = self.normalized.reshape(-1, 28, 28, 1)

    def train_val_split(self, val_size):
        if val_size > 1 or val_size <= 0:
            raise SyntaxError("Validation size should be a float between 0 and 1")
        perm = list(np.random.permutation(self.final.shape[0]))
        split_index = int(self.final.shape[0] * val_size)
        train_idxs = perm[:split_index]
        test_idxs = perm[split_index:]
        self.train = self.final[train_idxs]
        self.val = self.final[test_idxs]

class encoder:
    def __init__(self, input):
        self.input = tf.keras.Input(shape=[28, 28, 1], dtype=tf.float32, name='encoder_input')
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(self.input)
        self.max1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='max1')(self.conv1)
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(self.max1)
        self.max2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='max2')(self.conv2)
        self.conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3')(self.max2)
        self.max3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same', name='max3')(self.conv3)

class decoder:
    def __init__(self, input):
        self.input = tf.keras.Input(shape=[4, 4, 8], dtype=tf.float32, name='decoder_input')
        self.conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv4')(self.input)
        self.up4 = tf.keras.layers.UpSampling2D((2, 2), name='up4')(self.conv4)
        self.conv5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv5')(self.up4)
        self.up5 = tf.keras.layers.UpSampling2D((2, 2), name='up5')(self.conv5)
        self.conv6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv6')(self.up5)
        self.up6 = tf.keras.layers.UpSampling2D((2, 2), name='up6')(self.conv6)
        self.outputs = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='output')(self.up6)

(x_train, _ ), (x_test, _ ) = mnist.load_data()

tf.keras.backend.clear_session()

input = data(x_train, "x_train")

input.train_val_split(0.2)

input.train.shape
input.val.shape

encod = encoder(input.train)
decod = decoder(encod.max3)

encoder_model = tf.keras.Model(encod.input, encod.max3, name='Encoder')
decoder_model = tf.keras.Model(decod.input, decod.outputs, name='Decoder')

autoencoder = tf.keras.Model(encod.input, decoder_model(encoder_model(encod.input)), name="Autoencoder")

encoder_model.summary()
decoder_model.summary()
autoencoder.summary()

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5,  min_delta=0.01, restore_best_weights=True)

history = autoencoder.fit(x=input.train, y=input.train, epochs=1, shuffle=True, batch_size=256, validation_data=(input.val, input.val), callbacks=[es])


test = data(x_test, "x_test")
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
