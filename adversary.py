#%%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

#%%

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

#%%

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

#%%

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

#%%

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

vae_ckpt = "weights/vae.ckpt"

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
try:
    vae.load_weights(vae_ckpt)
except Exception as e:
    print(e)
    vae.fit(x_train, epochs=30, batch_size=32)
    vae.save_weights(vae_ckpt)

#%% 

z = vae.encoder.predict(x_train)
#%%
x_out = vae.decoder.predict(z[0])

#%%

i = 5
plt.imshow(x_train[i].reshape((28,28)))
plt.imshow(x_out[i].reshape((28,28)))

# %%

num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

#%%

batch_size = 128
epochs = 15

model_ckpt = "weights/model.ckpt"

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
try:
    model.load_weights(model_ckpt)
except Exception as e:
    print(e)
    model.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save_weights(model_ckpt)

#%%

score = model.evaluate(x_test, y_test_cat, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%

class AdversaryLayer(keras.layers.Layer):
    def __init__(self, vae, regularization, **kwargs):
        super(AdversaryLayer, self).__init__(**kwargs)
        self.vae_model = vae
        self.regularization_fac = regularization

    def call(self, inputs, training = False):
        z = self.vae_model.encoder(inputs)
        # if training:

        perturbed = self.vae_model.decoder(z[0])
        delta = perturbed - inputs
        regularization_loss = self.regularization_fac * tf.reduce_mean(tf.square(delta))
        self.add_loss(regularization_loss)
        return perturbed

class Adversary2(keras.Model):
    def __init__(self, classifier, adv, **kwargs):
        super(Adversary2, self).__init__(**kwargs)
        self.classifier_model = classifier
        self.adv = adv
        for layer in self.classifier_model.layers:
            layer.trainable = False

    def call(self, inputs):
        perturbed = self.adv(inputs)
        y_pred = self.classifier_model(perturbed)
        return y_pred

# %%

y_from, y_to = 6, 5
zeros = y_train == y_from
x_train_dec = x_train[zeros] 
y_train_dec_cat = keras.utils.to_categorical(y_to * np.ones((len(x_train_dec),)), num_classes)

adv_ckpt = "weights/adv.ckpt"

adv = Adversary2(model, AdversaryLayer(vae, 5e2))

adv.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
try:
    adv.load_weights(adv_ckpt)
except Exception as e:
    print(e)
    adv.fit(x_train_dec, y_train_dec_cat, epochs=200, batch_size=32)
    adv.save_weights(adv_ckpt)

# %%

z = adv.adv.vae_model.encoder.predict(x_train_dec)
x_dec = adv.adv.vae_model.decoder.predict(z[0])
delta = x_dec - x_train_dec
dec_classes = model(x_dec)
orig_classes = model(x_train_dec)

#%%

i = 1
plt.imshow(np.reshape(x_train_dec[i], (28,28)),  vmin=0, vmax=1)
#%%
plt.imshow(np.reshape(abs(delta[i]), (28,28)),  vmin=0, vmax=1)
#%%
plt.imshow(np.reshape(x_dec[i], (28,28)),  vmin=0, vmax=1)
#%% Modified image is classified as 5 with 87.35% confidence
print(dec_classes[i])
print(np.argmax(dec_classes[i]), ':', 100 * np.max(dec_classes[i]))
#%% Original image is classified as 6 with 99.98% confidence
print(orig_classes[i])
print(np.argmax(orig_classes[i]), ':', 100 * np.max(orig_classes[i]))
#%%