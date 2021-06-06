#%%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import ipywidgets as widgets

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%

"""
First, a variational autoencoder (VAE) is trained.
It consists of an encoder and a decoder: 
The encoder reduces the input image to a so-called latent vector, and the encoder 
converts this latent vector back to an approximation of the original image.

The encoder consists of convolutional layers and then dense layers to represent 
the mean and variance of the distribution of the latent vectors. A vector is then 
sampled from this distribution.  
The Kullback–Leibler (KL) divergence is used as a regularization term, which helps 
to make the output of the encoder as close as possible to normally distributed.

The decoder takes the latent vector (output of the encoder) as input and 
consists of the inverse/transposed layers as the encoder (e.g. Conv2DTranspose 
instead of Conv2D and upsampling instead of pooling). The output of this decoder
has the same dimensions as the input of the encoder. The loss function then 
measures how well the output of the decoder approximates the input of the encoder.
"""

class KLLossSampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample the latent vector z, and adds KL loss"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(1e-1 * kl_loss)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#%%

"""
Build the encoder model
"""

latent_dim = 3

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = KLLossSampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, z, name="encoder")
encoder.summary()

#%%

"""
Build the decoder model
"""

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

#%%

"""
Build the variational autoencoder by composition of the encoder and decoder
"""

vae_input = keras.layers.Input(shape=(28, 28, 1), name="vae_input")
vae_encoder_output = encoder(vae_input)
vae_decoder_output = decoder(vae_encoder_output)
vae = keras.models.Model(vae_input, vae_decoder_output, name="vae")

#%%

"""
Load the training data
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
x_test = np.expand_dims(x_test, -1).astype("float32") / 255


"""
Define the loss function, compile and train the model
"""

loss = lambda a, b: tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(a, b), axis=(1, 2)))
vae.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=loss)

vae_ckpt = "weights/vae2.ckpt"
try:
    vae.load_weights(vae_ckpt)
except Exception as e:
    print(e)
    vae.fit(x_train, x_train, epochs=30, batch_size=128)
    vae.save_weights(vae_ckpt)

#%% 

"""
Apply the autoencoder to the training data and visualize some of the results
"""

x_out = vae.predict(x_train)

def show_vae_results(x_in, x_out):
    def f(i):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x_in[i].reshape((28,28)))
        ax2.imshow(x_out[i].reshape((28,28)))
    return f

interact(show_vae_results(x_train, x_out), i=widgets.IntSlider(min=0, max=100, step=1, value=0))

#%% 

"""
Apply the autoencoder to the training data and visualize some of the results
"""

x_out_test = vae.predict(x_test)
interact(show_vae_results(x_test, x_out_test), i=widgets.IntSlider(min=0, max=100, step=1, value=0))

# %%

"""
Next, we'll train a classifier for the dataset, based on the following model:
https://keras.io/examples/vision/mnist_convnet/
"""

num_classes = 10
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

classifier = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
classifier.summary()

#%%

batch_size = 128
epochs = 15

model_ckpt = "weights/model.ckpt"

classifier.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
try:
    classifier.load_weights(model_ckpt)
except Exception as e:
    print(e)
    classifier.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    classifier.save_weights(model_ckpt)

#%%

score = classifier.evaluate(x_test, y_test_cat, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%

"""
The following is the architecture for the adversary attack:
The input image is first passed through the VAE, which is trainable, 
the output of the VAE is the perturbed image.
The difference between the input image and the perturbed image is penalized 
using the mean squared error, this is the regularization loss.
Finally, this perturbed image is given to the classifier, which is not
trainable. The output of the classifier is then compared to the given 
label (which can be a correct label or a deceptive label), where a 
categorical cross entropy loss function is used.

The classifier is available as a white box model, so its weights and 
architecture are known, and we can compute derivatives of the loss function
with respect to the weights of the VAE.
"""

class AdversaryLayer(layers.Layer):
    def __init__(self, vae, regularization, **kwargs):
        super(AdversaryLayer, self).__init__(**kwargs)
        self.vae = vae
        self.regularization_fac = regularization

    def call(self, inputs):
        perturbed = self.vae(inputs)
        delta = perturbed - inputs
        regularization_loss = self.regularization_fac * tf.reduce_mean(tf.square(delta))
        self.add_loss(regularization_loss)
        return perturbed

class Adversary(keras.Model):
    def __init__(self, classifier, adv, **kwargs):
        super(Adversary, self).__init__(**kwargs)
        self.classifier = classifier
        self.adv = adv
        for layer in self.classifier.layers:
            layer.trainable = False

    def call(self, inputs):
        perturbed = self.adv(inputs)
        y_pred = self.classifier(perturbed)
        return y_pred

# %%

"""
All labels “6” in the training data are now replaced with by deceptive labels “5”,
and the combined model (adversarial VAE + classifier) is trained on this training 
data. As mentioned before, only the weights of the VAE are trained, the classifier 
doesn't change.
"""

y_from, y_to = 6, 5
sixes = y_train == y_from

train_all = False
if not train_all:
    x_train_dec = x_train[sixes] 
    y_train_dec_cat = keras.utils.to_categorical(y_to * np.ones((len(x_train_dec),)), num_classes)
else:
    x_train_dec = x_train 
    y_train_dec = y_train.copy()
    y_train_dec[sixes] = y_to
    y_train_dec_cat = keras.utils.to_categorical(y_train_dec, num_classes)

adv_ckpt = "weights/adv2.ckpt"

adv = Adversary(classifier, AdversaryLayer(vae, 5e2))

adv.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
try:
    adv.load_weights(adv_ckpt)
except Exception as e:
    print(e)
    adv.fit(x_train_dec, y_train_dec_cat, epochs=200, batch_size=128)
    adv.save_weights(adv_ckpt)

# %%
"""
Now display a selection of the sixes and see if we were able to fool the classifier.
"""

x_dec = vae.predict(x_train_dec)
delta = x_dec - x_train_dec
dec_classes = classifier(x_dec)
orig_classes = classifier(x_train_dec)

#%%

"""
The perturbed image still looks like a “6”, but the classifier classifies it as a “5”
with over 80% confidence, even though the original image was classified as a “6” with
100% confidence.
"""

i = 0
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.imshow(np.reshape(x_train_dec[i], (28,28)),  vmin=0, vmax=1)
ax0.set_title('Original')
ax1.imshow(np.reshape(abs(delta[i]), (28,28)),  vmin=0, vmax=1)
ax1.set_title('Delta')
ax2.imshow(np.reshape(x_dec[i], (28,28)),  vmin=0, vmax=1)
ax2.set_title('Perturbed')
print("Original")
print(orig_classes[i])
print(np.argmax(orig_classes[i]), ':', 100 * np.max(orig_classes[i]))
print("Perturbed")
print(dec_classes[i])
print(np.argmax(dec_classes[i]), ':', 100 * np.max(dec_classes[i]))

# %%
