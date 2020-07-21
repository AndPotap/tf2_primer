import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.vae_funcs import rounded_accuracy, show_reconstructions
from models.nets_tf import VAE
from models.opt_tf import compute_xe_loss

seed = 42
epochs, batch_size, latent_n = 25, 128, 10
tf.random.set_seed(seed)
np.random.seed(seed)
# architecture = 'linear'
architecture = 'conv'

(xtrain_full, _), (xtest, _) = tf.keras.datasets.fashion_mnist.load_data()
xtrain_full = xtrain_full.astype(np.float32) / 255
xtest = xtest.astype(np.float32) / 255
xtrain, xval = xtrain_full[:-5000], xtrain_full[-5000:]
xtrain, xval = np.expand_dims(xtrain, axis=-1), np.expand_dims(xval, axis=-1)
# ytrain, yval = ytrain_full[:-5000], ytrain_full[-5000:]

vae = VAE(architecture=architecture, latent_n=latent_n)
vae.compile(loss=compute_xe_loss, optimizer="rmsprop", metrics=[rounded_accuracy])
tic = time.time()
history = vae.fit(xtrain, xtrain, epochs=epochs,
                  batch_size=batch_size, validation_data=(xval, xval))
toc = time.time()
print(f'It took {toc-tic:4.2f} sec')
show_reconstructions(vae, images=xval)
plt.show()
