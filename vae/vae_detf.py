import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from models.opt_detf import OptVAE
from models.nets_tf import VAE
import tensorflow.compat.v1 as tfc

seed = 9331
# epochs, batch_size, latent_n = 1, 6, 2
epochs, batch_size, latent_n = 10, 128, 64
lr = 1.e-3
# lr = 1.e-4
# architecture = 'linear'
# architecture = 'nonlinear'
architecture = 'conv'
tfc.set_random_seed(seed)

x_train = tfds.load(name='binarized_mnist', split='train', shuffle_files=False)
x_train = x_train.shuffle(buffer_size=60000).batch(batch_size)

x_test = tfds.load(name='binarized_mnist', split='test', shuffle_files=False)
x_test = x_test.shuffle(buffer_size=10000).batch(batch_size)

vae = VAE(architecture, latent_n)
optimizer = tfc.train.RMSPropOptimizer(learning_rate=lr)
vae_opt = OptVAE(vae, optimizer=optimizer)

vae_opt.initialize_session()
print(f'Running on TF version {tf.__version__}')
mean_time = np.zeros(epochs)
tic = time.time()
for epoch in range(epochs):
    t0 = time.time()
    vae_opt.train_on_epoch(x_train)
    t1 = time.time()

    loss = vae_opt.test_on_epoch(x_test)
    message = f'Epoch: {epoch + 1:4d} | Test ELBO {loss:+2.3e} | '
    message += f'{t1 - t0:4.2f} sec'
    mean_time[epoch] = t1 - t0
    print(message)
toc = time.time()
print(f'It took {toc-tic:4.2f} sec')
print(f'Mean time per epoch is {np.mean(mean_time):4.2f} sec')
