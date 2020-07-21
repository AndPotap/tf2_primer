import time
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
from utils.vae_funcs import show_reconstructions
from utils.vae_funcs import preprocess_images
from utils.vae_funcs import convert_to_dataset
from models.nets_tf import VAE
from models.opt_tf import OptVAE

seed = 9331
# epochs, batch_size, latent_n = 1, 6, 2
epochs, batch_size, latent_n = 5, 128, 64
lr = 1.e-3
architecture = 'linear'
# architecture = 'nonlinear'
# architecture = 'conv'
tf.random.set_seed(seed)

(x_train, _), (x_test, _) = tfk.datasets.mnist.load_data()
x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)
xval = x_test[:, :, :, 0]

x_train = convert_to_dataset(x_train, batch_size, buffer_size=6 * int(1.e5))
x_test = convert_to_dataset(x_test, batch_size, buffer_size=int(1.e5))

vae = VAE(architecture=architecture, latent_n=latent_n)
optimizer = tfk.optimizers.RMSprop(lr=lr)
vae_opt = OptVAE(vae, optimizer)

print(f'Running on TF version {tf.__version__}')
tf.profiler.experimental.start('./profiler/vae_tf/')
tic = time.time()
mean_time = tfk.metrics.Mean()
for epoch in range(epochs):
    start_time = time.time()
    vae_opt.train_on_epoch(x_train)
    end_time = time.time()

    vae_opt.test_on_epoch(x_test)
    elbo = -vae_opt.loss_tracker.result()
    mean_time(end_time - start_time)
    message = f'Epoch: {epoch + 1:4d} | Test ELBO {elbo:+2.3e} | '
    message += f'{end_time - start_time:4.2f} sec'
    print(message)
toc = time.time()
tf.profiler.experimental.stop()

print(f'It took {toc-tic:4.2f} sec')
print(f'Mean time per epoch is {mean_time.result():4.2f} sec')
show_reconstructions(vae_opt, images=xval)
plt.show()
