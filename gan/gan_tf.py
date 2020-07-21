import tensorflow as tf
import tensorflow.keras as tfk
from utils.load_data import ProcessData
import time
from models.nets_tf import GAN
from models.opt_tf import OptGAN
from tensorflow.python.profiler import profiler_v2 as profiler

run_with_sample = False
seed = 9331
tf.random.set_seed(seed)
buffer_size = 2 * int(1.e4)
batch_size = 256
epochs = 2
latent_n = 100
ngf = 64
ndf = 64
lr = 2 * 1.e-4
architecture = 'celeb_a'

processor = ProcessData(dataset_name='celeb_a', image_shape=(64, 64, 3),
                        buffer_size=buffer_size, run_with_sample=run_with_sample)
ds, *_ = processor.generate_train_and_test_partitions(batch_size, epochs)
train_ds, _ = ds

nets = GAN(architecture, ngf, ndf, latent_n)
gen_optimizer = tf.keras.optimizers.Adam(lr)
dis_optimizer = tf.keras.optimizers.Adam(lr)
gan_opt = OptGAN(nets, gen_optimizer, dis_optimizer, batch_size, latent_n)

print(f'Running on TF version {tf.__version__}')
tic = time.time()
# tf.profiler.experimental.start('./profiler/gan_tf/')
profiler.warmup()
profiler.start('./profiler/gan_tf/')
mean_time = tfk.metrics.Mean()
for epoch in range(epochs):
    start_time = time.time()
    gan_opt.train_on_epoch(train_ds)
    end_time = time.time()

    mean_time(end_time - start_time)
    message = f'Epoch: {epoch + 1:4d} |'
    message += f'{end_time - start_time:4.2f} sec'
    print(message)
toc = time.time()
# tf.profiler.experimental.stop()
profiler.stop()

print(f'It took {toc-tic:4.2f} sec')
print(f'Mean time per epoch is {mean_time.result():4.2f} sec')
noise = tf.random.normal([1, 100])
generated_image = nets.gen(noise, training=False)
# plt.show()
