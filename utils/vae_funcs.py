import tensorflow as tf
import time
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def convert_to_dataset(v, batch_size, buffer_size=int(1.e5), strategy=None):
    ds = tf.data.Dataset.from_tensor_slices(v)
    ds = ds.shuffle(buffer_size).batch(batch_size)
    if strategy:
        ds = strategy.experimental_distribute_dataset(ds)
    else:
        ds = ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def show_reconstructions(model, images, n_images=5):
    reconstructions = tf.math.sigmoid(model.predict(images[:n_images]))
    reconstructions = np.squeeze(reconstructions)
    plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig('./results/fmnist_recon_' + timestamp + '.png')


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
