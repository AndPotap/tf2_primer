import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.layers import Conv2DTranspose as Conv2DT
from models.opt_tf import compute_gauss_kl


class GAN(tf.keras.models.Model):

    def __init__(self, architecture, ngf, ndf, latent_n, **kwargs):
        super().__init__(**kwargs)
        self.ngf = ngf
        self.ndf = ndf
        self.latent_n = latent_n
        self.architecture = architecture
        self.create_nets()

    def create_nets(self):
        if self.architecture == 'mnist':
            self.shape = (28, 28, 1)
            self.create_discriminator()
            self.create_generator()
        elif self.architecture == 'celeb_a':
            self.shape = (64, 64, 3)
            self.create_celeba_discriminator()
            self.create_celeba_generator()
        else:
            NotImplementedError

    def create_celeba_generator(self):
        inputs = Input(shape=(self.latent_n,))
        hidden = Reshape(target_shape=(1, 1, self.latent_n))(inputs)
        hidden = Conv2DT(filters=self.ngf * 8, kernel_size=4,
                         strides=1, use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)

        hidden = Conv2DT(filters=self.ngf * 4, kernel_size=4,
                         strides=2, padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)

        hidden = Conv2DT(filters=self.ngf * 2, kernel_size=4,
                         strides=2, padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)

        hidden = Conv2DT(filters=self.ngf, kernel_size=4, strides=2,
                         padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = ReLU()(hidden)

        outputs = Conv2DT(filters=self.shape[2], kernel_size=4, strides=2, padding='same',
                          use_bias=False, activation='tanh')(hidden)
        self.gen = Model(inputs=[inputs], outputs=[outputs])

    def create_celeba_discriminator(self):
        inputs = Input(shape=self.shape)
        hidden = Conv2D(filters=self.ndf, kernel_size=4, strides=2,
                        padding='same', use_bias=False)(inputs)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Conv2D(filters=self.ndf * 2, kernel_size=4,
                        strides=2, padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Conv2D(filters=self.ndf * 4, kernel_size=4,
                        strides=2, padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Conv2D(filters=self.ndf * 8, kernel_size=4,
                        strides=2, padding='same', use_bias=False)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        outputs = Conv2D(filters=1, kernel_size=4, strides=1,
                         use_bias=False, activation='sigmoid')(hidden)
        self.dis = Model(inputs=[inputs], outputs=[outputs])

    def create_generator(self):
        model = tf.keras.Sequential()
        model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DT(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DT(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DT(1, (5, 5), strides=(2, 2),
                          padding='same', use_bias=False, activation='tanh'))

        self.gen = model

    def create_discriminator(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=[28, 28, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))
        self.dis = model


class VAE(tf.keras.models.Model):
    def __init__(self, architecture, latent_n, **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.latent_n = latent_n
        self.shape = (28, 28, 1)
        self.select_architecture()

    def select_architecture(self):
        if self.architecture == 'linear':
            self.create_linear_encoder()
            self.create_linear_decoder()
        elif self.architecture == 'nonlinear':
            self.create_nonlinear_encoder()
            self.create_nonlinear_decoder()
        elif self.architecture == 'conv':
            self.create_conv_encoder()
            self.create_conv_decoder()
        self.create_vae()

    def call(self, inputs):
        return self.vae(inputs)

    def create_vae(self):
        x = Input(self.shape)
        mean, log_var = self.encoder(x)
        z = GaussianSampling()([mean, log_var])
        x_hat = self.decoder(z)
        kl = compute_gauss_kl(mean, log_var)
        self.vae = Model(inputs=[x], outputs=[x_hat])
        self.vae.add_loss(tf.math.reduce_mean(kl) / 784.)

    def create_conv_encoder(self):
        inputs = Input(shape=self.shape)
        h = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(inputs)
        h = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(h)
        h = Flatten()(h)
        mean = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        log_var = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        self.encoder = Model(inputs=[inputs], outputs=[mean, log_var])

    def create_conv_decoder(self):
        z = Input(shape=self.latent_n)
        h = Dense(units=7 * 7 * 32, activation='relu')(z)
        h = Reshape(target_shape=(7, 7, 32))(h)
        h = Conv2DT(filters=64, kernel_size=3, strides=2,
                    padding='same', activation='relu')(h)
        h = Conv2DT(filters=32, kernel_size=3, strides=2,
                    padding='same', activation='relu')(h)
        x_hat = Conv2DT(filters=1, kernel_size=3, strides=1,
                        padding='same', activation=None, dtype=tf.float32)(h)
        self.decoder = Model(inputs=[z], outputs=[x_hat])

    def create_nonlinear_encoder(self):
        inputs = Input(shape=self.shape)
        flattened = Flatten()(inputs)
        h = Dense(units=400, activation='relu')(flattened)
        h = Dense(units=240, activation='relu')(h)
        mean = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        log_var = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        self.encoder = Model(inputs=[inputs], outputs=[mean, log_var])

    def create_nonlinear_decoder(self):
        z = Input(shape=self.latent_n)
        h = Dense(units=240, activation='relu')(z)
        h = Dense(units=400, activation='relu')(z)
        h = Dense(units=28 * 28, activation=None)(h)
        x_hat = Reshape(self.shape, dtype=tf.float32)(h)
        self.decoder = Model(inputs=[z], outputs=[x_hat])

    def create_linear_encoder(self):
        inputs = Input(shape=self.shape)
        flattened = Flatten()(inputs)
        h = Dense(units=400, activation=None)(flattened)
        mean = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        log_var = Dense(units=self.latent_n, activation=None, dtype=tf.float32)(h)
        self.encoder = Model(inputs=[inputs], outputs=[mean, log_var])

    def create_linear_decoder(self):
        z = Input(shape=self.latent_n)
        h = Dense(units=400, activation=None)(z)
        h = Dense(units=28 * 28, activation=None)(h)
        x_hat = Reshape(self.shape, dtype=tf.float32)(h)
        self.decoder = Model(inputs=[z], outputs=[x_hat])


class GaussianSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        stddev = tf.math.exp(0.5 * log_var)
        epsilon = tf.random.normal(shape=tf.shape(mean), dtype=mean.dtype)
        z = mean + epsilon * stddev
        return z
