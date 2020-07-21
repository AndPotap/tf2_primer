import time
import tensorflow as tf
from tensorflow_probability import distributions as tfpd


class OptGAN:
    def __init__(self, nets, gen_optimizer, dis_optimizer, batch_size, latent_n):
        self.nets = nets
        self.loss_tracker = tf.keras.metrics.Mean()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.batch_size = batch_size
        self.latent_n = latent_n

    def train_on_epoch(self, dataset):
        for x in dataset:
            tic = time.time()
            self.get_loss_and_apply_grads(x)
            toc = time.time()
            message = f'{toc - tic:4.2f} sec'
            print(message)

    @tf.function
    def get_loss_and_apply_grads(self, images):
        noise = tf.random.normal([self.batch_size, self.latent_n])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.nets.gen(noise, training=True)

            real_output = self.nets.dis(images, training=True)
            fake_output = self.nets.dis(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.nets.gen.trainable_variables)
        dis_grads = disc_tape.gradient(disc_loss, self.nets.dis.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_grads, self.nets.gen.trainable_variables))
        self.dis_optimizer.apply_gradients(zip(dis_grads, self.nets.dis.trainable_variables))

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


class OptVAE:

    def __init__(self, nets, optimizer):
        self.nets = nets
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean()

    # @tf.function()
    def train_on_epoch(self, dataset):
        self.loss_tracker.reset_states()
        for x in dataset:
            loss = self.get_loss_and_apply_grads(x)
            self.loss_tracker(loss)

    @tf.function()
    def get_loss_and_apply_grads(self, x):
        with tf.GradientTape() as tape:
            loss = self.perform_fwd_pass(x)
        grads = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nets.trainable_variables))
        return loss

    def compute_loss(self, x, x_hat, mean, log_var):
        log_px_z = -tf.reduce_mean(compute_xe_loss(x, x_hat))
        kl = tf.reduce_mean(compute_gauss_kl(mean, log_var))
        neg_elbo = kl - log_px_z
        return neg_elbo

    def perform_fwd_pass(self, x):
        mean, log_var, x_hat = self.get_params_and_recon(x)
        loss = self.compute_loss(x, x_hat, mean, log_var)
        return loss

    def get_params_and_recon(self, x):
        mean, log_var = self.nets.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.nets.decoder(z)
        return mean, log_var, x_hat

    def reparameterize(self, mean, log_var):
        stddev = tf.math.exp(0.5 * log_var)
        epsilon = tf.random.normal(shape=tf.shape(mean))
        z = mean + epsilon * stddev
        return z

    def test_on_epoch(self, dataset):
        self.loss_tracker.reset_states()
        for x in dataset:
            loss = self.perform_fwd_pass(x)
            self.loss_tracker(loss)

    def predict(self, im):
        _, _, x_hat = self.get_params_and_recon(im)
        return x_hat


class OptVAEMix(OptVAE):

    def __init__(self, nets, optimizer):
        super().__init__(nets, optimizer)
        if hasattr(self.optimizer, 'loss_scale'):
            self.mixed_precision = True
        else:
            self.mixed_precision = False

    @tf.function()
    def get_loss_and_apply_grads(self, x):
        if self.mixed_precision:
            loss = self.get_loss_and_apply_grads_mixed(x)
        else:
            loss = self.get_loss_and_apply_grads_normal(x)
        return loss

    def get_loss_and_apply_grads_normal(self, x):
        with tf.GradientTape() as tape:
            loss = self.perform_fwd_pass(x)
        grads = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nets.trainable_variables))
        return loss

    def get_loss_and_apply_grads_mixed(self, x):
        with tf.GradientTape() as tape:
            loss = self.perform_fwd_pass(x)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_grads = tape.gradient(target=scaled_loss,
                                     sources=self.nets.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.nets.trainable_variables))
        return loss


class OptVAEDist(OptVAE):

    def __init__(self, nets, optimizer, global_batch_size, strategy):
        super().__init__(nets, optimizer)
        self.global_batch_size = global_batch_size
        self.strategy = strategy

    # @tf.function
    def train_on_epoch(self, dataset):
        # self.loss_tracker.reset_states()
        total_loss = 0.0
        num_batches = 0
        for x in dataset:
            total_loss += self.distributed_train_step(x)
            num_batches += 1
        # loss = total_loss / tf.cast(num_batches, total_loss.dtype)
        # self.loss_tracker(loss)

    @tf.function
    def distributed_train_step(self, x):
        per_replica_losses = self.strategy.run(self.get_loss_and_apply_grads, args=(x,))
        reduction = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return reduction

    def compute_loss(self, x, x_hat, mean, log_var):
        log_px_z = -compute_xe_loss(x, x_hat)
        kl = compute_gauss_kl(mean, log_var)
        neg_elbo = kl - log_px_z
        neg_elbo = tf.nn.compute_average_loss(neg_elbo, global_batch_size=self.global_batch_size)
        return neg_elbo

    def test_on_epoch(self, dataset):
        total_loss = 0.0
        num_batches = 0
        for x in dataset:
            total_loss += self.distributed_test_step(x)
            num_batches += 1
        output = total_loss / tf.cast(num_batches, total_loss.dtype)
        return output

    @tf.function
    def distributed_test_step(self, x):
        per_replica_losses = self.strategy.run(self.perform_fwd_pass, args=(x,))
        reduction = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return reduction


def compute_xe_loss(x, x_hat):
    xe = tf.nn.sigmoid_cross_entropy_with_logits(x, x_hat)
    xe = tf.reduce_sum(xe, axis=(1, 2, 3))
    return xe


def compute_gauss_kl(mean, log_var):
    stddev = tf.math.exp(0.5 * log_var)
    gaussian_prior = tfpd.Normal(loc=tf.zeros_like(mean), scale=tf.ones_like(log_var))
    gaussian_posterior = tfpd.Normal(loc=mean, scale=stddev)
    kl = tfpd.kl_divergence(gaussian_posterior, gaussian_prior)
    # kl = -0.5 * (1 + log_var - tf.math.exp(log_var) - tf.math.square(mean))
    kl = tf.reduce_sum(kl, axis=1)
    return kl
