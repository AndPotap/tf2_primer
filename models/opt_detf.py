import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow_probability import distributions as tfpd
tfc = tf.compat.v1


class OptVAE:

    def __init__(self, nets, optimizer):
        self.nets = nets
        self.optimizer = optimizer
        self.loss_tracker = []
        self.x = tfc.placeholder(tf.float32, [None, 28, 28, 1])

    def train_on_epoch(self, dataset):
        for i, batch in enumerate(tfds.as_numpy(dataset)):
            np_x = batch['image']
            self.sess.run(self.train_op, {self.x: np_x})

    def test_on_epoch(self, dataset):
        loss = []
        for i, batch in enumerate(tfds.as_numpy(dataset)):
            np_x = batch['image']
            np_elbo = self.sess.run([self.elbo], {self.x: np_x})
            loss.append(np_elbo)
        return np.mean(loss)

    def initialize_session(self):
        self.perform_fwd_pass()
        self.train_op = self.optimizer.minimize(-self.elbo)
        init_op = tfc.global_variables_initializer()

        self.sess = tfc.InteractiveSession()
        self.sess.run(init_op)

    def perform_fwd_pass(self):
        mean, log_var = self.nets.encoder(self.x)
        stddev = tf.exp(0.5 * log_var)

        qz_x = tfpd.Normal(loc=mean, scale=stddev)
        z = qz_x.sample()

        logits = self.nets.decoder(z)
        px_z = tfpd.Bernoulli(logits=logits)

        p_z = tfpd.Normal(loc=tf.zeros_like(z), scale=tf.ones_like(z))
        kl = tf.reduce_sum(tfpd.kl_divergence(qz_x, p_z), axis=1)
        expected_log_likelihood = tf.reduce_sum(px_z.log_prob(self.x), axis=(1, 2, 3))

        self.elbo = tf.reduce_mean(expected_log_likelihood - kl, axis=0)
