import tensorflow as tf
from utils.examples_fns import increase_until_something


v = tf.constant(0.)
v = increase_until_something(v)
# print(tf.autograph.to_code(increase_until_something.python_function))
# print(v)
