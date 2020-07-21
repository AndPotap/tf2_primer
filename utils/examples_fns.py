import tensorflow as tf


@tf.function
def increase_until_something(v):
    while v < 10:
        if v % 2 == 0:
            print(v)
            tf.print(v)
            v += 1
        else:
            v += 2
            print(v)
            tf.print(v)
    return v
