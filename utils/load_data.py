import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data.experimental import AUTOTUNE as auto


class ProcessData:

    def __init__(self, dataset_name, image_shape: tuple = (28, 28, 1),
                 buffer_size=1000, run_with_sample=False):
        self.dataset_name = dataset_name
        self.run_with_sample = run_with_sample
        self.image_shape = image_shape
        self.buffer_size = buffer_size

    def generate_train_and_test_partitions(self, batch_size, epochs):
        data = fetch_data_via_tf_datasets(dataset_name=self.dataset_name)
        batch_size, epochs = re_assign_batch_size_epochs(
            self.run_with_sample, batch_size=batch_size, epochs=epochs)
        split_names, split_data = ['train', 'test'], []
        for idx, split in enumerate(split_names):
            if self.dataset_name == 'omniglot':
                processed_data = self.preprocess(data_split=data[idx],
                                                 batch_size=batch_size)
            elif self.dataset_name == 'celeb_a':
                processed_data = self.preprocess(data_split=data[split],
                                                 batch_size=batch_size)
            split_data.append(processed_data)
        return split_data, batch_size, epochs

    def preprocess(self, data_split, batch_size):
        if self.dataset_name == 'omniglot':
            data_split = data_split.map(preprocess_omniglot, num_parallel_calls=auto)
        elif self.dataset_name == 'celeb_a':
            data_split = data_split.map(preprocess_celeb_a, num_parallel_calls=auto)
        else:
            raise NotImplementedError
        data_split = data_split.shuffle(self.buffer_size)
        data_split = data_split.batch(batch_size)
        data_split = data_split.cache()
        data_split = data_split.prefetch(auto)
        return data_split


def fetch_data_via_tf_datasets(dataset_name):
    if dataset_name != 'omniglot':
        builder = tfds.builder(name=dataset_name)
        builder.download_and_prepare()
        data = builder.as_dataset(shuffle_files=False)
    else:
        data = tfds.load(dataset_name,
                         split=['train', 'test'],
                         shuffle_files=False)
    return data


def re_assign_batch_size_epochs(run_with_sample, batch_size, epochs):
    if run_with_sample:
        batch_size, epochs = 5, 10
    return batch_size, epochs


def preprocess_celeb_a(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = tf.image.resize(images=example['image'], size=(64, 64))
    example['image'] = tf.image.per_image_standardization(example['image'])
    example['image'] = (1 / tf.sqrt(2.)) * example['image'] + 0.5
    return example['image']


def preprocess_omniglot(example):
    example['image'] = binarize_tensor(example['image'])
    example['image'] = round_tensor(example['image'])
    example['image'] = example['image'][:, :, 0:1]
    example['image'] = tf.image.resize(images=example['image'], size=(28, 28))
    example['image'] = tf.image.rot90(image=example['image'], k=1)
    example['image'] = tf.image.rot90(image=example['image'], k=1)
    return example['image']


def binarize_tensor(tensor):
    new_tensor = tf.cast(tensor, dtype=tf.float32)
    new_tensor = new_tensor / 255.
    return new_tensor


def round_tensor(tensor):
    rounded_tensor = tf.cast(tensor, dtype=tf.int32)
    rounded_tensor = tf.cast(rounded_tensor, dtype=tf.float32)
    return rounded_tensor


def _parse_image_function_float(example_proto):
    single = tf.io.parse_tensor(example_proto, tf.float32)
    return single


def decode_im_features_float(im):
    image_shape = im.shape
    im_flat = tf.reshape(im, shape=784)

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }
    for j in range(im_flat.shape[0]):
        feature.update({'im_' + str(j): _float_feature(im_flat[j])})

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_image_function(example_proto):
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto,
                                      image_feature_description)


def decode_im_features(image_string):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def show_some_records(num_records, ds):
    for r in ds.take(num_records):
        print(repr(r))


def _parse_function(example_proto):
    feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def generator(ds):
    for features in ds:
        yield serialize_example(*features)


def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(serialize_example, (f0, f1, f2, f3), tf.string)
    return tf.reshape(tf_string, ())


def serialize_example(feature0, feature1, feature2, feature3):
    feature_dict = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
