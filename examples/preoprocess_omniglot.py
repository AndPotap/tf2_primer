import time
import tensorflow as tf
from utils.load_data import ProcessData
from utils.load_data import _parse_image_function_float
from matplotlib import pyplot as plt


run_with_sample = False
batch_size, epochs = 64, 10
seed = 9331
tf.random.set_seed(seed)
processor = ProcessData(dataset_name='omniglot',
                        run_with_sample=run_with_sample)
ds, *_ = processor.generate_train_and_test_partitions(batch_size, epochs)
ds_train, _ = ds

tic = time.time()
output_file = 'omniglot_train.tfrecords'

tic = time.time()
other = ds_train.map(tf.io.serialize_tensor)
writer = tf.data.experimental.TFRecordWriter(output_file)
writer.write(other)
toc = time.time()
print(f'\nIt took {toc - tic:4.2f} sec')

raw_ims = tf.data.TFRecordDataset(output_file)
parsed_image_dataset = raw_ims.map(_parse_image_function_float)

for r in parsed_image_dataset:
    plt.figure()
    plt.imshow(r) if r.shape[3] > 1 else plt.imshow(r[0, :, :, 0])
    breakpoint()
    plt.show()
