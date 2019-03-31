"""Reference TensorFlow implementation of the pneumonia kaggle challange."""

# %% imports
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE, shuffle_and_repeat
import pathlib
import random


# %% definitions
class FLAGS:
    """Flags."""

    batch_size = 32
    shuffle_buffer_size = 1000
    data_root = pathlib.Path("chest_xray")


def load_and_prepocess_image(path: str, label: str) -> Tensor:
    """Load and preprocess image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, [192, 192])
    return image, label


label_names = sorted(
    dir.name for dir in FLAGS.data_root.glob("train/*") if dir.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))


def dataset(path: str) -> Dataset:
    """Create a dataset given a path."""
    data_root = FLAGS.data_root/path

    image_paths = list(str(path) for path in data_root.glob("*/*"))
    random.shuffle(image_paths)

    image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in image_paths]

    dataset = Dataset.from_tensor_slices((image_paths, image_labels)) \
        .map(load_and_prepocess_image, num_parallel_calls=AUTOTUNE) \
        .cache() \
        .apply(shuffle_and_repeat(buffer_size=FLAGS.shuffle_buffer_size)) \
        .batch(FLAGS.batch_size) \
        .prefetch(buffer_size=AUTOTUNE)

    return dataset, len(image_paths)


# %% training
train_dataset, train_count = dataset("train")
val_dataset, val_count = dataset("val")
test_dataset, test_count = dataset("test")

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.Accuracy()])

model.fit(train_dataset,
          epochs=3,
          steps_per_epoch=train_count // FLAGS.batch_size,
          validation_data=val_dataset,
          validation_steps=val_count // FLAGS.batch_size,
          callbacks=[
            tf.keras.callbacks.ModelCheckpoint('checkpoints/cp.ckpt'),
            tf.keras.callbacks.TensorBoard()])
