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
    shuffle_buffer_size = 500
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


# %% data sets
train_dataset, train_count = dataset("train")
val_dataset, val_count = dataset("val")
test_dataset, test_count = dataset("test")


# %% model
xception = tf.keras.applications.Xception(
    input_shape=(192, 192, 3), include_top=False)
xception.trainable = False

model = tf.keras.Sequential([
    xception,
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, tf.keras.activations.sigmoid)])


# %% training
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])


model.fit(train_dataset,
          epochs=3,
          steps_per_epoch=train_count // FLAGS.batch_size,
          validation_data=val_dataset,
          validation_steps=1,
          callbacks=[
            tf.keras.callbacks.ModelCheckpoint('checkpoints/cp.ckpt'),
            tf.keras.callbacks.TensorBoard()])

model.evaluate(test_dataset, steps=test_count // FLAGS.batch_size)


# %% fine tuning
xception.trainable = True

for layer in xception.layers[:100]:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])


model.fit(train_dataset,
          epochs=3,
          steps_per_epoch=train_count // FLAGS.batch_size,
          validation_data=val_dataset,
          validation_steps=1,
          callbacks=[
            tf.keras.callbacks.ModelCheckpoint('checkpoints/cp.ckpt'),
            tf.keras.callbacks.TensorBoard()])

model.evaluate(test_dataset, steps=test_count // FLAGS.batch_size)
