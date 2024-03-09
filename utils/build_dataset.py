import numpy as np
import tensorflow as tf

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32)
    img = 1./255*img
    return img


def load_mask(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.bool)
    img = tf.cast(img, tf.int8)
    return img


def preprocess_image(input_img_path, target_img_path):
    input_img = load_image(input_img_path)
    target_img = load_mask(target_img_path)
    return input_img, target_img


def build_dataset(input_files: list[str], output_files: list[str], batch_size: int):
    dataset = tf.data.Dataset.from_tensor_slices((input_files, output_files))
    # dataset = dataset.shuffle(len(input_files))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

def split_dataset(dataset: tf.data.Dataset, train_size: float):
    train_size = int(0.9*dataset.cardinality().numpy())
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    return train_dataset, val_dataset