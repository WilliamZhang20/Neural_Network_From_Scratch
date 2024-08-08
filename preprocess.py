import tensorflow as tf
import pickle
import numpy as np

def load_and_preprocess_mnist():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], 28 * 28))  # Flattening
    test_images = test_images.reshape((test_images.shape[0], 28 * 28))  # Flattening

    train_labels = np.eye(10)[train_labels]  # One-hot encoding
    test_labels = np.eye(10)[test_labels]  # One-hot encoding

    return train_images, train_labels, test_images, test_labels

def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist()

dir = 'pkl_files'

save_to_pickle(train_images, f'{dir}/train_images.pkl')
save_to_pickle(train_labels, f'{dir}/train_labels.pkl')
save_to_pickle(test_images, f'{dir}/test_images.pkl')
save_to_pickle(test_labels, f'{dir}/test_labels.pkl')