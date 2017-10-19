# Mute tensorflow debugging information console
import os
import pickle
import argparse
import keras
import numpy as np

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def reshape(img, width, height):
    # Used to rotate images (for some reason they are transposed on read-in)
    img.shape = (width, height)
    img = img.T
    img = list(img)
    img = [item for sublist in img for item in sublist]
    return img


def extract_images_and_labels(dataset, training_data=True):
    if training_data:
        training_images = dataset['dataset'][0][0][0][0][0][0]
        training_labels = dataset['dataset'][0][0][0][0][0][1]
    else:
        training_images = dataset['dataset'][0][0][1][0][0][0]
        training_labels = dataset['dataset'][0][0][1][0][0][1]

    return training_images, training_labels


def append_datasets(arr1, arr2):
    return np.append(arr1, arr2, axis=0)


def load_data(emnist_file_path, wlc_file_path, width=28, height=28, verbose=True):
    wlc_training_images, wlc_training_labels = None, None
    wlc_testing_images, wlc_testing_labels = None, None

    # Load .mat dataset
    emnist = loadmat(emnist_file_path)
    wlc = None

    if wlc_file_path:
        wlc = loadmat(wlc_file_path)

    mapping = {kv[0]: kv[1:][0] for kv in emnist['dataset'][0][0][2]}

    if wlc_file_path:
        mapping = {kv[0]: kv[1:][0] for kv in wlc['dataset'][0][0][2]}

    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    # Load training data
    training_images, training_labels = extract_images_and_labels(emnist)
    testing_images, testing_labels = extract_images_and_labels(emnist, training_data=False)

    if wlc_file_path:
        wlc_training_images, wlc_training_labels = extract_images_and_labels(wlc)
        wlc_testing_images, wlc_testing_labels = extract_images_and_labels(wlc, training_data=False)

        training_images = append_datasets(training_images, wlc_training_images)
        training_labels = append_datasets(training_labels, wlc_training_labels)

        testing_images = append_datasets(testing_images, wlc_testing_images)
        testing_labels = append_datasets(testing_labels, wlc_testing_labels)

    # Reshape training data to be valid
    if verbose: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        training_images[i] = reshape(training_images[i], width, height)
    if verbose: print('')

    # Reshape testing data to be valid
    if verbose: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        testing_images[i] = reshape(testing_images[i], width, height)
    if verbose: print('')

    if wlc_file_path:
        wlc_training_images, wlc_training_labels = extract_images_and_labels(wlc)
        wlc_testing_images, wlc_testing_labels = extract_images_and_labels(wlc, training_data=False)

        training_images = append_datasets(training_images, wlc_training_images)
        training_labels = append_datasets(training_labels, wlc_training_labels)

        testing_images = append_datasets(testing_images, wlc_testing_images)
        testing_labels = append_datasets(testing_labels, wlc_testing_labels)

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0], height, width, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], height, width, 1)

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return (training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes


def build_net(training_data, width=28, height=28, verbose=False):
    # Initialize data
    _, _, mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if verbose: print(model.summary())
    return model


def train(model, training_data, callback=True, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if callback:
        # Callback for analysis in TensorBoard
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
                                                 write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tb_callback] if callback else None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('--emnist', type=str, help='Path emnist-byclass-extended.mat file data', required=True)
    parser.add_argument('--wlc', type=str, help='Path wlc-byclass.mat file data')
    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.emnist, args.wlc, width=args.width, height=args.height, verbose=args.verbose)
    model = build_net(training_data, width=args.width, height=args.height, verbose=args.verbose)
    train(model, training_data, epochs=args.epochs)
