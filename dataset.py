import numpy as np

from datetime import datetime
from scipy.io import loadmat, savemat


class Dataset:

    def __init__(self):
        self._load_dataset()

    def _load_dataset(self):
        self.data = loadmat('dataset/emnist-byclass-extended.mat')

    def add_image(self, image, label, test=False):

        if len(image) != len(self.data['dataset'][0][0][0][0][0][0][0]):
            raise Exception("Image data should be an array of length 784")

        reverse_mapping = {kv[1:][0]:kv[0] for kv in self.data['dataset'][0][0][2]}
        print(reverse_mapping)

        if reverse_mapping.get(ord(label)) is None:
            raise Exception("The dataset doesn't have a mapping for {}".format(label))

        if test:
            test_data = self.data['dataset'][0][0][1][0][0]
            self.data['dataset'][0][0][1][0][0][0] = np.append(test_data[0], [image], axis=0)
            self.data['dataset'][0][0][1][0][0][1] = np.append(test_data[1], [[label]], axis=0)
        else:
            train_data = self.data['dataset'][0][0][0][0][0]
            self.data['dataset'][0][0][0][0][0][0] = np.append(train_data[0], [image], axis=0)
            self.data['dataset'][0][0][0][0][0][0] = np.append(train_data[1], [[label]], axis=0)

    def save(self, do_compression=True):
        file_name = 'dataset/emnist-byclass-extended-{}.mat'.format(datetime.now())
        savemat(file_name=file_name, mdict=self.data, do_compression=do_compression)
