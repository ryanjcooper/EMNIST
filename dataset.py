import numpy as np

from datetime import datetime
from scipy.io import loadmat, savemat


class Dataset:

    def __init__(self, batch_size=32):
        self._train_images = list()
        self._train_labels = list()

        self._test_images = list()
        self._test_labels = list()

        self.batch_size = batch_size
        self._load_dataset()

    def _load_dataset(self):
        self.data = loadmat('dataset/wlc-byclass.mat')

    def _append_to_dataset(self, test_data=False):
        if test_data:
            test_data = self.data['dataset'][0][0][1][0][0]
            self.data['dataset'][0][0][1][0][0][0] = np.append(test_data[0], self._test_images, axis=0)
            self.data['dataset'][0][0][1][0][0][1] = np.append(test_data[1], self._test_labels, axis=0)

            self._test_labels = list()
            self._test_images = list()
        else:
            train_data = self.data['dataset'][0][0][0][0][0]
            self.data['dataset'][0][0][0][0][0][0] = np.append(train_data[0], self._train_images, axis=0)
            self.data['dataset'][0][0][0][0][0][1] = np.append(train_data[1], self._train_labels, axis=0)

            self._train_labels = list()
            self._train_images = list()

    def add_image(self, image, label, test_data=False):

        if len(image) != len(self.data['dataset'][0][0][0][0][0][0][0]):
            raise Exception("Image data should be an array of length 784")

        reverse_mapping = {kv[1:][0]:kv[0] for kv in self.data['dataset'][0][0][2]}
        m_label = reverse_mapping.get(ord(label))

        if m_label is None:
            raise Exception("The dataset doesn't have a mapping for {}".format(label))

        if test_data:
            self._test_images.append(image)
            self._test_labels.append([m_label])
        else:
            self._train_images.append(image)
            self._train_labels.append([m_label])

        if len(self._test_images) >= self.batch_size or len(self._train_images) >= self.batch_size:
            self._append_to_dataset(test_data)

    def save(self, do_compression=True):
        if len(self._test_images) > 0:
            self._append_to_dataset(test_data=True)

        if len(self._train_images) > 0:
            self._append_to_dataset()

        file_name = 'dataset/wlc-byclass-{}.mat'.format(datetime.now())
        savemat(file_name=file_name, mdict=self.data, do_compression=do_compression)
