# Mute tensorflow debugging information on console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle

app = Flask(__name__)

def load_model(bin_dir):
    ''' Load model from .yaml and the weights from .h5

        Arguments:
            bin_dir: The directory of the bin (normally bin/)

        Returns:
            Loaded model from file
    '''

    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

@app.route("/")
def index():
    ''' Render index for user connecting to /
    '''
    return render_template('index.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    ''' Called when user presses the predict button.
        Processes the canvas and handles the image.
        Passes the loaded image into the neural network and it makes
        class prediction.
    '''

    # Local functions
    def crop(x):
        # Experimental
        _len = len(x) - 1
        for index, row in enumerate(x[::-1]):
            z_flag = False
            for item in row:
                if item != 0:
                    z_flag = True
                    break
            if z_flag == False:
                x = np.delete(x, _len - index, 0)
        return x
    def parseImage(imgData):
        # parse canvas bytes and save as output.png
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open('output.png','wb') as output:
            output.write(base64.decodebytes(imgstr))

    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    x = np.invert(x)

    ### Experimental
    # Crop on rows
    # x = crop(x)
    # x = x.T
    # Crop on columns
    # x = crop(x)
    # x = x.T

    # Visualize new array
    imsave('resized.png', x)
    x = imresize(x,(28,28))

    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model
    out = model.predict(x)

    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),
                'confidence': str(max(out[0]) * 100)[:6]}

    return jsonify(response)

if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='A webapp for testing models generated from training.py on the EMNIST dataset')
    parser.add_argument('--bin', type=str, default='bin', help='Directory to the bin containing the model yaml and model h5 files')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to run the flask server on')
    parser.add_argument('--port', type=int, default=5000, help='The port to run the flask server on')
    args = parser.parse_args()

    # Overhead
    model = load_model(args.bin)
    mapping = pickle.load(open('%s/mapping.p' % args.bin, 'rb'))

    app.run(host=args.host, port=args.port)
