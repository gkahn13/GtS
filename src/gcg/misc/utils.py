import random
import importlib.util

import numpy as np
import tensorflow as tf
import PIL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    assert (len(shape) == 3)
    assert (shape[-1] == 1 or shape[-1] == 3)
    assert (image.shape[0] / image.shape[1] == shape[0] / shape[1]) # maintain aspect ratio
    height, width, channels = shape

    if channels == 1 and len(image.shape) > 2:
        image = image[:,:,0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert (im.shape == tuple(shape))

    return im

def im2gray(image):
    assert (image.shape[-1] == 3)

    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

def multiple_expand_dims(x, axes):
    for axis in axes:
        x = np.expand_dims(x, axis)
    return x

def import_params(py_config_path):
    spec = importlib.util.spec_from_file_location('config', py_config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params
