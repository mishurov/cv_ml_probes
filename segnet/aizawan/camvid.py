""" camvid.py
    This script is to convert CamVid dataset to tfrecord format.
"""

import numpy as np


class_names = np.array([
    'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol',
    'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled'])

cmap = np.array([
    [128, 128, 128],
    [128, 0, 0],
    [192, 192, 128],
    [128, 64, 128],
    [60, 40, 222],
    [128, 128, 0],
    [192, 128, 128],
    [64, 64, 128],
    [64, 0, 128],
    [64, 64, 0],
    [0, 128, 192],
    [0, 0, 0]])

cb = np.array([
    0.2595,
    0.1826,
    4.5640,
    0.1417,
    0.5051,
    0.3826,
    9.6446,
    1.8418,
    6.6823,
    6.2478,
    3.0,
    7.3614])

label_info = {
    'name': class_names,
    'num_class': len(class_names),
    'id': np.arange(len(class_names)),
    'cmap': cmap,
    'cb': cb
}
