import os
import pandas as pd
import numpy as np


def load_numpy_data(shuffle=False):

    """
    Load train data as numpy array.
    :param shuffle: Option to shuffle rows.
    :return: numpy array.
    """

    sources = {"PositiveExamples.txt": 1,
               "NegativeExamples.txt": 0}

    current_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

    x_train, y_train = [], []

    for source, label in sources.items():

        x_source = pd.read_csv(current_path + "/Data/Processed/" + source, header=None, sep="\s")
        x_train.append(x_source)
        y_train.append(np.ones(len(x_source)).fill(label))

    x_train, y_train = pd.concat(x_train, axis=0), np.concatenate(y_train)

    if shuffle:
        return x_train.reindex(np.random.permutation(x_train.index)), \
               y_train[np.random.permutation(x_train.index)]
    else:
        return x_train, y_train
