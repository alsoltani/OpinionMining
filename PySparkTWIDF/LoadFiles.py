import os
import numpy as np


def load_labeled(path):

    """
    Assumes labelled data is stored into a positive and negative folder.
    Returns two lists one with the text per file and another with the corresponding class.
    """

    rootdir_pos = path + '/pos'
    rootdir_neg = path + '/neg'
    data = []

    for subdir, dirs, files in os.walk(rootdir_pos):	
        for _file in files:
            with open(rootdir_pos + "/" + _file, 'r') as content_file:
                content = content_file.read()  # assume that there are NO "new line characters"
                data.append(content)
    tmpc1 = np.ones(len(data))
    
    for subdir, dirs, files in os.walk(rootdir_neg):	
        for _file in files:
            with open(rootdir_neg + "/" + _file, 'r') as content_file:
                content = content_file.read()  # assume that there are NO "new line characters"
                data.append(content)
    tmpc0 = np.zeros(len(data)-len(tmpc1))

    classes = np.concatenate((tmpc1, tmpc0), axis=0)
    return data, classes


def load_unknown(path):

    """
    Loads unlabelled data.
    Returns two lists : one with the data per file
    and another with the respective filenames (without the file extension).
    """

    rootdir = path
    data = []
    names = []
    for subdir, dirs, files in os.walk(rootdir):
        for _file in files:
            with open(rootdir + "/" + _file, 'r') as content_file:
                content = content_file.read()  # assume that there are NO "new line characters"
                data.append(content)
                names.append(_file.split(".")[0])
    return data, names
