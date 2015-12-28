# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:58:06 2015

@author: pacard
"""

import matplotlib.pyplot as plt
import Utility.LoadFiles as lF
import numpy as np
import nltk
import os
from nltk.stem.porter import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
import time
import string
import math
from sklearn.metrics import roc_curve, auc
from MyGraph import create_graph_features
import pandas as pd
import itertools
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder


def split(_string, divs):
    for d in divs[1:]:
        _string = _string.replace(d, divs[0])
    return _string.split(divs[0])


def pre_processing(dataset, stop_words, additional_words, stemmer=None):

    """
    For each document in the dataset, do the pre-processing.
    :param dataset:
    :return:
    """

    for doc_id, text in enumerate(dataset):

        text = re.sub("[^a-zA-Z]", " ", text)

        # On decode un peu le bordel
        doc = nltk.word_tokenize(text.decode('utf-8').lower())

        # On enlève la ponctuation
        punctuation = set(string.punctuation)
        doc = [w for w in doc if w not in punctuation]

        # On casse les mots à la con du type 8/10, ou quand le FdP a oublié un espace après le point..
        doc = [split(x, '.') for x in doc]
        doc = list(itertools.chain(*doc))
        doc = [split(x, '/') for x in doc]
        doc = list(itertools.chain(*doc))
        doc = [split(x, '`') for x in doc]
        doc = list(itertools.chain(*doc))
        
        # On enlève les stopwords
        doc = [w for w in doc if w not in stop_words]
        doc = [w for w in doc if w not in additional_words]
        
        s = " "
        doc = s.join(doc)
        # On stemme un peu tout ça
        # doc = [stemmer.stem(w) for w in doc]
        dataset[doc_id] = doc


def predict_proba(x_mat, model):

    f = np.vectorize(lambda r: 1/(1 + np.exp(-r)))
    raw_predictions = model.decision_function(x_mat)
    platt_predictions = f(raw_predictions)
    probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]

    return probs


def build_features(n_documents, clean_train_documents, unique_words, sliding_window, idfs):
    
    print "Building features..."
    start = time.time()

    '''fit_transform() does two functions: First, it fits the model
    and learns the vocabulary; second, it transforms our training data
    into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_documents)'''
    
    tfidf_vect = TfidfVectorizer(analyzer="word", lowercase=True, norm=None)
    # features = tfidf_vect.fit_transform(clean_train_documents)
    
    # tw-idf features on train data
    features, idfs_learned, nodes = create_graph_features(n_documents, clean_train_documents,
                                                          unique_words, sliding_window, True, idfs)

    end = time.time()
    print "Total time to build features:\t" + str(end - start)

    return features, idfs_learned, nodes
    

def train_classifier(classifier, features, labels):

    print "Training the classifier..."
    start = time.time()

    # Fit a LabelEncoder to the unique label values.

    le = LabelEncoder()
    le.fit(labels[0].unique())
    print "Number of classes:" + str(len(labels[0].unique()))

    # Map the values of the class to the numbers.

    mapped_labels = le.transform(labels[0])

    # Fit the classifier.

    classifier.fit(features, mapped_labels)

    end = time.time()
    print "Total time to train classifier:\t" + str(end - start)
    
if __name__ == "__main__":

    current_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

    data, y = lF.load_labeled(current_path + "/Data/train")
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english")

    words = ['br', "n't", "''", "```", "'s", "...", "``", "'ll", "'d", "'m"]

    # 0. Pre-process data.
    # --------------------

    pre_processing(data, stop_words, words, stemmer)
    data_frame, label_frame = pd.DataFrame(data), pd.DataFrame(y)

    # 1. Build features.
    # Parameters for the Graph-of-Words algorithm.
    # --------------------

    # Idfs is a dictionary containing all the idfs (needed for the tw-idf).

    sliding_window = 2
    idfs = {}

    n_documents = data_frame.shape[0]
    clean_train_documents = data_frame[0].values
    unique_words = pd.DataFrame(data_frame[0].str.split(' ').tolist()).stack()\
        .value_counts().index.values
    # unique_words = list(set(data_frame[0].str.split(' ').apply(pd.Series, 1).stack().values))

    print "Unique words:" + str(len(unique_words))

    features, idfs_learned, nodes = build_features(n_documents, clean_train_documents,
                                                   unique_words, sliding_window, idfs)

    # 2. Train classifier.
    # --------------------

    clf = svm.LinearSVC(loss="hinge")
    print train_classifier(clf, features, label_frame)
