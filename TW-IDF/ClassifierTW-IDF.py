# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:58:06 2015

@author: pacard
"""

import matplotlib.pyplot as plt
import Utility.LoadFiles as lF
import numpy as np
import nltk
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

data, Y = lF.load_labeled("./data/train")
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")


def split(_string, divs):
    for d in divs[1:]:
        _string = _string.replace(d, divs[0])
    return _string.split(divs[0])


words = ['br', "n't", "''", "```", "'s", "...", "``", "'ll", "'d", "'m"]


def pre_processing(data):   
                                       
    """
    For each document in the dataset, do the pre-processing.
    :param data: 
    :return: 
    """
    
    for doc_id, text in enumerate(data):
        
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
        doc = [w for w in doc if w not in stopwords]
        doc = [w for w in doc if w not in words]     
        
        s = " "
        doc = s.join(doc)
        # On stemme un peu tout ça
        # doc = [stemmer.stem(w) for w in doc]
        data[doc_id] = doc


def predict_proba(x_mat, model):

    f = np.vectorize(lambda r: 1/(1+np.exp(-r)))
    raw_predictions = model.decision_function(x_mat)
    platt_predictions = f(raw_predictions)
    probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
    return probs


# parameters for graph of words
sliding_window = 2
# a dictionary with all  the idfs (needed for the tw-idf)
idfs = {}

dataFrame = pd.DataFrame(data)
labelFrame = pd.DataFrame(Y)

# num of docs = rows
n_documents = dataFrame.shape[0]
clean_train_documents = dataFrame[0].values

# print n_documents
# print clean_train_documents

unique_words = list(set(dataFrame[0].str.split(' ').apply(pd.Series, 1).stack().values))
print "Unique words:"+str(len(unique_words))


print "Building features..."
# tf-idf features on train data
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
print "Total time to build features:\t"+str(end - start)

print "Training the classifier..."
start = time.time()


clf = svm.LinearSVC(loss="hinge")
Y = labelFrame[0]

# build a dictionary to assign numerical values to class labels
class_to_num = {}
classLabels = labelFrame[0].unique()

for i, val in enumerate(classLabels):
    class_to_num[val] = i
print "Class correspondence"	
print class_to_num
# map the values of the class to the numbers
y = labelFrame[0].map(class_to_num).values

print "Number of classes:" + str(len(classLabels))
model = clf.fit(features, y)
end = time.time()
print "Total time to train classifier:\t" + str(end - start)
