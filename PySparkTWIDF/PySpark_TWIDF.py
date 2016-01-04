import os
import re
import sys
import math
import string
import networkx as nx
import LoadFiles as lF
from functools import partial
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from Preprocessing import RegexpReplacer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.tree import RandomForest
    from pyspark.mllib.linalg import SparseVector
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.classification import LogisticRegressionWithSGD
    from pyspark.mllib.feature import PCA

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Cannot import Spark Modules", e)
    sys.exit(1)

# Pre-processing tools.
# ---------------------

wnl = WordNetLemmatizer()
stops = set(stopwords.words("english"))
additional_stops = list(string.ascii_lowercase) + ["adult", "br", "isn", "ll",
                                                   "le", "didn", "doesn", "should",
                                                   "could", "would", "aren", "shouldn",
                                                   "couldn", "wouldn", "re", "ve"]


def create_binary_labeled_point(doc_class, dictionary, window, idf_col, term_num_docs):
    
    """
    Create the binary labeled points in a TW-IDF fashion.
    """
    
    # Pre-process text.
    # ---------------------
    
    vector_dict = {}

    review_text = re.compile(r"<[^>]+>").sub('', doc_class[0])
    review_text = RegexpReplacer().replace(review_text)
    alphanumeric = re.sub(r"\W+", " ", review_text)
    
    words = alphanumeric.lower().split()
    # words = [wnl.lemmatize(w, 'n') for w in words]
    # words = [w for w in words if w not in stops.union(set(additional_stops))]

    # stemmer = SnowballStemmer("english")
    # words = [stemmer.stem(w) for w in words]

    word_list = " ".join(words).split(None)

    # Compute the graph.
    # ---------------------

    dg = nx.Graph()
    if len(word_list) > 1:

        populate_graph(word_list, dg, window)
        dg.remove_edges_from(dg.selfloop_edges())
        centrality = nx.closeness_centrality(dg)

        for k, node_term in enumerate(dg.nodes()):
            if node_term in idf_col:
                if node_term in dictionary:
                    if term_num_docs[node_term] > 5:  # Arbitrary threshold

                        vector_dict[dictionary[node_term]] = centrality[node_term] * idf_col[node_term]

    return LabeledPoint(doc_class[1], SparseVector(len(dictionary), vector_dict))


def predict(text, dictionary, window, idf_col, model, term_num_docs):
    
    """
    Predict label values.
    """
    
    # Pre-process text.
    # ---------------------
    
    vector_dict = {}

    review_text = re.compile(r"<[^>]+>").sub('', text)
    review_text = RegexpReplacer().replace(review_text)
    alphanumeric = re.sub(r"\W+", " ", review_text)
    
    words = alphanumeric.lower().split()
    # words = [wnl.lemmatize(w, 'n') for w in words]
    # words = [w for w in words if w not in stops.union(set(additional_stops))]

    word_list = " ".join(words).split(None)

    # Compute the graph.
    # ---------------------

    dg = nx.Graph()

    if len(word_list) > 1:
        populate_graph(word_list, dg, window)
        dg.remove_edges_from(dg.selfloop_edges())
        centrality = nx.closeness_centrality(dg)

        for k, node_term in enumerate(dg.nodes()):
            if node_term in idf_col:
                if node_term in dictionary:
                    if term_num_docs[node_term] > 5:  # Arbitrary threshold

                        vector_dict[dictionary[node_term]] = centrality[node_term] * idf_col[node_term]

    return model.predict(SparseVector(len(dictionary), vector_dict))


def populate_graph(word_list, dg, window):

    for k, word in enumerate(word_list):

        if not dg.has_node(word):
            dg.add_node(word)
        temp_w = window

        if k + window > len(word_list):
            temp_w = len(word_list) - k

        for j in xrange(1, temp_w):
            next_word = word_list[k + j]
            dg.add_edge(word, next_word)
            
if __name__ == "__main__":

    # Set PySparkTWIDF Context and load data.
    # ---------------------

    sc = SparkContext("local", "TW-IDF App", pyFiles=['PySparkTWIDF/Preprocessing.py', 'PySparkTWIDF/LoadFiles.py'])
    current_path = os.getcwd()

    print "Loading data..."

    data, Y = lF.load_labeled(current_path + "/Data/train")

    data_train, data_test, labels_train, labels_test = train_test_split(data, Y, test_size=0.2, random_state=42)
    data_rdd = sc.parallelize(data_train, numSlices=16)

    # Map data to a binary matrix.
    # Get the dictionary of the data.
    # ---------------------

    print "Pre-processing data and broadcasting the dictionary..."

    lists = data_rdd \
        .map(lambda r: re.compile(r"<[^>]+>").sub('', r)) \
        .map(RegexpReplacer().replace) \
        .map(lambda r: re.sub(r"\W+", " ", r)) \
        .map(lambda r: r.lower().split()) \
        .collect()

    # .map(lambda r: [wnl.lemmatize(w, 'n') for w in r]) \
    # .map(lambda r: [w for w in r if w not in stops.union(set(additional_stops))]) \
    # .collect()

    # Combine lists together.
    # ---------------------
    # We need the dictionary to be available as a whole throughout the cluster.
    # This dictionary will be used when computing the SparseVector.

    all_lists = []
    for l in lists:
        all_lists.extend(l)

    dictionary = {word: i for (i, word) in enumerate(set(all_lists))}
    dict_broad = sc.broadcast(dictionary)

    # Num_words for TW-IDF vectors.
    # ---------------------

    print "Getting word counts and broadcasting IDF values..."
    
    num_words_rdd = sc.parallelize(lists, numSlices=16)
    num_words_doc = num_words_rdd\
        .map(lambda x: " ".join(x))\
        .flatMap(lambda x: x.split())\
        .map(lambda x: (x, 1))\
        .reduceByKey(lambda x, y: x + y).collect()

    term_num_docs = dict(num_words_doc)
    term_num_docs_broad = sc.broadcast(term_num_docs)

    idf_col = {}
    for term_x in dict(num_words_doc):
        idf_col[term_x] = math.log10(float(len(data_train)) / term_num_docs[term_x])

    idf_col_broad = sc.broadcast(idf_col)

    # Build labeled points from data.
    # ---------------------

    print "Building labeled points..."
    
    data_class = zip(data_train, labels_train)
    data_class_rdd = sc.parallelize(data_class, numSlices=16)
    sliding_window = 4

    # Get labelled points.
    # ---------------------

    labeled_rdd = data_class_rdd.map(partial(
        create_binary_labeled_point,
        dictionary=dict_broad.value,
        window=sliding_window,
        idf_col=idf_col_broad.value,
        term_num_docs=term_num_docs_broad.value))

    # Train and broadcast the supervised model.
    # ---------------------

    print "Training the model...\n"

    # model = SVMWithSGD.train(labeled_rdd, iterations=300)
    #model = LogisticRegressionWithLBFGS.train(labeled_rdd, regParam=0.003,regType='l1',iterations=500)
    model = LogisticRegressionWithSGD.train(labeled_rdd)
    mb = sc.broadcast(model)

    # Make predictions.
    # ---------------------

    predictions = sc.parallelize(data_test)\
        .map(partial(predict,
                     dictionary=dict_broad.value,
                     window=sliding_window,
                     idf_col=idf_col_broad.value,
                     model=mb.value,
                     term_num_docs=term_num_docs_broad.value))\
        .collect()

    # Classification report.
    # ---------------------

    print classification_report(labels_test, predictions)
    print "The accuracy score is {:.2%}".format(accuracy_score(labels_test, predictions))
