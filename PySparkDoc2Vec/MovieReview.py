import os
import re
import sys
import numpy as np
from word2vec import Word2Vec
from doc2vec import TaggedDocument
from DistDoc2VecFast import DistDoc2VecFast
from Preprocessing import pre_processing

# Path for spark source folder
os.environ['SPARK_HOME'] = "/usr/local/Cellar/apache-spark/1.5.2"

# Append pyspark  to Python Path
sys.path.append("/usr/local/Cellar/apache-spark/1.5.2/libexec/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.mllib.linalg import SparseVector
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes
    from pyspark.mllib.classification import SVMWithSGD, SVMModel
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.feature import PCA

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


def swap_kv(tp):
    return tp[1], tp[0]


def sentences(l):
    l = re.compile(r'([!\?])').sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")


def parse_sentences(rdd):
    raw = rdd.zipWithIndex().map(swap_kv)

    data = raw.flatMap(lambda (_id, text): [(_id, pre_processing(s).split()) for s in sentences(text)])
    return data


def parse_paragraphs(rdd):
    raw = rdd.zipWithIndex().map(swap_kv)

    def clean_paragraph(text):
        paragraph = []
        for s in sentences(text):
            paragraph = paragraph + pre_processing(s).split()

        return paragraph

    data = raw.map(lambda (id, text): TaggedDocument(words=clean_paragraph(text), tags=[id]))
    return data


def word2vec(rdd):

    sentences = parse_sentences(rdd)
    sentences_without_id = sentences.map(lambda (_id, sent): sent)
    model = Word2Vec(size=100, hs=0, negative=8)

    dd2v = DistDoc2VecFast(model, learn_hidden=True, num_partitions=15, num_iterations=20)
    dd2v.build_vocab_from_rdd(sentences_without_id)

    print "*** done training words ****"
    print "*** len(model.vocab): %d ****" % len(model.vocab)
    return dd2v, sentences


def doc2vec(dd2v, rdd):

    paragraphs = parse_paragraphs(rdd)
    dd2v.train_sentences_cbow(paragraphs)
    print "**** Done Training Doc2Vec ****"

    def split_vec(iterable):
        dvecs = iter(iterable).next()
        dvecs = dvecs['doctag_syn0']
        n = np.shape(dvecs)[0]
        return (dvecs[i] for i in xrange(n))

    return dd2v, dd2v.doctag_syn0.mapPartitions(split_vec)


def regression(reg_data):
    
    train_data, test_data = reg_data.randomSplit([0.7, 0.3])
    model = LogisticRegressionWithLBFGS.train(train_data)
    labels_predictions = test_data.map(lambda p: (p.label, model.predict(p.features)))

    train_error = labels_predictions.filter(lambda (v, p): v != p).count() / float(test_data.count())
    false_pos = labels_predictions.filter(lambda (v, p): v != p and v == 0.0).count() / float(
        test_data.filter(lambda lp: lp.label == 0.0).count())
    false_neg = labels_predictions.filter(lambda (v, p): v != p and v == 1.0).count() / float(
        test_data.filter(lambda lp: lp.label == 1.0).count())

    print "*** Error Rate: %f ***" % train_error
    print "*** False Positive Rate: %f ***" % false_pos
    print "*** False Negative Rate: %f ***" % false_neg


if __name__ == "__main__":

    conf = SparkConf().set("spark.driver.maxResultSize", "4g")
    sc = SparkContext("local", "Doc2Vec App", conf=conf,
                      pyFiles=['PySparkDoc2Vec/LoadFiles.py',
                               'PySparkDoc2Vec/Preprocessing.py',
                               'PySparkDoc2Vec/DistDoc2VecFast.py',
                               'PySparkDoc2Vec/word2vec.py',
                               'PySparkDoc2Vec/doc2vec.py',
                               #'PySparkDoc2Vec/doc2vec_inner.c',
                               #'PySparkDoc2Vec/doc2vec_inner.pyx',
                               #'PySparkDoc2Vec/word2vec_inner.c',
                               #'PySparkDoc2Vec/word2vec_inner.pyx',
                               'PySparkDoc2Vec/utils.py',
                               'PySparkDoc2Vec/matutils.py'
                               ])

    # sqlContext = SQLContext(sc)

    #current_path = os.getcwd()
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    print "Loading dataset..."

    pos = sc.textFile("file://" + current_path + "/Data/train/positive.txt")
    neg = sc.textFile("file://" + current_path + "/Data/train/negative.txt")
    both = pos + neg

    print "Building Word2Vec model..."

    dd2v, _ = word2vec(both)

    print "Building Doc2Vec model..."

    dd2v, docvecs = doc2vec(dd2v, both)

    dd2v.model.save("PySparkDoc2Vec/Models/MovieReview")

    print "Classification."

    npos = pos.count()
    reg_data = docvecs.zipWithIndex().map(lambda (v, i): LabeledPoint(1.0 if i < npos else 0.0, v))
    regression(reg_data)
