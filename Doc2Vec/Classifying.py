import os
import numpy as np
from Doc2VecModel import Doc2VecModel
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc


def load_d2v_model():

    print "Loading Doc2Vec Model..."

    doc2vec = Doc2VecModel()
    doc2vec.load()
    return doc2vec


def create_train_test(n_samples, doc2vec, save_svmlight=True):

    print "Creating train & test sets..."

    # Create labelled data arrays.

    data = np.zeros((n_samples, doc2vec.size))
    labels = np.zeros(n_samples)

    for i in range(n_samples / 2):

        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)

        data[i] = doc2vec.model.docvecs[prefix_train_pos]
        data[n_samples / 2 + i] = doc2vec.model.docvecs[prefix_train_neg]

        labels[i] = 1

    # Split in train and validation arrays.

    train, test, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.3, random_state=42)

    if save_svmlight:

        current_path = os.path.abspath(
            os.path.join(os.getcwd(), os.pardir))

        dump_svmlight_file(train, train_labels, current_path + "/Data/Processed/TrainSet.svm")
        dump_svmlight_file(test, test_labels, current_path + "/Data/Processed/TestSet.svm")

    return train, test, train_labels, test_labels


def load_train_test():

    current_path = os.path.abspath(
            os.path.join(os.getcwd(), os.pardir))

    train, train_labels = load_svmlight_file(current_path + "/Data/Processed/TrainSet.svm")
    test, test_labels = load_svmlight_file(current_path + "/Data/Processed/TestSet.svm")

    return train, test, train_labels, test_labels


def classification_results(classifier, train, test, train_labels, test_labels):

    test_score = classifier.fit(train, train_labels).predict_proba(test)
    test_predicted = classifier.predict(test)

    # Evaluation of the prediction
    print classification_report(test_labels, test_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(test_labels, test_predicted))

    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(test_labels, test_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc


if __name__ == "__main__":

    d2v = load_d2v_model()

    data_train, data_test, y_train, y_test = create_train_test(25000, d2v)
    # data_train, data_test, y_train, y_test = load_train_test()

    clf = LogisticRegression(class_weight="auto", penalty="l2")
    print classification_results(clf, data_train, data_test, y_train, y_test)

