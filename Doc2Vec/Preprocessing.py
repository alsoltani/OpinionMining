import os
import re
import string
import Utility.LoadFiles as lF
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# current_path = os.getcwd()
current_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

# Replacement patterns.
replacement_patterns = [
    (r"won\'t", "will not"),
    (r"wont", "will not"),
    (r"can\'t", "cannot"),
    (r"cant", "cannot"),
    (r"i\'m", "i am"),
    (r"ain\'t", "is not"),
    (r"aint", "is not"),
    (r"let\'s", "let us"),
    ("didnt", "did not"),
    ("couldnt", "could not"),
    ("wouldnt", "would not"),
    ("mustnt", "must not"),
    ("shouldnt", "should not"),
    ("youre", "you are"),
    ("theyre", "they are"),
    (r"(\w+)\'ll", "\g<1> will"),
    (r"(\w+)n\'t", "\g<1> not"),
    (r"(\w+)\'ve", "\g<1> have"),
    (r"(\w+)\'s", "\g<1> is"),
    (r"(\w+)\'re", "\g<1> are"),
    (r"(\w+)\'d", "\g<1> would"),
    ("wanna", "want to"),
    ("gonna", "going to"),
    ("dunno", "do not know")
]

# Additional stopwords.
additional_stops = list(string.ascii_lowercase) + ["adult", "br", "isn", "ll",
                                                   "le", "didn", "doesn", "should",
                                                   "could", "would", "aren", "shouldn",
                                                   "couldn", "wouldn", "re", "ve"]


class RegexpReplacer(object):

    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for regex, repl in patterns]

    def replace(self, text):

        s = text
        for pattern, repl in self.patterns:
            s, count = re.subn(pattern, repl, s)

        return s


def pre_processing(raw_review,
                   lemmatize=False,
                   remove_stopwords=False,
                   stem=False):

    """
    Function to convert a document - i.e. a raw review.
    Optionally removes stop words.

    :param raw_review: string (a raw movie review).
    :param lemmatize: optional.
    :param remove_stopwords: optional.
    :param stem: optional.
    :return: string (a preprocessed movie review).
    """

    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # Replace abbreviations
    review_text = RegexpReplacer().replace(review_text)

    # TODO: Keep exclamation points for Doc2Vec?
    # Remove non-alphanumeric characters
    alphanumeric = re.sub(r"\W+", " ", review_text)

    # Convert to lower case, split into individual words
    words = alphanumeric.lower().split()

    # Lemmatize
    if lemmatize:
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(w, 'n') for w in words]

    # Optionally remove stop words and/or stem (false by default, not used in Doc2Vec)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops.union(set(additional_stops))]

    if stem:
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(w) for w in words]

    # Return the processed document as a string.
    return " ".join(words)


def pre_process_labeled(save=True, remove_stopwords=False):

    """
    Full pre-processing on labeled data.

    :param save: Optionally save the processed results.
    :param remove_stopwords: optional.
    """

    current_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

    train, y = lF.load_labeled(current_path + "/Data/train")
    processed_train = map(lambda r: pre_processing(r, remove_stopwords), train)

    if save:
        # Pickle positive examples.
        positive = np.where(y == 1)[0]

        with open(current_path + "/Data/Processed/PositiveExamples.txt", 'wb') as text_file:
            for idx in positive:
                text_file.write("%s\n" % processed_train[idx])

        # Pickle negative examples.
        negative = np.where(y == 0)[0]

        with open(current_path + "/Data/Processed/NegativeExamples.txt", 'wb') as text_file:
            for idx in negative:
                text_file.write("%s\n" % processed_train[idx])


def pre_process_unlabeled(save=True, remove_stopwords=False):

    """
    Full pre-processing on unlabeled data.

    :param save: Optionally save the processed results.
    :param remove_stopwords: optional.
    """

    current_path = os.path.abspath(
        os.path.join(os.getcwd(), os.pardir))

    test, names = lF.load_unknown(current_path + "/Data/test")
    processed_test = map(lambda r: pre_processing(r, remove_stopwords), test)

    if save:

        # Pickle examples.
        with open(current_path + "/Data/Processed/Unlabeled.txt", 'wb') as text_file:
            for idx in xrange(len(processed_test)):
                text_file.write("%s\n" % processed_test[idx])

if __name__ == "__main__":

    pre_process_labeled()
    pre_process_unlabeled()
