import os
import re
import Utility.LoadFiles as lF
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def review_to_words(raw_review, remove_stopwords=False):

    """
    Function to convert a document - i.e. a raw review.
    Optionally removes stop words.

    :param raw_review: string (a raw movie review).
    :param remove_stopwords: optional.
    :return: string (a preprocessed movie review).
    """

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    # 5. Return the processed document as a string.
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
    processed_train = map(lambda r: review_to_words(r, remove_stopwords), train)

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
    processed_test = map(lambda r: review_to_words(r, remove_stopwords), test)

    if save:

        # Pickle examples.
        with open(current_path + "/Data/Processed/Unlabeled.txt", 'wb') as text_file:
            for idx in xrange(len(processed_test)):
                text_file.write("%s\n" % processed_test[idx])


if __name__ == "__main__":

    pre_process_unlabeled()
