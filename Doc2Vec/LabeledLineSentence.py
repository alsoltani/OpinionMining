import os
from random import shuffle
from gensim import utils
from gensim.models.doc2vec import LabeledSentence


class LabeledLineSentence(object):

    def __init__(self, _sources):

        self.sources = _sources
        self.sentences = []

        flipped = {}

        # Make sure that keys are unique.
        for key, value in _sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception("Non-unique prefix encountered.")

    def __iter__(self):

        current_path = os.path.abspath(
            os.path.join(os.getcwd(), os.pardir))

        for source, prefix in self.sources.items():
            with utils.smart_open(current_path + "/Data/Processed/" + source) as fin:
                for item_no, line in enumerate(fin):

                    # TODO: Update LabelSentence to TaggedDocument
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):

        self.sentences = []
        current_path = os.path.abspath(
            os.path.join(os.getcwd(), os.pardir))

        for source, prefix in self.sources.items():
            with utils.smart_open(current_path + "/Data/Processed/" + source) as fin:
                for item_no, line in enumerate(fin):

                    # TODO: Update LabelSentence to TaggedDocument
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

if __name__ == "__main__":

    sources = {"PositiveExamples.txt": "TRAIN_POS",
               "NegativeExamples.txt": "TRAIN_NEG"}

    sentences = LabeledLineSentence(sources).to_array()
