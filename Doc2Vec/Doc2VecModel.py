from gensim.models import Doc2Vec
from LabeledLineSentence import LabeledLineSentence


class Doc2VecModel:

    def __init__(self,
                 min_count=1,
                 window=10,
                 size=400,
                 sample=1e-4,
                 negative=5,
                 workers=8):

        """
        Set up Doc2Vec parameters.

        :param min_count: ignore all words with lower frequency than this.
                          You have to set this to 1, since the sentence labels appear
                          only once. Setting it any higher than 1 will miss out on
                          the sentences.
        :param window:    maximum distance between the current and the predicted word
                          in a sentence. Word2Vec uses a skip-gram model, and this is
                          simply the window size of the skip-gram model.
        :param size:      dimensionality of the feature vectors in output.
                          100 is a good number. Up to 400 for extreme cases.
        :param sample:    threshold for configuring which higher-frequency words are
                          randomly down-sampled.
        :param workers:   use this many worker threads to train the model.
        """

        self.min_count = min_count
        self.window = window
        self.size = size
        self.sample = sample
        self.negative = negative
        self.workers = workers

        self.sentences = None
        self.model = None

    def train(self, n_epochs=20):

        # Import processed sentences.
        sources = {"PositiveExamples.txt": "TRAIN_POS",
                   "NegativeExamples.txt": "TRAIN_NEG",
                   "Unlabeled.txt": "TEST_UNSUP"}
        self.sentences = LabeledLineSentence(sources)

        # Build Doc2Vec vocabulary.
        self.model = Doc2Vec(min_count=self.min_count, window=self.window, size=self.size,
                             sample=self.sample, negative=self.negative, workers=self.workers)
        self.model.build_vocab(self.sentences.to_array())

        # Train model.
        for epoch in range(n_epochs):
            self.model.train(self.sentences.sentences_perm())

    def save(self, file_name="Doc2VecModel.d2v"):

        self.model.save("./" + file_name)

    def load(self, file_name="Doc2VecModel.d2v"):

        self.model = Doc2Vec.load("./" + file_name)

if __name__ == "__main__":

    d2v = Doc2VecModel()
    d2v.train()
    d2v.save()
    d2v.load()
    print d2v.model.docvecs["TRAIN_NEG_0"]
