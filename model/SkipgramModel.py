import numpy as np
from collections import defaultdict
from text_generator import TextGenerator


class BiGramModel(object):


    def __init__(self, transcripts, train_size=0.75, target_period=3, max_vocab_size=None, min_count=50, verbose=True,
                 problem='classification'):

        # Initiate some necessary variables
        self._transcripts = transcripts
        self._vocab = defaultdict(dict)
        self._xtrain = list()
        self._ytrain = list()
        self._xtest = list()
        self._ytest = list()
        self._features = list()
        self._targets = list()
        self._train = np.random.choice(transcripts.keys(), int(len(transcripts.keys()) * train_size))
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.problem = problem
        self.stop_words = "a an the for which our by as we that on in with or is are also and of to you these " \
                          "from at last where will i now how have be per during about these this - was year " \
                          "were call it has us than so not like very but million quarter over new first " \
                          "third their would those there . ? ! , & % $ [ ] ( ) ; and/or your 's 're 'd 've".split()

        # Set the target period
        assert target_period in (3, 30, 60, 90), 'Invalid target period: {} days'.format(target_period)
        self.target_period = target_period

        # Set up training and testing splits
        train_index = np.random.choice(transcripts.keys(), int(len(transcripts.keys()) * train_size))
        self._train = {key: value for (key, value) in transcripts.iteritems() if key in train_index}

        test_index = [key for key in transcripts.keys() if key not in train_index]
        self._test = {key: value for (key, value) in transcripts.iteritems() if key in test_index}

        # Build the vocabulary
        self.build_train_vocab(verbose)
        self.finalize_vocab()
        self.TFIDF()


    def build_train_vocab(self, verbose):

        for i, key, transcript in enumerate(self._train.iteritems()):

            abnormal_return = self.get_target(transcript)

            if abnormal_return:
                self._ytrain.append(abnormal_return)
            else:
                continue

            # Tracks how often words appear in the transcript
            tracker = defaultdict(int)

            for paragraph in TextGenerator(transcript):

                # Remembers last word for bigrams
                last_word = None

                for word in paragraph:

                    # Ignore stop words
                    if word in self.stop_words:
                        continue

                    # Ignore numbers
                    try:
                        float(word)
                        continue
                    except ValueError:
                        if last_word:
                            tracker[' '.join([last_word, word])] += 1
                        tracker[word] += 1
                        last_word = word

            for word in tracker:
                self._vocab[word][key] = tracker.get(word)

            if verbose:
                if i % 250 == 0:
                    print("PROGRESS: at transcript #{}, keeping {} word types".format(i, len(self._vocab)))

            if i % 1000 == 0:
                self.trim_vocab(min_reduce=2)


    def build_test_vocab(self, verbose):

        test_vocab = defaultdict(int)

        for i, key, transcript in enumerate(self._test.iteritems()):

            abnormal_return = self.get_target(transcript)

            if abnormal_return:
                self._ytest.append(abnormal_return)
            else:
                continue

            # Tracks how often words appear in the transcript
            tracker = defaultdict(int)

            for paragraph in TextGenerator(transcript):

                # Remembers last word for bigrams
                last_word = None

                for word in paragraph:

                    # Ignore stop words
                    if word in self._vocab:
                        tracker[word] += 1

                    if last_word and ' '.join([last_word, word]) in self._vocab:
                        tracker[' '.join([last_word, word])] += 1

                    last_word = word

            for word in tracker:
                test_vocab[word][key] = tracker.get(word)

            if verbose:
                if i % 250 == 0:
                    print("TESTING: at transcript #{}".format(i))


    def TFIDF(self, transcripts, vocab, train=True):

        print("Computing TF-IDF...",)
        # Compute inverse document frequency
        N = len(transcripts)
        idf = defaultdict(float)
        for word in vocab:
            freq = len(vocab[word])
            idf[word] = np.log(float(N) / freq)

        # Reaugment dictionary so transcripts are first-order keys and words are second-order keys
        # Also set self._vocab to now just be the actual vocabulary
        if train:
            self._vocab = vocab.keys()
            self._xtrain = {doc: {word: vocab[word][doc] for word in vocab if doc in vocab[word]}
                              for doc in transcripts}
        else:
            self._xtest = {doc: {word: vocab[]}}

        for doc in self._features:
            max_freq = float(max(self._features[doc].values()))
            self._features[doc] = {word: ((count / max_freq) * 0.5 + 0.5) * idf[word] for word, count in
                                 self._features[doc].iteritems()}
        print("Done!")


    def trim_vocab(self, min_reduce):

        """ Trims all words that appear less than min_reduce times """

        start = len(self._vocab)
        for word in self._vocab.keys():
            if sum(self._vocab[word].values()) < min_reduce:
                del self._vocab[word]
        print("trimmed {} word types, {} word types remaining".format(start - len(self._vocab), len(self._vocab)))


    def finalize_vocab(self):

        while self.max_vocab_size and len(self._vocab) > self.max_vocab_size:
            self.trim_vocab(min_reduce=self.min_count)
            self.min_count += np.sqrt(np.sqrt(len(self.vocab)))

        print("collected {} word types".format(len(self._vocab)))


    def get_target(self, transcript):

        switcher = {3: transcript.return_3days,
                    30: transcript.return_30days,
                    60: transcript.return_60days,
                    90: transcript.return_90days}

        abnormal_return = switcher.get(self.target_period)

        if self.problem == 'regression':
            return abnormal_return
        elif abnormal_return < -0.10:
            return 0
        elif -0.10 <= abnormal_return < -0.05:
            return 1
        elif -0.05 <= abnormal_return < 0.05:
            return 2
        elif 0.05 <= abnormal_return < 0.10:
            return 3
        elif abnormal_return >= 0.10:
            return 4
        else:
            return None


    @property
    def transcripts(self):
        return self._transcripts

    @property
    def vocab(self):
        return self._vocab

    @property
    def inputs(self):
        return self._features

    @property
    def targets(self):
        return self._targets