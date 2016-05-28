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
        train_index = np.random.choice(transcripts.keys(), int(len(transcripts.keys()) * train_size), replace=False)
        self._train = {key: value for (key, value) in transcripts.iteritems() if key in train_index}
        self._test = {key: value for (key, value) in transcripts.iteritems() if key not in train_index}

        print("{} transcripts for training, {} for testing".format(len(self._train.keys()), len(self._test.keys())))

        # Build the vocabulary and generate features for train and test sets
        self.build_train_vocab(verbose)
        self.finalize_vocab()
        self._xtrain = self.TFIDF(self._train, self._vocab, train=True)

        test_vocab = self.build_test_vocab(verbose)
        self._xtest = self.TFIDF(self._test, test_vocab, train=False)

        print("Re-augmenting training and testing data.."),
        self._xtrain = np.asarray([np.asarray([doc[word] if word in doc else 0 for word in self.vocab])
                                   for doc in self._xtrain.values()])
        self._xtest = np.asarray([np.asarray([doc[word] if word in doc else 0 for word in self.vocab])
                                   for doc in self._xtest.values()])
        self._ytrain = np.asarray(self._ytrain)
        self._ytest = np.asarray(self._ytest)
        print("Done!")

    def build_train_vocab(self, verbose):

        for i, key in enumerate(self._train.keys()):

            transcript = self._train.get(key)
            abnormal_return = self.get_target(transcript)

            if abnormal_return:
                self._ytrain.append(abnormal_return)
            else:
                del self._train[key]
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
                if i % 250 == 0 and i != 0:
                    print("PROGRESS: at transcript #{}, keeping {} word types".format(i, len(self._vocab)))

            if i % 1000 == 0 and i != 0:
                self.trim_vocab(min_reduce=2)

    def build_test_vocab(self, verbose):

        test_vocab = defaultdict(dict)

        for i, key in enumerate(self._test.keys()):

            transcript = self._test.get(key)
            abnormal_return = self.get_target(transcript)

            if abnormal_return:
                self._ytest.append(abnormal_return)
            else:
                del self._test[key]
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
                if i % 250 == 0 and i != 0:
                    print("PROGRESS: preparing testing transcripts, at transcript #{}".format(i))

        return test_vocab


    def TFIDF(self, transcripts, vocab, train=True):

        print("Computing TF-IDF..."),

        # Compute inverse document frequency
        N = len(transcripts)
        idf = defaultdict(float)
        for word in vocab:
            freq = len(vocab[word])
            idf[word] = np.log(float(N) / freq)

        # Reaugment dictionary so transcripts are first-order keys and words are second-order keys
        # Also set self._vocab to now just be the actual vocabulary
        if train:
            new_vocab = vocab.keys()

        features = {doc: {word: vocab[word][doc] for word in vocab if doc in vocab[word]}
                        for doc in transcripts.keys()}

        for doc in features:
            if len(features[doc].values()) != 0:
                max_freq = float(max(features[doc].values()))
            else:
                continue
            features[doc] = {word: ((count / max_freq) * 0.5 + 0.5) * idf[word] for word, count in
                                 features[doc].iteritems()}
        if train:
            self._vocab = new_vocab

        print("Done!")
        return features


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
    def xtrain(self):
        return self._xtrain

    @property
    def xtest(self):
        return self._xtest

    @property
    def ytrain(self):
        return self._ytrain

    @property
    def ytest(self):
        return self._ytest