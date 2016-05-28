from text_generator import TextGenerator
from stop_words import stop_words as SW
import numpy as np
import gzip
import cPickle as pickle
from collections import namedtuple, defaultdict
import logging
from pprint import pprint


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

Transcript = namedtuple('Transcript', ['company',
                                       'ticker',
                                       'date',
                                       'return_3days',
                                       'return_30days',
                                       'return_60days',
                                       'return_90days',
                                       'prepared',
                                       'QandA'])

def load_transcripts(debug=False):
    file = '../data/transcripts_debug.p.gz' if debug else '../data/transcripts.p.gz'
    with gzip.open(file) as f:
        return pickle.load(f)


class BiGramModel(object):

    def __init__(self, transcripts, max_vocab_size=None, min_count=5,
                 task='classification', target_period=3, train_size=0.75):

        self._transcripts = transcripts
        self._vocab = set()
        self._train = defaultdict(int)
        self._test = defaultdict(int)
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.task = task
        self.target_period = target_period
        self.train_size = train_size
        self.stop_words = SW()

        train_index = np.random.choice(transcripts.keys(), int(len(transcripts.keys()) * train_size), replace=False)
        for doc, info in transcripts.iteritems():
            if doc in train_index:
                self._train[doc] = info
            else:
                self._test[doc] = info

        logging.debug('{} transcripts for training, {} for testing'.format(len(self.train), len(self.test)))

        logging.info('Building vocabulary...')
        self._xtrain, self._ytrain = self.build_vocab(training=True)
        self._xtest, self._ytest = self.build_vocab(training=False)

        logging.info('Computing TF_IDF...')
        self._xtest = self.tf_idf(self.xtest)
        self._xtrain = self.tf_idf(self.xtrain)



    def build_vocab(self, verbose=True, training=True):

        word_count = defaultdict(dict)
        targets = list()

        if training:
            transcripts = self.train
        else:
            transcripts = self.test

        for i, doc in enumerate(transcripts.keys()):

            transcript = transcripts.get(doc)
            abnormal_return = self.get_target(transcript)

            if abnormal_return:
                targets.append(abnormal_return)
            else:
                del transcripts[doc]
                continue

            # Tracks how often words appear in the transcript
            tracker = defaultdict(int)

            for paragraph in TextGenerator(transcript):

                last_word = None
                for word in paragraph:

                    if word in self.stop_words:
                        continue

                    try:
                        float(word)
                        continue

                    except ValueError:

                        if last_word:
                            bigram = ' '.join([last_word, word])
                            tracker[bigram] += 1

                        tracker[word] += 1

                        if training and last_word:
                            self._vocab.add(bigram)
                            self._vocab.add(word)
                        elif training:
                            self._vocab.add(word)

                        last_word = word

            word_count[doc] = tracker

            if verbose:
                if i % 250 == 0:
                    logging.info('PROGRESS: at transcript #{}, keeping {} word types'.format(i, len(self.vocab)))

            if i % 1000 == 0 and training:
                word_count = self.trim_vocab(min_reduce=2, word_count=word_count)

        if training:
            logging.info('Finalizing vocabulary...')
            word_count = self.finalize_vocab(word_count)

        return word_count, targets


    def get_target(self, transcript):

        switcher = {3: transcript.return_3days,
                    30: transcript.return_30days,
                    60: transcript.return_60days,
                    90: transcript.return_90days}

        abnormal_return = switcher.get(self.target_period)
        if self.task == 'regression':
            return abnormal_return
        elif abnormal_return < 0:
            return 0
        elif abnormal_return >= 0:
            return 1
        else:
            return None


    def trim_vocab(self, min_reduce, word_count):

        start = len(self.vocab)
        drop_words = list()
        for i, word in enumerate(self.vocab):
            total = sum([words[word] for doc, words in word_count.iteritems()])
            if total < min_reduce:
                drop_words.append(word)

        for doc, words in word_count.iteritems():
            for word in drop_words:
                try:
                    del word_count[doc][word]
                except KeyError:
                    continue

        print len(drop_words)
        for word in drop_words:
            self._vocab.remove(word)

        logging.debug('Trimmed {} word types, {} word types remaining'.format(len(drop_words), start - len(self.vocab)))
        return word_count


    def finalize_vocab(self, word_count):

        while self.max_vocab_size and len(self.vocab) > self.max_vocab_size:
            word_count = self.trim_vocab(min_reduce=self.min_count, word_count=word_count)
            self.min_count += int(np.sqrt(np.sqrt(len(self.vocab))))

        logging.debug('Collected {} word types'.format(len(self.vocab)))
        return word_count


    def tf_idf(self, word_count):

        # Compute inverse document frequency
        N = float(len(word_count))
        idf = defaultdict(float)
        for word in self.vocab:
            freq = np.count_nonzero([words[word] for doc, words in word_count.iteritems()])
            idf[word] = np.log(N / freq)

        for doc in word_count:
            if len(word_count[doc].values()) != 0:
                max_freq = float(max(word_count[doc].values()))

            if max_freq == 0:
                continue
            else:
                word_count[doc] = {word: ((count / max_freq) * 0.5 + 0.5) * idf[word]
                                   for word, count in word_count[doc].iteritems()}

        return word_count

    @property
    def transcripts(self):
        return self._transcripts

    @property
    def vocab(self):
        return self._vocab

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

    @property
    def xtrain(self):
        return self._xtrain

    @property
    def ytrain(self):
        return self._ytrain

    @property
    def xtest(self):
        return self._xtest

    @property
    def ytest(self):
        return self._ytest


if __name__ == '__main__':
    transcripts = load_transcripts(debug=True)
    m = BiGramModel(transcripts, 50000, 5)