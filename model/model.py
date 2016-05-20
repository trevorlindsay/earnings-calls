from SkipgramModel import BiGramModel
from collections import namedtuple
import cPickle as pickle
import gzip
import logging

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

def load_transcripts():
    with gzip.open('../data/transcripts_debug.p.gz') as f:
        return pickle.load(f)


def init_model(transcripts):
    return BiGramModel(transcripts, target_period=30, max_vocab_size=25000, min_count=50)