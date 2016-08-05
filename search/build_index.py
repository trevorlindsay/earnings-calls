__author__ = 'trevorlindsay'


import os
import gzip
import cPickle as pickle
from collections import namedtuple, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize


Transcript = namedtuple('Transcript', ['company',
                                       'ticker',
                                       'date',
                                       'return_3days',
                                       'return_30days',
                                       'return_60days',
                                       'return_90days',
                                       'prepared',
                                       'QandA'])


class InvertedIndex(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)
        self._ids = set()

    def merge(self, dict2):
        for token, posting in dict2.iteritems():
            self[token].append(posting)
            self._ids |= set(posting[0])

    def updateIds(self):
        for posting in self.values():
            self._ids |= set([post[0] for post in posting])

    @property
    def ids(self):
        return self._ids



def loadTranscripts():
    with gzip.open('../data/transcripts.p.gz', 'rb') as f:
        return pickle.load(f)


def parseTranscript(transcript):

    assert isinstance(transcript, Transcript), \
        "transcript must be stored in custom namedtuple, not {}".format(type(transcript))

    text = transcript.prepared.append(transcript.QandA)
    id = "{ticker}-{year}-{month}-{day}".format(ticker=transcript.ticker.split(':')[-1],
                                                year=transcript.date.year,
                                                month=transcript.date.month,
                                                day=transcript.date.day)

    tokenizer = wordpunct_tokenize
    stemmer = PorterStemmer()
    index = dict()
    pos = 0

    for row in text:

        for i, token in enumerate(tokenizer(row.lower())):
            token = stemmer.stem(token)
            if token not in index and '|' not in token:
                index[token] = [id, [str(pos + i)]]
            elif '|' not in token:
                index[token][-1].append(str(pos + i))

        try:
            pos += (i + 1)
        except:
            pass

    return index


def buildIndex(transcripts):

    # Initiate InvertedIndex object (extends defaultdict)
    index = InvertedIndex()
    path = 'index/index.txt'
    indices = 1

    for key, transcript in transcripts.iteritems():

        print "{}, {}".format(key, transcript.company.encode('utf-8'))
        index.merge(parseTranscript(transcript))

        if key % 1000 == 0:
            filesize = writeIndexToFile(index, path)
            if filesize >= 1e+8:
                indices += 1
                path = 'index/index{}.txt'.format(indices)
                index = InvertedIndex()

    writeIndexToFile(index, path)
    return index


def writeIndexToFile(index, path='index/index.txt'):

    with open(path, 'wb') as f:
        for token, postings in index.iteritems():
            line = token + '||' + ';'.join(['{}:{}'.format(id, ','.join(posting))
                                            for id, posting in postings])
            f.write(line.encode('utf-8') + '\n')

    return os.path.getsize(path)


def readIndexFromFile(path):

    index = InvertedIndex()

    with open(path, 'rb') as f:
        for line in f:
            token, postings = line.strip().split('||')
            postings = [[id, locs.split(',')] for id, locs in
                        [posting.split(':') for posting in postings.split(';')]]
            index[token] = postings

    index.updateIds()
    return index


def updateIndex():

    from_scratch = False

    if os.path.isfile('index/index.txt'):
        index = readIndexFromFile()
    else:
        from_scratch = True

    if not from_scratch:
        for transcript in loadTranscripts():
            id = "{ticker}-{year}-{month}-{day}".format(ticker=transcript.ticker,
                                                        year=transcript.date.year,
                                                        month=transcript.date.month,
                                                        day=transcript.date.day)
            if id not in index.ids:
                index.merge(parseTranscript(transcript))
    else:
        create = input("No index exists! Would you like to build it from scratch? (y/n")
        if create.lower() == 'y':
            print "Creating index from scratch!"
            index = buildIndex(loadTranscripts())
        else:
            print "The index was not updated."
            index = None

    return index


if __name__ == '__main__':
    buildIndex(loadTranscripts())





