import gensim, logging
import cPickle as pickle
from collections import namedtuple
import string


# Some required lines of code
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
Transcript = namedtuple('Tramscript', ['company', 'ticker', 'date', 'abnoromal_return', 'prepared', 'QandA'])
LabeledParagraph = namedtuple('LabeledParagraph', ['words', 'tagmodels'])


class LabeledDocuments(object):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        for id, doc in enumerate(self.data):
            doc = doc.apply(format_string)
            for uid, paragraph in enumerate(doc):
                yield LabeledParagraph(words=paragraph.split(), tags=['SENT_{}{}'.format(id, uid)])


def load_data(filename='../data/transcripts.p'):
    return pickle.load(open(filename, 'r'))


def format_string(x):
    exclude = set(string.punctuation)
    try:
        x = ''.join(ch.lower() for ch in x if ch not in exclude)
    except:
        pass
    return x

if __name__ == '__main__':
    transcripts = load_data()
    data = [transcript.prepared for transcript in transcripts.values()]
    model = gensim.models.Doc2Vec(LabeledDocuments(data), size=100, min_count=25, verbose=True)



