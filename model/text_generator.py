from nltk.tokenize import word_tokenize
from unidecode import unidecode

class TextGenerator(object):

    def __init__(self, transcript, iter_prepared=True):

        self._transcript = transcript
        self._company, self._ticker, self._date = (transcript.company, transcript.ticker, transcript.date)
        self._abnormal_returns = \
            (transcript.return_3days, transcript.return_30days, transcript.return_60days, transcript.return_90days)
        self._prepared, self._QandA = (transcript.prepared, transcript.QandA)
        self._iter_prepared = iter_prepared

    @property
    def transcript(self):
        return self._transcript

    @property
    def company(self):
        return self._company

    @property
    def ticker(self):
        return self._ticker

    @property
    def date(self):
        return self._date

    @property
    def abnormal_returns(self):
        return self._abnormal_returns

    @property
    def prepared(self):
        return self._prepared

    @property
    def QandA(self):
        return self._QandA

    @property
    def iter_prepared(self):
        return self._iter_prepared

    @iter_prepared.setter
    def iter_prepared(self, iter_prepared):
        self._iter_prepared = iter_prepared

    def __iter__(self):
        if self.iter_prepared:
            for line in self.prepared:
                yield word_tokenize(unidecode(line.lower()))
        else:
            for line in self.QandA:
                yield word_tokenize(line.lower())
