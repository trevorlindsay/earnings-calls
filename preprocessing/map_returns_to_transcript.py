import cPickle as pickle
from collections import namedtuple
import pandas as pd
import os
from collections import deque
import gzip


Transcript = namedtuple('Transcript', ['company',
                                       'ticker',
                                       'date',
                                       'return_3days',
                                       'return_30days',
                                       'return_60days',
                                       'return_90days',
                                       'prepared',
                                       'QandA'])


def concat_returns(folder='../data'):

    data_folder = os.path.join(folder)
    files = deque([f for f in os.listdir(os.path.join(data_folder)) if f.find('abnormal_returns') != -1])

    returns = pd.read_csv(os.path.join(data_folder, files.popleft()))
    for file in files:
        frame = pd.read_csv(os.path.join(data_folder, file))
        returns = returns.append(frame)

    # Only return the ones without errors
    return returns[pd.isnull(returns['error_code'])]


def map_returns(returns, transcripts):


    new_transcripts = {}
    counter = 1

    for row in returns.itertuples():

        company, ticker, date, _, _, _, _, prepared, QandA = transcripts[row[1]]

        new_transcripts[counter] = Transcript(company=company,
                                              ticker=ticker,
                                              date=date,
                                              return_3days=row.return_3days,
                                              return_30days=row.return_30days,
                                              return_60days=row.return_60days,
                                              return_90days=row.return_90days,
                                              prepared=prepared,
                                              QandA=QandA)

        counter += 1
        print 'New Key {}'.format(counter)


    return new_transcripts


if __name__ == '__main__':

    returns = concat_returns()

    with gzip.open('../data/transcripts.p.gz', 'rb') as f:
        transcripts = pickle.load(f)

    new_transcripts = map_returns(returns, transcripts)
    pickle.dump(new_transcripts, open('../data/transcripts.p', 'wb'))