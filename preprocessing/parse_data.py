import pandas as pd
from bs4 import BeautifulSoup
from collections import namedtuple
import warnings
import sys
import os
import gzip


# Filter warnings
warnings.filterwarnings(action='ignore')

# Note: this namedtuple must be defined in order for the pickled file to be opened
Transcript = namedtuple('Transcript', ['company',
                                       'ticker',
                                       'date',
                                       'return_3days',
                                       'return_30days',
                                       'return_60days',
                                       'return_90days',
                                       'prepared',
                                       'QandA'])


def clean_data(folder='../data', subfolder='../raw_data'):

    """ Cleans the scraped data by removing all quotes and empty lines. """

    data_folder = os.path.join(folder)

    w = open(os.path.join(data_folder, 'clean_data.csv'), 'wb')
    w.write('text\n') # Header for file

    raw_data_folder = os.path.join(data_folder, subfolder)
    files = [os.path.join(raw_data_folder, f) for f in os.listdir(raw_data_folder) if f[-2:] == 'gz']

    for file in files:

        print 'Cleaning {}...'.format(file)

        with gzip.open(file, 'rb') as f:
            csvfile = f.read().split('\r\n')
            for i, line in enumerate(csvfile):
                # Skip the 1st line (column name)
                if i == 0:
                    continue
                # Remove all quotes
                line = line.replace('"', '')
                # Filter out lines that are blank
                if len(line) > 18 or line == 'END OF TRANSCRIPT':
                    # Use quotes to ensure commas in text do not cause problems
                    w.write('"{}"'.format(line))
                    w.write('\n')
    w.close()


def split_transcripts(folder='../data', file='../clean_data.csv'):

    """

    Splits the data file into namedtuples, each with a different transcript. The tuples are stored in a dictionary.

    Format of namedtuple:
    Transcript(company=<string>,
               ticker=<string>,
               date=<timestamp>,
               prepared=<dataframe>,
               QandA=<dataframe>)

    """

    filepath = os.path.join(folder, file)
    df = pd.read_csv(filepath, verbose=True)

    end_of_transcripts = df[df.text == 'END OF TRANSCRIPT'].index
    print 'Total Number of Transcripts: {}'.format(len(end_of_transcripts))

    transcripts = {}
    last_end = -1
    n_transcript = 1

    for i, end in enumerate(end_of_transcripts):

        sys.stdout.write("Transcripts Completed: %d%%   \r" % (100 * float(i) / len(end_of_transcripts)))
        sys.stdout.flush()

        transcript = df[(last_end + 1) : end].reset_index(drop=True)
        last_end = end

        # Remove transcripts without required text (e.g. ones that reference an audio call only)
        if len(transcript) <= 4:
            continue

        # Extract company name from first line of transcript
        company = BeautifulSoup(transcript.iloc[0].values[0]).get_text()

        # Remove transcripts where the first line does not end with a closing parenthesis (indicates end of ticker)
        if len(company) < 1:
            continue

        if company[-1] != ')':
            continue

        # Extract the ticker from the company name
        open_paren = company.rfind('(')
        close_paren = company.find(')', open_paren)
        ticker = company[open_paren + 1 : close_paren]

        # Extract the date of the call from the third line of the transcript
        date =  BeautifulSoup(transcript.iloc[2].values[0]).get_text()

        # Remove transcripts with dates that cannot easily be converted into timestamps (for whatever reason)
        try:
            date = pd.to_datetime(date)
        except ValueError:
            # Uncomment the line below for examples of dates that are in an incorrect format
            # print date
            continue

        # Remove transcripts with improperly tagged Q&A sections
        try:
            begin_q_and_a = transcript[transcript.text.str.contains('id=question-answer-session')].index[0]
        except IndexError:
            continue

        # Split the remaining text into prepared remarks and the Q&A session
        q_and_a = transcript[begin_q_and_a : ].text.map(lambda x: BeautifulSoup(x).get_text())
        prepared = transcript[3 : begin_q_and_a].text.map(lambda x: BeautifulSoup(x).get_text())

        # Store namedtuple in a dictionary, keys range from 1 to the number of transcripts
        transcripts[n_transcript] = Transcript(company=company,
                                               ticker=ticker,
                                               date=date,
                                               return_3days=None,
                                               return_30days=None,
                                               return_60days=None,
                                               return_90days=None,
                                               prepared=prepared,
                                               QandA=q_and_a)
        n_transcript += 1


    print 'Transcripts Remaining after Filtering: {}'.format(len(transcripts.keys()))
    return transcripts


def main():

    # clean_data()
    transcripts = split_transcripts()

    # Uncomment below for an example of output
    # print transcripts[1]

    # Save transcripts dictionary to pickle file for easy re-opening
    import cPickle as pickle

    folder = '../data'
    filename = 'transcripts.p'
    path = os.path.join(folder, filename)

    pickle.dump(transcripts, open(path, 'wb'))


if __name__ == '__main__':
    main()

