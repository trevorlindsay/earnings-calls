import glob
from multiprocessing import Process

from datetime import datetime
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

from build_index import readIndexFromFile


def search(ngrams, index, path, counts, id):

    print 'Searching {}'.format(path.split('/')[-1])

    # If 'Graph!' button was hit with nothing in box
    if ngrams == '':
        return None

    if len(ngrams) > 1:
        ngrams = ngrams.replace(', ', ',').encode('utf-8').lower().split(',')
    else:
        ngrams = ngrams.encode('utf-8').lower()

    ngram_count = {ngram: defaultdict(int) for ngram in ngrams}
    stemmer = PorterStemmer()

    for ngram in ngrams:

        transcripts = list()

        for word in ngram.split():

            # Get stem of word
            word = stemmer.stem(word)

            try:
                # Get set of books the word appears in
                transcripts.append(set([posting[0] for posting in index[word]]))
            except:
                # If the word is not in the index
                pass

        # Get the set of transcripts in which all words in the ngram appear
        transcripts = set.intersection(*transcripts) if len(transcripts) > 0 else set()

        for transcript in transcripts:

            year = int(transcript.split('-')[1])
            month = int(transcript.split('-')[2])
            day = int(transcript.split('-')[3])
            date = datetime(year, month, day)
            locs = []

            # For each transcript, get all of the locations of where the words in the ngram appear
            for word in ngram.split():
                word = stemmer.stem(word)
                locs.extend([posting[1] for posting in index[word] if posting[0] == transcript])

            # Check if the words are next to each other
            # e.g. ngram = 'very high profit margin' and the positions of the words are [[2,10] [3], [4,8,12,29], [5]]
            # This line of code will shift the position of each word over by its distance from the
            # beginning of the ngram to produce new positions [[2,10], [2], [2,6,10,29], [2]]
            # Then I take the intersection of these positions -- if it's not empty,
            # then the ngram appears in the transcript
            locs = [set([int(pos) - i for pos in loc]) for i, loc in enumerate(locs)]
            ngram_count[ngram][date] += len(set.intersection(*locs))

    counts[id] = ngram_count
    print 'Finished searching {}'.format(path.split('/')[-1])


def main(ngrams):

    indices = glob.glob('index/index*')
    processes = []
    counts = dict()

    for i, path in enumerate(indices):

        print 'Loading {}'.format(path.split('/')[-1])
        index = readIndexFromFile(path)
        process = Process(target=search, args=(ngrams, index, path, counts, i))
        processes.append(process)
        process.start()

    alive = True
    while alive:
        alive = sum([process.is_alive() for process in processes])

    print counts


main('profit margin, unexpected loss')

