## Data
Preprocessed data is stored in <code>data/transcripts.gz</code> (Pickle file). The raw data is in <code>data/raw_data.gz</code> (CSV files).

In order to open the Pickle file, you must include the following two lines of code:
<pre><code>from collections import namedtuple
Transcript = namedtuple('Transcript', ['company', 'ticker', 'date', 'prepared', 'QandA'])</code></pre>