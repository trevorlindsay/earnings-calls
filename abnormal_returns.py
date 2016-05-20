import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
from pandas.tseries.offsets import BDay
from datetime import datetime
import cPickle as pickle
from collections import namedtuple
import csv
import gzip
import os


# Must be included to read contents of pickle file
Transcript = namedtuple('Transcript', ['company',
                                       'ticker',
                                       'date',
                                       'return_3days',
                                       'return_30days',
                                       'return_60days',
                                       'return_90days',
                                       'prepared',
                                       'QandA'])


def load_data(folder='data', file='transcripts.p.gz'):
    filepath = os.path.join(folder, file)
    print 'Loading {}'.format(filepath)
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)


def get_tbill_historical():

    """ Reads in historical interest rate data for the 4-week Treasury Bill. """

    tbill = pd.read_csv('data/historical_tbill.csv')
    tbill.fillna(method='pad', inplace=True)
    tbill['rate'] = tbill.rate.map(lambda x: x / 100.)
    tbill['date'] = pd.to_datetime(tbill.date)
    tbill.set_index(keys='date', drop=True, inplace=True)
    tbill.sort_index(inplace=True)

    return tbill


def calculate_abnormal_returns(transcripts,
                               output,
                               time_periods=(3, 30, 60, 90),
                               tbill=get_tbill_historical):

    """

    Calculates the abnormal returns for each company.
    Time Periods (where t = date of earnings call):
    1. (t - 1) -> (t + 3)
    2. (t - 1) -> (t + 30)
    3. (t - 1) -> (t + 60)
    4. (t - 1) -> (t + 90)

    Abnormal Returns = Actual Return - E[Return]
    E[Return] = (CAPM) = Return on Risk-Free Asset + Beta of Stock * (Return on Market - Return on Risk-Free Asset)
    Where:
        Return on Risk-Free Asset = interest rate on treasury bill over time period
        Beta of Stock = Cov(daily returns of stock, daily returns of market) / Var(daily returns of market)
        Return on Market = return of S&P 500 over time period

    """

    # Load in S&P 500 and Treasury Bill data
    tbill = tbill()

    # Column headers
    output.writerow(['key', 'name', 'ticker', 'date', 'error_code', 'return_3days', 'return_30days', 'return_60days', 'return_90days'])

    for key in transcripts.keys():

        # Last stopping point
        if key <= 56810:
            continue

        company = transcripts[key]
        print 'Key: {}, Ticker: {}'.format(key, company.ticker)

        # Ensure date is not in the future or too far in the past
        if company.date >= datetime.today() or company.date < datetime(2001, 7, 31):
            output.writerow([key,
                             company.company.encode('utf-8'),
                             company.ticker,
                             company.date,
                             'DATE_ERROR'])
            print 'Key: {}'.format(key), 'DATE_ERROR'
            continue

        # Hack to check if ticker looks correct
        if company.ticker != company.ticker.upper():
            output.writerow([key,
                             company.company.encode('utf-8'),
                             company.ticker,
                             company.date,
                             'TICKER_ERROR'])
            print 'Key: {}'.format(key), 'TICKER_ERROR'
            continue

        # Compute start date and end dates for various time periods
        start = company.date.date() - BDay(1)
        ends = [company.date.date() + BDay(days) for days in time_periods]

        # Drop the time periods that are in the future
        ends = [end for end in ends if end < datetime.today()]

        # Adjust start and ends based on whether the market was open
        # Also, retrieve beginning and end values of S&P 500 to calculate market returns
        index_prices, _ = get_stock_prices(symbol='^GSPC', start=start, ends=ends)

        # Use returns over the prior 30 days to calculate beta
        beta_start = start - BDay(31)
        beta_end = [start - BDay(1)]

        # Approximate 1-week risk free rate by diving 4-week rate by 4
        risk_free = tbill.rate[start] / 4

        symbol = company.ticker.split(':')[-1]

        # Pull price history for stock
        stock_prices, _ = get_stock_prices(symbol=symbol, start=start, ends=ends)
        beta_stock_prices, beta_stock_returns = get_stock_prices(symbol=symbol, start=beta_start, ends=beta_end)

        # Pull index returns for beta
        _, beta_index_returns = get_stock_prices(symbol='^GSPC', start=beta_start, ends=beta_end)

        # Calculate beta
        beta = calculate_beta(beta_stock_returns, beta_index_returns)
        if beta is None:
            output.writerow([key,
                             company.company.encode('utf-8'),
                             company.ticker,
                             company.date,
                             'BETA_ERROR'])
            print 'Key: {}'.format(key), 'BETA_ERROR'
            continue

        abnormal_returns = []

        for i, end in enumerate(ends):

            # If no stock price history was found for time period, return None
            if stock_prices[i] is None:
                abnormal_returns.append(None)
                continue

            try:
                market_return = index_prices[i][-1] / index_prices[i][0] - 1
            except IndexError:
                print 'Key: {}'.format(key), 'MARKET_RETURN_ERROR'
                continue

            expected_return = risk_free + beta * (market_return - risk_free)
            stock_return = stock_prices[i][-1] / stock_prices[i][0] - 1
            abnormal_returns.append(stock_return - expected_return)

        output.writerow([key,
                         company.company.encode('utf-8'),
                         company.ticker,
                         company.date,
                         None] + abnormal_returns)


def get_stock_prices(symbol, start, ends):

    """ Fetches stock prices for time periods using Yahoo! API """

    data = []
    for end in ends:
        try:
            stock_prices = web.DataReader(symbol, 'yahoo', start, end)['Adj Close']
            daily_stock_returns = stock_prices.pct_change()[start : end]
            data.append((stock_prices, daily_stock_returns))
        except RemoteDataError:
            print 'Symbol: {}'.format(symbol), 'API_ERROR'
            data.append((None, None))

    return zip(*data)


def calculate_beta(beta_stock_returns, beta_index_returns):

    """ Calculate beta of stock and market"""

    try:
        covariance = np.cov(beta_stock_returns[0].values[1:], beta_index_returns[0].values[1:])[0][1]
        variance = np.var(beta_index_returns[0].values[1:])
        return covariance / variance
    except (ValueError, AttributeError):
        return None


if __name__ == '__main__':

    with open('data/abnormal_returns7.csv', 'wb') as csvfile:
        w = csv.writer(csvfile)
        calculate_abnormal_returns(load_data(), output=w)