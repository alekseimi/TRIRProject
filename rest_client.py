import requests
import pandas as pd
import util


currency_symbols = {
    'bitcoin': 'BTC',
    'komodo': 'KMD',
    'eos': 'EOS',
    'cardano': 'ADA',
    'lisk': 'LSK',
    '0x1': 'ZRX'
}


def parse_contributors(data):
    contributor_number_list = [0] * 52
    for i in range(0, len(data)):
        json_weeks = data[i]
        iter_ = 0
        for week in json_weeks['weeks'][-52:]:
            if week['c'] != 0:
                contributor_number_list[iter_] = contributor_number_list[iter_] + 1
            iter_ = iter_ + 1
    return contributor_number_list


def parse_activities(data):
    commit_number_list = []
    for i in range(0, len(data)):
        json_week = data[i]
        commit_number_list.append(json_week['total'])
    return commit_number_list


def parse_frequencies(data):
    weeks_list = []
    additions_number_list = []
    deletions_number_list = []

    for week in data[-52:]:
        weeks_list.append(week[0])
        additions_number_list.append(week[1])
        deletions_number_list.append(abs(week[2]))

    frequencies_list = [weeks_list,
                        additions_number_list,
                        deletions_number_list]
    return frequencies_list


def parse_historical_prices(data, currency):
    currency_symbol = currency_symbols[currency]
    currency_price_list = [data[currency_symbol]['BTC'],
                           data[currency_symbol]['EUR']]
    return currency_price_list


def request_contributors(url, user, repo, token):
    request_url = url + user + '/' + repo + '/' + 'stats/contributors'
    r = requests.get(request_url, headers={'Authorization': 'access_token ' + token}).json()
    return r


def request_activity(url, user, repo, token):
    request_url = url + user + '/' + repo + '/' + 'stats/commit_activity'
    r = requests.get(request_url, headers={'Authorization': 'access_token ' + token}).json()
    return r


def request_frequency(url, user, repo, token):
    request_url = url + user + '/' + repo + '/' + 'stats/code_frequency'
    r = requests.get(request_url, headers={'Authorization': 'access_token ' + token}).json()
    return r


def request_historical_price(currency, timestamp):
    currency_symbol = currency_symbols[currency]
    request_url = 'https://min-api.cryptocompare.com/data/pricehistorical?' \
                  'fsym=' + currency_symbol + '&tsyms=BTC,EUR&ts='+str(timestamp)
    r = requests.get(request_url).json()
    return r


def get_currency_data(user, repo):
    url = 'https://api.github.com/repos/'
    token = util.get_auth_token()

    user = user
    repo = repo
    json_contributors = request_contributors(url, user, repo, token)
    contributors_list = parse_contributors(json_contributors)

    json_activity = request_activity(url, user, repo, token)
    activity_list = parse_activities(json_activity)

    json_frequency = request_frequency(url, user, repo, token)
    frequencies_list = parse_frequencies(json_frequency)

    columns = ['timestamp', 'contributors', 'commits', 'additions', 'deletions', 'BTC_price', 'EUR_price']
    df = pd.DataFrame(columns=columns)

    for i in range(0, 52):
        json_historical_prices = request_historical_price(repo,  frequencies_list[0][i])
        historical_prices = parse_historical_prices(json_historical_prices, repo)
        row = [frequencies_list[0][i], contributors_list[i], activity_list[i],
               frequencies_list[1][i], frequencies_list[2][i], historical_prices[0], historical_prices[1]]
        df.loc[i] = row
    return df








