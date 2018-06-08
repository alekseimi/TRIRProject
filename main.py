from rest_client import get_currency_data

user = 'bitcoin'
repo = 'bitcoin'
df = get_currency_data('bitcoin', 'bitcoin')
df.to_csv('files/data_bitcoin.csv')

