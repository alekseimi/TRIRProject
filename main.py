from rest_client import get_currency_data

#df_btc = get_currency_data('bitcoin', 'bitcoin')
#df_btc.to_csv('files/data_bitcoin.csv')

#df_kmd = get_currency_data('jl777', 'SuperNET')
#df_kmd.to_csv('files/data_komodo.csv')

#df_eos = get_currency_data('eosio', 'eos')
#df_eos.to_csv('files/data_eos.csv')

#df_ada = get_currency_data('input-output-hk', 'cardano-sl')
#df_ada.to_csv('files/data_ada.csv')

#df_lisk = get_currency_data('liskHQ', 'lisk')
#df_lisk.to_csv('files/data_lisk.csv')

df_zrx = get_currency_data('0xProject', '0x.js')
df_zrx.to_csv('files/data_zrx.csv')

