from rest_client import get_currency_data
import pandas as pd
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stats

#BTC, EOS, ADA, LSK, ZRX, SKY
#df_bitcoin, df_eos, df_ada, df_lisk, df_zrx, df_sky

#df_btc = get_currency_data('bitcoin', 'bitcoin')
#df_btc.to_csv('files/data_bitcoin.csv')

#df_eos = get_currency_data('eosio', 'eos')
#df_eos.to_csv('files/data_eos.csv')

#df_ada = get_currency_data('input-output-hk', 'cardano-sl')
#df_ada.to_csv('files/data_ada.csv')

#df_lisk = get_currency_data('liskHQ', 'lisk')
#df_lisk.to_csv('files/data_lisk.csv')

#df_zrx = get_currency_data('0xProject', '0x.js')
#df_zrx.to_csv('files/data_zrx.csv')

#df_sky = get_currency_data('skycoin', 'skycoin')
#df_sky.to_csv('files/data_skycoin.csv')

df_eos = pd.read_csv('files/data_eos.csv', sep=',', decimal='.', index_col=0)
df_bitcoin = pd.read_csv('files/data_bitcoin.csv', sep=',', decimal='.',  index_col=0)
df_ada = pd.read_csv('files/data_ada.csv', sep=',', decimal='.',  index_col=0)
df_lisk = pd.read_csv('files/data_lisk.csv', sep=',', decimal='.', index_col=0)
df_zrx = pd.read_csv('files/data_zrx.csv', sep=',', decimal='.',  index_col=0)
df_sky = pd.read_csv('files/data_skycoin.csv', sep=',', decimal='.', index_col=0)

df_list = [df_bitcoin, df_eos, df_ada, df_lisk, df_zrx, df_sky]
df = reduce(lambda left, right: pd.merge(left, right, on='timestamp'), df_list)
df.reset_index(drop=True, inplace=True)
sns.set_style("darkgrid")

#BTC, LSK, SKY, ZRX, EOS, ADA
average_commits = [df['commits_BTC'].mean(), df['commits_LSK'].mean(), df['commits_SKY'].mean(),
                   df['commits_ZRX'].mean(), df['commits_EOS'].mean(), df['commits_ADA'].mean()]

average_additions = [df['additions_BTC'].mean(), df['additions_LSK'].mean(), df['additions_SKY'].mean(),
                   df['additions_ZRX'].mean(), df['additions_EOS'].mean(), df['additions_ADA'].mean()]

average_deletions = [df['deletions_BTC'].mean(), df['deletions_LSK'].mean(), df['deletions_SKY'].mean(),
                     df['deletions_ZRX'].mean(), df['deletions_EOS'].mean(), df['deletions_ADA'].mean()]

average_contributors = [df['contributors_BTC'].mean(), df['contributors_LSK'].mean(), df['contributors_SKY'].mean(),
                     df['contributors_ZRX'].mean(), df['contributors_EOS'].mean(), df['contributors_ADA'].mean()]

x_tick_label = ['BTC', 'LSK', 'SKY', 'ZRX', 'EOS', 'ADA']

#average_deletions = [df['commits_BTC'].mean(), df['commits_LSK'].mean(), df['commits_SKY'].mean(),
#                  df['commits_ZRX'].mean(), df['commits_EOS'].mean(), df['commits_ADA'].mean()]

#average_commits = [df['commits_BTC'].mean(), df['commits_LSK'].mean(), df['commits_SKY'].mean(),
#                   df['commits_ZRX'].mean(), df['commits_EOS'].mean(), df['commits_ADA'].mean()]


#Commits
#Lineplot
btc_plot = plt.plot(df.index, df.commits_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.commits_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.commits_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.commits_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.commits_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.commits_SKY, label="SKY")

plt.legend(handles=[btc_plot[0], ada_plot[0], lsk_plot[0], eos_plot[0], zrx_plot[0], sky_plot[0]])
plt.suptitle("Število commitov po tednih")
plt.ylabel("Commiti", fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()

#histogram of averages
sns.distplot(df['commits_BTC'], kde=False)
plt.show()

#ADDITIONS
#lineplot
btc_plot = plt.plot(df.index, df.additions_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.additions_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.additions_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.additions_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.additions_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.additions_SKY, label="SKY")

plt.legend(handles=[btc_plot[0], ada_plot[0], lsk_plot[0], eos_plot[0], zrx_plot[0], sky_plot[0]])
plt.suptitle("Število dodatkov po tednih")
plt.ylabel("Dodatki", fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()

#DELETIONS
btc_plot = plt.plot(df.index, df.deletions_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.deletions_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.deletions_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.deletions_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.deletions_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.deletions_SKY, label="SKY")

plt.legend(handles=[btc_plot[0], ada_plot[0], lsk_plot[0], eos_plot[0], zrx_plot[0], sky_plot[0]])
plt.suptitle("Število izbrisov po tednih")
plt.ylabel("Deletions", fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()


fig, ax = plt.subplots()



#CONTRIBUTORS

#Pregled števila commit-ov skozi čas
#Pregled števila contributor-jev skozi čas
#Napoved cene na podlagi števila commitov, števila contributorje in števila vrstic kode
#Pregled gibanje cene kriptovalute skozi čas




