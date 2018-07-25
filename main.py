from rest_client import get_currency_data
import pandas as pd
from functools import reduce
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
import matplotlib.pyplot as plt

def predict_regression(input, output, naziv_boxplot):
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.33, random_state=0)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_result = lr.predict(x_test)

    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    dtr_result = dtr.predict(x_test)

    lin_vec = SVR()
    lin_vec.fit(x_train, y_train)
    lin_vec_result = lin_vec.predict(x_test)

    df_a = pd.DataFrame({'y_test': y_test, 'lr_result': lr_result,
                         'dtr_result': dtr_result, 'lin_vec_result': lin_vec_result})

    list_val = ('mean absolute error', 'metoda regresije')
    df_boxplot_mae = pd.DataFrame([[metrics.mean_absolute_error(lr_result, y_test), 'linearna regresija'],
                           [metrics.mean_absolute_error(dtr_result, y_test), 'regresijsko drevo'],
                           [metrics.mean_absolute_error(lin_vec_result, y_test), 'regresija SVM']], columns=list_val)
    sns.barplot(x="metoda regresije", y="mean absolute error", data=df_boxplot_mae, palette="Reds").set_title(naziv_boxplot)
    plt.xticks(rotation=0)
    plt.show()

    list_val = ('mean squared error', 'metoda regresije')
    df_boxplot_mse = pd.DataFrame([[metrics.mean_squared_error(lr_result, y_test), 'linearna regresija'],
                                   [metrics.mean_squared_error(dtr_result, y_test), 'regresijsko drevo'],
                                   [metrics.mean_squared_error(lin_vec_result, y_test), 'regresija SVM']],
                                  columns=list_val)
    sns.barplot(x="metoda regresije", y="mean squared error", data=df_boxplot_mse, palette="Reds").set_title(naziv_boxplot)
    plt.xticks(rotation=0)
    plt.show()

    list_val = ('r^2', 'metoda regresije')
    df_boxplot_r2 = pd.DataFrame([[metrics.r2_score(lr_result, y_test), 'linearna regresija'],
                               [metrics.r2_score(dtr_result, y_test), 'regresijsko drevo'],
                               [metrics.r2_score(lin_vec_result, y_test), 'regresija SVM']],
                              columns=list_val)
    sns.barplot(x="metoda regresije", y="r^2", data=df_boxplot_r2, palette="Reds").set_title(naziv_boxplot)
    plt.xticks(rotation=0)
    plt.show()


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


currency_abbr_list = ['BTC', 'ADA', 'LSK', 'EOS', 'ZRX', 'SKY']


#Pregled števila commitov, dodatkov in izbrisov skozi čas
#Commits
#Lineplot

'''


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

#seaborn distplot commitov
for currency_abbr in currency_abbr_list:
    sns.distplot(df['commits_'+currency_abbr], kde=False)
    plt.show()

#ADDITIONS
#lineplot
btc_plot = plt.plot(df.index, df.additions_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.additions_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.additions_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.additions_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.additions_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.additions_SKY, label="SKY")

plt.legend(handles=[btc_plot[0],
                    ada_plot[0],
                    lsk_plot[0],
                    eos_plot[0],
                    zrx_plot[0],
                    sky_plot[0]])
plt.suptitle("Število dodatkov po tednih")
plt.ylabel("Dodatki", fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()

for currency_abbr in currency_abbr_list:
    sns.distplot(df['additions_'+currency_abbr], kde=False)
    plt.show()

#DELETIONS
btc_plot = plt.plot(df.index, df.deletions_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.deletions_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.deletions_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.deletions_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.deletions_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.deletions_SKY, label="SKY")

plt.legend(handles=[btc_plot[0],
                    ada_plot[0],
                    lsk_plot[0],
                    eos_plot[0],
                    zrx_plot[0],
                    sky_plot[0]])
plt.suptitle("Število izbrisov po tednih")
plt.ylabel("Deletions", fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()

for currency_abbr in currency_abbr_list:
    sns.distplot(df['deletions_'+currency_abbr], kde=False)
    plt.show()


#fig, ax = plt.subplots()

#CONTRIBUTORS
#Pregled števila contributor-jev skozi čas
btc_plot = plt.plot(df.index, df.contributors_BTC, label="BTC")
ada_plot = plt.plot(df.index, df.contributors_ADA, label="ADA")
lsk_plot = plt.plot(df.index, df.contributors_LSK, label="LSK")
eos_plot = plt.plot(df.index, df.contributors_EOS, label="EOS")
zrx_plot = plt.plot(df.index, df.contributors_ZRX, label="ZRX")
sky_plot = plt.plot(df.index, df.contributors_SKY, label="SKY")

plt.legend(handles=[btc_plot[0],
                    ada_plot[0],
                    lsk_plot[0],
                    eos_plot[0],
                    zrx_plot[0],
                    sky_plot[0]])
plt.suptitle("Število sodelujočih po tednih")
plt.ylabel('Število sodelujočih', fontsize=18)
plt.xlabel("Tedni", fontsize=16)
plt.show()

for currency_abbr in currency_abbr_list:
    sns.distplot(df['contributors_'+currency_abbr], kde=False)
    plt.show()
'''
#REGRESIJA
#EOS
vhodi_eos = ['contributors_EOS', 'commits_EOS', 'additions_EOS',
             'deletions_EOS']
izhodi_eos = 'BTC_price_EOS'
df_eos_change = df_eos.pct_change()
df_eos_change.dropna(axis=0, inplace=True)
predict_regression(df_eos_change[vhodi_eos], df_eos_change[izhodi_eos], 'Valuta EOS')

#ADA
vhodi_ada = ['contributors_ADA', 'commits_ADA', 'additions_ADA',
             'deletions_ADA']
izhodi_ada = 'BTC_price_ADA'
df_ada_change = df_ada.pct_change()
df_ada_change = df_ada_change.replace([np.inf, -np.inf], np.nan)
df_ada_change.dropna(axis=0, inplace=True)
print(df_ada_change)
predict_regression(df_ada_change[vhodi_ada], df_ada_change[izhodi_ada], 'Valuta ADA')

#LSK
vhodi_lsk = ['contributors_LSK', 'commits_LSK', 'additions_LSK',
             'deletions_LSK']
izhodi_lsk = 'BTC_price_LSK'
df_lisk_change = df_lisk.pct_change()
df_lisk_change.dropna(axis=0, inplace=True)
predict_regression(df_lisk_change[vhodi_lsk], df_lisk_change[izhodi_lsk], 'Valuta LISK')

#SKY
vhodi_sky = ['contributors_SKY', 'commits_SKY', 'additions_SKY',
             'deletions_SKY']
izhodi_sky = 'BTC_price_SKY'
df_sky_change = df_sky.pct_change()
df_sky_change = df_sky_change.replace([np.inf, -np.inf], np.nan)
df_sky_change.dropna(axis=0, inplace=True)
predict_regression(df_sky_change[vhodi_sky], df_sky_change[izhodi_sky], 'Valuta SKY')

#ZRX
vhodi_zrx = ['contributors_ZRX', 'commits_ZRX', 'additions_ZRX',
             'deletions_ZRX']
izhodi_zrx = 'BTC_price_ZRX'
df_zrx_change = df_zrx.pct_change()
df_zrx_change = df_zrx_change.replace([np.inf, -np.inf], np.nan)
df_zrx_change.dropna(axis=0, inplace=True)
predict_regression(df_zrx_change[vhodi_zrx], df_zrx_change[izhodi_zrx], 'Valuta ZRX')

#Napoved cene na podlagi števila commitov, števila contributorje in števila vrstic kode

#Pregled gibanje cene kriptovalute skozi čas




