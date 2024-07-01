#%%
##################################################
##################################################
##                                              ##
##             0) Import packages               ##
##                                              ##
##################################################
##################################################
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, ccf, InfeasibleTestError
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from datetime import datetime, timedelta
from networkx.algorithms.community import girvan_newman
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
import datetime as dt
import openpyxl
import requests
import seaborn as sns
import statsmodels.api as sm
import os
import networkx as nx
import pyperclip

##################################################
##################################################
##                                              ##
##             0) Define functions              ##
##                                              ##
##################################################
##################################################
# Download stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    return stock_data[["Close"]]

# Creating the dataframe for the returns
def prices_to_returns(df):
    return df.pct_change().fillna(0)

# Dickey-Fuller test
def analyze_stationarity(df):
    results = []
    for column in df.columns:
        series = df[column].dropna()
        series = series.reset_index(drop = True)
        # Initial Dickey-Fuller test
        dftest = adfuller(series, maxlag = 10, autolag = 'AIC')
        if dftest[1] > 0.05:
            print(f"{column} is not stationary. Differencing the series.")
            series_diff = df[column].diff().dropna().reset_index(drop = True)
            dftest = adfuller(series_diff, maxlag = 5, autolag = 'AIC')
        dfoutput = {
            'Column': column,
            'Test Statistic': dftest[0],
            'p-value': dftest[1],
            '#Lags Used': dftest[2],
            'Number of Observations Used': dftest[3],
            'Critical Value (1%)': dftest[4]['1%'],
            'Critical Value (5%)': dftest[4]['5%'],
            'Critical Value (10%)': dftest[4]['10%']
            }
        results.append(dfoutput)
    return pd.DataFrame(results)   

# Calculate Volatility
def calc_volatility(df):
    volatility = {column: df[column].std() for column in df.columns}
    return volatility

def table_corr(df1, n1, df2, n2):
    result = []
    for i in n1:
        for j in n2:
            temp = pd.concat([df1.iloc[:200, i], df2.iloc[:200, j]], axis = 1)
            corr_value = temp.corr().iloc[0, 1]
            result.append(((df1.columns[i], df2.columns[j]), corr_value))
    return result

# Garch estimation
def fit_garch_model(returns, model_type = 'GARCH', p = 1, q = 1, dist = 'Normal'):
    try:
        model = None
        if model_type == 'GARCH':
            model = arch_model(returns, vol = 'Garch', p = p, q = q, dist = dist)
        elif model_type == 'EGARCH':
            model = arch_model(returns, vol = 'EGarch', p = p, q = q, dist = dist)
        elif model_type == 'GJR-GARCH':
            model = arch_model(returns, vol = 'GARCH', p = p, o = 1, q = q, dist = dist)
        else:
            raise ValueError("Unsupported model type")
        return model.fit(disp='off')
    except Exeption as e:
        print(f"Error fitting GARCH model {e}")
        return None

##################################################
##################################################
##                                              ##
##          0) Define static variables          ##
##                                              ##
##################################################
##################################################
start_date = '2019-01-01'
end_date = '2024-04-30'

ticker_equity = ["^GSPC", "^N225", "^RUT", "FTSEMIB.MI", "^IXIC", "^DJI", "^FCHI", "^GDAXI", "^FTSE"]
names_equity = ["S&P 500", "Nikkei 225", "Russell 2000", "FTSE MIB", "NASDAQ", "Dow Jones", "CAC 40", "DAX", "FTSE 100"]

ticker_crypto = ["BTC", "ETH", "USDT", "XRP", "DOGE", "ADA"]
names_crypto = ["Bitcoin", "Ethereum", "USDT", "Ripple", "Dogecoin", "Cardano"]

##################################################
##################################################
##                                              ##
##             1) Importing Data                ##
##                                              ##
##################################################
##################################################
directory = r'C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Script'
os.makedirs(directory, exist_ok = True)
##### Cryptocurrencies -- Hourly Data #####
df1 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "BTC")
df2 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "ETH")
df3 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "USDT")
df4 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "XRP")
df5 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "DOGE")
df6 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Crypto_data.xlsx", sheet_name = "ADA")
# Downloading equity indices
df_i = download_stock_data(ticker_equity, start_date, end_date)
# Rename the downloaded data
df_i.columns = names_equity
df_i1 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "BTC")
df_i2 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "ETH")
df_i3 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "USDT")
df_i4 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "XRP")
df_i5 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "DOGE")
df_i6 = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\Daily\Daily.xlsx", sheet_name = "ADA")

##################################################
##################################################
##                                              ##
##             1.1) Preprocessing Data          ##
##                                              ##
##################################################
##################################################
#Daily
df_i1 = df_i1[["Date", "Price"]]
df_i1 = df_i1.rename(columns = {"Price" : "BTC"})
df_i2 = df_i2[["Price"]]
df_i2 = df_i2.rename(columns = {"Price" : "ETH"})
df_i3 = df_i3[["Price"]]
df_i3 = df_i3.rename(columns = {"Price" : "USDT"})
df_i4 = df_i4[["Price"]]
df_i4 = df_i4.rename(columns = {"Price" : "XRP"})
df_i5 = df_i5[["Price"]]
df_i5 = df_i5.rename(columns = {"Price" : "DOGE"})
df_i6 = df_i6[["Price"]]
df_i6 = df_i6.rename(columns = {"Price" : "ADA"})
df_ic = pd.concat([df_i1, df_i2, df_i3, df_i4, df_i5, df_i6], axis = 1)
df_ic["BTC"] = pd.to_numeric(df_ic["BTC"], errors = 'coerce')
df_ic["ETH"] = pd.to_numeric(df_ic["ETH"], errors = 'coerce')
df_ic["USDT"] = pd.to_numeric(df_ic["USDT"], errors = 'coerce')
df_ic["XRP"] = pd.to_numeric(df_ic["XRP"], errors = 'coerce')
df_ic["DOGE"] = pd.to_numeric(df_ic["DOGE"], errors = 'coerce')
df_ic["ADA"] = pd.to_numeric(df_ic["ADA"], errors = 'coerce')
df_ic["Date"] = pd.to_datetime(df_ic["Date"])
#### Cryptocurrencies #####
df1 = df1[["Date", "Price"]]
df1 = df1.rename(columns = {"Price" : "BTC"})
df2 = df2[["Price"]]
df2 = df2.rename(columns = {"Price" : "ETH"})
df3 = df3[["Price"]]
df3 = df3.rename(columns = {"Price" : "USDT"})
df4 = df4[["Price"]]
df4 = df4.rename(columns = {"Price" : "XRP"})
df5 = df5[["Price"]]
df5 = df5.rename(columns = {"Price" : "DOGE"})
df6 = df6[["Price"]]
df6 = df6.rename(columns = {"Price" : "ADA"})
# Converting to numeric
df_c = pd.concat([df1, df2, df3, df4, df5, df6], axis = 1)
df_c["BTC"] = pd.to_numeric(df_c["BTC"], errors = 'coerce')
df_c["ETH"] = pd.to_numeric(df_c["ETH"], errors = 'coerce')
df_c["USDT"] = pd.to_numeric(df_c["USDT"], errors = 'coerce')
df_c["XRP"] = pd.to_numeric(df_c["XRP"], errors = 'coerce')
df_c["DOGE"] = pd.to_numeric(df_c["DOGE"], errors = 'coerce')
df_c["ADA"] = pd.to_numeric(df_c["ADA"], errors = 'coerce')
df_c["Date"] = pd.to_datetime(df_c["Date"])
#### Equity ####
return_i = prices_to_returns(df_i)
return_i = return_i.sort_index(ascending = True)
# Crypto
return_c = df_c
return_c.index = return_c["Date"]
return_c = return_c.iloc[:,1:]
return_c = return_c.sort_index(ascending = True)
return_c = return_c.iloc[:-4]

a = return_i.index.astype(str).str[:10]
b = return_c.index.astype(str).str[:10]
return_c2 = pd.DataFrame(columns = return_c.columns)
frames = []
j = 0
i = 0

for i in range(0, len(return_c) - 1, 4):
    if b[i] == a[j]:
        frames.append(return_c.iloc[i : (i + 4)])
        j += 1
return_c2 = pd.concat(frames)
return_df = pd.DataFrame(index = return_c2.index, columns = return_c2.columns)
for i in range(0, len(return_c2)):
    for col in return_c2.columns:
        previous_value = return_c2.iloc[i-4][col]
        current_value = return_c2.iloc[i][col]
        return_df.iloc[i][col] = (current_value - previous_value) / previous_value
return_c = return_df[4:]
return_i = return_i[1:]
### Deleting unnecessary variables ###
del df1, df2, df3, df4, df5, df6, a, b, col, return_df, return_c2, i, j, current_value, previous_value

##################################################
##################################################
##                                              ##
##          1.1) Creating all datasets          ##
##                                              ##
##################################################
##################################################
#Splitted sample
return_i = 100 * return_i
return_c = 100 * return_c
return_c_19_20 = return_c.iloc[:2080,:]
return_c_21_24 = return_c.iloc[2080:,:]
return_i_19_20 = return_i.iloc[:520,:]
return_i_21_24 = return_i.iloc[520:,:]
#Aggregated sample
#05:00:00
Crypto_19_20_5h = return_c_19_20.iloc[0::4]
Crypto_21_24_5h = return_c_21_24.iloc[0::4]
Crypto_19_20_5h.index = return_i_19_20.index
Crypto_21_24_5h.index = return_i_21_24.index
#15:00:00
Crypto_19_20_15h = return_c_19_20.iloc[1::4]
Crypto_21_24_15h = return_c_21_24.iloc[1::4]
Crypto_19_20_15h.index = return_i_19_20.index
Crypto_21_24_15h.index = return_i_21_24.index
#17:00:00
Crypto_19_20_17h = return_c_19_20.iloc[2::4]
Crypto_21_24_17h = return_c_21_24.iloc[2::4]
Crypto_19_20_17h.index = return_i_19_20.index
Crypto_21_24_17h.index = return_i_21_24.index
#20:00:00 (final)
Crypto_19_20_20h = return_c_19_20.iloc[3::4]
Crypto_21_24_20h = return_c_21_24.iloc[3::4]
Crypto_19_20_20h.index = return_i_19_20.index
Crypto_21_24_20h.index = return_i_21_24.index
# Crypto final
Crypto_final = pd.concat([Crypto_19_20_20h, Crypto_21_24_20h])
Crypto_final_5h = pd.concat([Crypto_19_20_5h, Crypto_21_24_5h])
Crypto_final_15h = pd.concat([Crypto_19_20_15h, Crypto_21_24_15h])
Crypto_final_17h = pd.concat([Crypto_19_20_17h, Crypto_21_24_17h])
Crypto_final_20h = pd.concat([Crypto_19_20_20h, Crypto_21_24_20h])
Index_final = pd.concat([return_i_19_20, return_i_21_24])
df_final = pd.concat([Crypto_final, Index_final], axis = 1)
# Nikkei vs Crypto (5:00)
temp_JP = pd.concat([Crypto_19_20_5h, return_i_19_20], axis = 1)
temp_JPc = pd.concat([Crypto_21_24_5h, return_i_21_24], axis = 1)
#DAX and FTSE 100 vs Crypto
temp_UK = pd.concat([Crypto_19_20_15h, return_i_19_20], axis = 1)
temp_UKc = pd.concat([Crypto_21_24_15h, return_i_21_24], axis = 1)
# FTSE MIB and CAC 40 vs Crypto
temp_IT = pd.concat([Crypto_19_20_17h, return_i_19_20], axis = 1)
temp_ITc = pd.concat([Crypto_21_24_17h, return_i_21_24], axis = 1)
# US vs Crypto
temp_US = pd.concat([Crypto_19_20_20h, return_i_19_20], axis = 1)
temp_USc = pd.concat([Crypto_21_24_20h, return_i_21_24], axis = 1)
# Full
temp_final = pd.concat([Crypto_final, Index_final], axis = 1)
temp_final_5h = pd.concat([Crypto_final_5h, Index_final], axis = 1)
temp_final_15h = pd.concat([Crypto_final_15h, Index_final], axis = 1)
temp_final_17h = pd.concat([Crypto_final_17h, Index_final], axis = 1)
temp_final_20h = pd.concat([Crypto_final_20h, Index_final], axis = 1)

##################################################
##################################################
##                                              ##
##          2) Descriptive Statistics           ##
##                                              ##
##################################################
##################################################
def calculate_descriptive_statistics(df):
    desc_stats = df.describe().T
    desc_stats['skewness'] = df.skew()
    desc_stats['kurtosis'] = df.kurtosis()
    desc_stats = desc_stats.drop(columns=['count', '25%', '75%'])
    return round(desc_stats, 2)

descriptive_stats_c = round(calculate_descriptive_statistics(Crypto_final_20h.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_c.tex', 'w') as f:
    f.write(descriptive_stats_c.to_latex())
descriptive_stats_i = round(calculate_descriptive_statistics(return_i.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_i.tex', 'w') as f:
    f.write(descriptive_stats_i.to_latex())
descriptive_stats_c_19_20 = round(calculate_descriptive_statistics(Crypto_19_20_20h.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_c_19_20.tex', 'w') as f:
    f.write(descriptive_stats_c_19_20.to_latex())
descriptive_stats_c_21_24 = round(calculate_descriptive_statistics(Crypto_21_24_20h.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_c_21_24.tex', 'w') as f:
    f.write(descriptive_stats_c_21_24.to_latex())
descriptive_stats_i_19_20 = round(calculate_descriptive_statistics(return_i_19_20.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_i_19_20.tex', 'w') as f:
    f.write(descriptive_stats_i_19_20.to_latex())
descriptive_stats_i_21_24 = round(calculate_descriptive_statistics(return_i_21_24.apply(pd.to_numeric, errors = 'coerce')), 2)
with open('descriptive_stats_i_21_24.tex', 'w') as f:
    f.write(descriptive_stats_i_21_24.to_latex())

##################################################
##################################################
##                                              ##
##             3) Data Visalization             ##
##                                              ##
##################################################
##################################################
# Bitcoin vs S&P 500 plot
plt.figure(figsize = (14, 7))
plt.plot(return_c.index, return_c['BTC'], label = 'Bitcoin', color = 'red')
plt.plot(return_i.index, return_i['S&P 500'], label = 'S&P 500', color = 'blue')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('Bitcoin vs S&P 500 - Returns Evolution')
plt.legend(loc = 'best')
plt.savefig('Bitcoin vs S&P 500 Return.png')
plt.show()
# Bitcoin vs S&P 500 plot - cumulative return
plt.figure(figsize = (14, 7))
plt.plot(return_c.index, return_c['BTC'].cumsum(), label = 'Bitcoin', color = 'red')
plt.plot(return_i.index, return_i['S&P 500'].cumsum(), label = 'S&P 500', color = 'blue')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Bitcoin vs S&P 500 - Cumulative Returns Evolution')
plt.legend(loc = 'best')
plt.savefig('Bitcoin vs S&P 500 Cumulative Return.png')
plt.show()
# Prices plot
plt.figure(figsize = (14, 7))
plt.plot(df_c.index, df_c['BTC'], label = 'Bitcoin', color = 'red')
plt.plot(df_i.index, df_i['S&P 500'], label = 'S&P 500', color = 'blue')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.title('Bitcoin vs S&P 500 - Price Evolution')
plt.legend(loc = 'best')
plt.savefig('Bitcoin vs S&P 500 - Price.png')
plt.show()

##################################################
##################################################
##                                              ##
##             3.0.1) Plotting data             ##
##                                              ##
##################################################
##################################################
# Plotting crypto prices
plt.figure(figsize = (15, 10))
for column in return_c.columns:
    if column != 'Date':
        plt.plot(df_c.index, df_c[column], label = column)
# Equity pices
for column in return_i.columns:
    if column != 'Date':
        plt.plot(df_i.index, df_i[column], label = column)
plt.title('Prices Evolution')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc = 'best')
plt.show()

# Cumulative return - Crypto
plt.figure(figsize = (15, 10))
for column in return_c.columns:
    if column != 'Date':
        plt.plot(return_c.index, return_c[column].cumsum(), label = column)
# Equity
for column in return_i.columns:
    if column != 'Date':
        plt.plot(return_i.index, return_i[column].cumsum(), label = column)
plt.title('Cumulative Returns Evolution')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc = 'best')
plt.show()

##################################################
##################################################
##                                              ##
##     3.0.2) Plotting crypto capitalization    ##
##                                              ##
##################################################
##################################################
##### Importing & pre-processing #####
# Crypto
capt_crypto = pd.read_csv(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\CoinGecko-GlobalCryptoMktCap-2024-06-17.csv")
capt_crypto = capt_crypto.rename(columns = {"market_cap": "MarketCap"})
capt_crypto["MarketCap"] = capt_crypto["MarketCap"].str.replace(',', '').astype(float)
capt_crypto["Date"] = pd.to_datetime(capt_crypto["Date"], format = '%d/%m/%Y')
capt_crypto.set_index("Date", inplace = True)
capt_standpoor = pd.read_excel(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\Data\MktcapStaAndPoor.xlsx")
capt_standpoor = capt_standpoor.rename(columns = {"MarketCap": "MarketCap_S&P 500"})
#capt_standpoor["MarketCap_S&P 500"] = capt_standpoor["MarketCap_S&P 500"].str.replace(',', '').astype(float)
capt_standpoor["Date"] = pd.to_datetime(capt_standpoor["Date"], format = '%d/%m/%Y')
capt_standpoor.set_index("Date", inplace = True)
# Plotting
plt.figure(figsize = (14, 7))
plt.plot(capt_crypto.index, capt_crypto["MarketCap"], label = "Crypto Market")
plt.plot(capt_standpoor.index, capt_standpoor["MarketCap_S&P 500"], label = "S&P 500")
plt.title('Market capitalization in USD')
plt.xlabel('Date')
plt.ylabel('Market Cap in USD')
plt.legend(loc = 'best')
plt.show()

##################################################
##################################################
##                                              ##
##                3.1) Volatility               ##
##                                              ##
##################################################
##################################################
#Crypto
vol_c = calc_volatility(Crypto_final_20h)
#Equity
vol_i = calc_volatility(return_i)
#Splitted Sample
vol_c_19_20 = calc_volatility(return_c_19_20)
vol_c_21_24 = calc_volatility(return_c_21_24)
vol_i_19_20 = calc_volatility(return_i_19_20)
vol_i_21_24 = calc_volatility(return_i_21_24)

##################################################
##################################################
##                                              ##
##             3.2.1) Correlation               ##
##                                              ##
##################################################
##################################################
df_cov_19_20 = pd.DataFrame(0, columns = names_equity, index = ticker_crypto)
df_cov_21_24 = pd.DataFrame(0, columns = names_equity, index = ticker_crypto)

# Nikkei vs Crypto (5:00)
temp2_JP_cov = temp_JP.cov()
df_cov_19_20.iloc[:, 1] = temp2_JP_cov.iloc[8, :6]

temp2_JPc_cov = temp_JPc.cov()
df_cov_21_24.iloc[:, 1] = temp2_JPc_cov.iloc[8, :6]

# DAX and FTSE 100 vs Crypto
temp2_UK_cov = temp_UK.cov()
df_cov_19_20.iloc[:, 7] = temp2_UK_cov.iloc[13, :6]
df_cov_19_20.iloc[:, 8] = temp2_UK_cov.iloc[14, :6]

temp2_UKc_cov = temp_UKc.cov()
df_cov_21_24.iloc[:, 7] = temp2_UKc_cov.iloc[13, :6]
df_cov_21_24.iloc[:, 8] = temp2_UKc_cov.iloc[14, :6]

# FTSE MIB and CAC 40 vs Crypto
temp2_IT_cov = temp_IT.cov()
df_cov_19_20.iloc[:, 6] = temp2_IT_cov.iloc[12, :6]
df_cov_19_20.iloc[:, 3] = temp2_IT_cov.iloc[9, :6]

temp2_ITc_cov = temp_ITc.cov()
df_cov_21_24.iloc[:, 6] = temp2_ITc_cov.iloc[12, :6]
df_cov_21_24.iloc[:, 3] = temp2_ITc_cov.iloc[9, :6]

# US vs Crypto
temp2_US_cov = temp_US.cov()
df_cov_19_20.iloc[:, 0] = temp2_US_cov.iloc[6, :6]
df_cov_19_20.iloc[:, 2] = temp2_US_cov.iloc[8, :6]
df_cov_19_20.iloc[:, 4] = temp2_US_cov.iloc[10, :6]
df_cov_19_20.iloc[:, 5] = temp2_US_cov.iloc[11, :6]

temp2_USc_cov = temp_USc.cov()
df_cov_21_24.iloc[:, 0] = temp2_USc_cov.iloc[6, :6]
df_cov_21_24.iloc[:, 2] = temp2_USc_cov.iloc[8, :6]
df_cov_21_24.iloc[:, 4] = temp2_USc_cov.iloc[10, :6]
df_cov_21_24.iloc[:, 5] = temp2_USc_cov.iloc[11, :6]

# Covariance 2019-2024
temp2_final_cov = temp_final.cov()
df_cov_full = round(temp2_final_cov, 2)
df_cov_21_24 = round(df_cov_21_24, 2)
df_cov_19_20 = round(df_cov_19_20, 2)

directory = r'C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\3. Deliverables'
os.makedirs(directory, exist_ok = True)
file_path_19_20_cov = os.path.join(directory, 'covariance_matrix_19_20.tex')
file_path_21_24_cov = os.path.join(directory, 'covariance_matrix_21_24.tex')
file_path_full_cov = os.path.join(directory, 'covariance_matrix_full.tex')
with open(file_path_19_20_cov, 'w') as f:
    f.write(df_cov_19_20.to_latex())
with open(file_path_21_24_cov, 'w') as f:
    f.write(df_cov_21_24.to_latex())
with open(file_path_full_cov, 'w') as f:
    f.write(df_cov_full.to_latex())
#%%
##################################################
##################################################
##                                              ##
##             3.2.1) Correlation               ##
##                                              ##
##################################################
##################################################
df_corr_19_20 = pd.DataFrame(0, columns = names_equity, index = ticker_crypto)
df_corr_21_24 = pd.DataFrame(0, columns = names_equity, index = ticker_crypto)

# Nikkei vs Crypto (5:00)
temp2_JP = temp_JP.corr()
df_corr_19_20.iloc[:, 1] = temp2_JP.iloc[8, :6]

temp2_JPc = temp_JPc.corr()
df_corr_21_24.iloc[:, 1] = temp2_JPc.iloc[8, :6]

#DAX and FTSE 100 vs Crypto
temp2_UK = temp_UK.corr()
df_corr_19_20.iloc[:, 7] = temp2_UK.iloc[13, :6]
df_corr_19_20.iloc[:, 8] = temp2_UK.iloc[14, :6]

temp2_UKc = temp_UKc.corr()
df_corr_21_24.iloc[:, 7] = temp2_UKc.iloc[13, :6]
df_corr_21_24.iloc[:, 8] = temp2_UKc.iloc[14, :6]

# FTSE MIB and CAC 40 vs Crypto
temp2_IT = temp_IT.corr()
df_corr_19_20.iloc[:, 6] = temp2_IT.iloc[12, :6]
df_corr_19_20.iloc[:, 3] = temp2_IT.iloc[9, :6]

temp2_ITc = temp_ITc.corr()
df_corr_21_24.iloc[:, 6] = temp2_ITc.iloc[12, :6]
df_corr_21_24.iloc[:, 3] = temp2_ITc.iloc[9, :6]

# US vs Crypto
temp2_US = temp_US.corr()
df_corr_19_20.iloc[:, 0] = temp2_US.iloc[6, :6]
df_corr_19_20.iloc[:, 2] = temp2_US.iloc[8, :6]
df_corr_19_20.iloc[:, 4] = temp2_US.iloc[10, :6]
df_corr_19_20.iloc[:, 5] = temp2_US.iloc[11, :6]

temp2_USc = temp_USc.corr()
df_corr_21_24.iloc[:, 0] = temp2_USc.iloc[6, :6]
df_corr_21_24.iloc[:, 2] = temp2_USc.iloc[8, :6]
df_corr_21_24.iloc[:, 4] = temp2_USc.iloc[10, :6]
df_corr_21_24.iloc[:, 5] = temp2_USc.iloc[11, :6]

# Corr 2019-2014
temp2_final = temp_final.corr()
df_corr_full = round(temp2_final, 2)
df_corr_21_24 = round(df_corr_21_24, 2)
df_corr_19_20 = round(df_corr_19_20, 2)

directory = r'C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\3. Deliverables'
os.makedirs(directory, exist_ok = True)
file_path_19_20 = os.path.join(directory, 'correlation_matrix_19_20.tex')
file_path_21_24 = os.path.join(directory, 'correlation_matrix_21_24.tex')
file_path_full = os.path.join(directory, 'correlation_matrix_full.tex')
with open(file_path_19_20, 'w') as f:
    f.write(df_corr_19_20.to_latex())
with open(file_path_21_24, 'w') as f:
    f.write(df_corr_21_24.to_latex())
with open(file_path_full, 'w') as f:
    f.write(df_corr_full.to_latex())

##################################################
##################################################
##                                              ##
##           3.2.1) Lagged Correlation          ##
##                                              ##
##################################################
##################################################
lagged_corr_JP = {}
lagged_corr_UK = {}
lagged_corr_IT = {}
lagged_corr_US = {}
def lagged_correlation(series1, series2, max_lag = 5):
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = series1.corr(series2)
        else:
            corr = series1[lag:].reset_index(drop = True).corr(series2[:-lag].reset_index(drop = True))
        correlations.append(corr)
    return correlations
max_lag = 10
for column1 in temp_JP.columns[[7]]:
    for column2 in temp_JP.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_JP[(column1, column2)] = lagged_correlation(temp_JP[column1], temp_JP[column2], max_lag)

for column1 in temp_UK.columns[list(range(13, 15))]:
    for column2 in temp_UK.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_UK[(column1, column2)] = lagged_correlation(temp_UK[column1], temp_UK[column2], max_lag)

for column1 in temp_IT.columns[[9] + [12]]:
    for column2 in temp_IT.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_IT[(column1, column2)] = lagged_correlation(temp_IT[column1], temp_IT[column2], max_lag)

for column1 in temp_US.columns[[6] + [8] + list(range(10, 12))]:
    for column2 in temp_US.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_US[(column1, column2)] = lagged_correlation(temp_US[column1], temp_US[column2], max_lag)

def plot_lagged_correlations2(lagged_corr):
    indices = set(key[0] for key in lagged_corr.keys())
    for index in indices:
        plt.figure(figsize = (10, 5))
        for key, values in lagged_corr.items():
            if key[0] == index:
                plt.plot(range(max_lag + 1), values, marker = 'o', label = f'{key[1]}')
        plt.title(f'Lagged Correlation for {index}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.legend(loc = 'best')
        plt.grid(True)
        plt.savefig(f'Lagged Correlation for {index}.png')
        plt.show()
# Plot the lagged correlations for each country
plot_lagged_correlations2(lagged_corr_JP)
plot_lagged_correlations2(lagged_corr_UK)
plot_lagged_correlations2(lagged_corr_IT)
plot_lagged_correlations2(lagged_corr_US)

def plot_lagged_correlations_combined(lagged_corr_dicts, max_lag):
    fig, axs = plt.subplots(4, 2, figsize = (20, 20))
    axs = axs.flatten()
    indices = [(key[0], i) for i, lagged_corr in enumerate(lagged_corr_dicts) for key in lagged_corr.keys()]
    unique_indices = list(set(indices))
    
    for ax, (index, dict_idx) in zip(axs, unique_indices):
        lagged_corr = lagged_corr_dicts[dict_idx]
        for key, values in lagged_corr.items():
            if key[0] == index:
                ax.plot(range(max_lag + 1), values, marker = 'o', label = f'{key[1]}')
        ax.set_title(f'Lagged Correlation for {index}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.legend(loc = 'best')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('Lagged_Correlation_Combined.png')
    plt.show()

# Combine the lagged correlation dictionaries and plot
lagged_corr_dicts = [lagged_corr_JP, lagged_corr_UK, lagged_corr_IT, lagged_corr_US]
plot_lagged_correlations_combined(lagged_corr_dicts, max_lag)

# Period 2021 - 2024
lagged_corr_JPc = {}
lagged_corr_UKc = {}
lagged_corr_ITc = {}
lagged_corr_USc = {}

def lagged_correlation(series1, series2, max_lag = 5):
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = series1.corr(series2)
        else:
            corr = series1[lag:].reset_index(drop = True).corr(series2[:-lag].reset_index(drop = True))
        correlations.append(corr)
    return correlations

for column1 in temp_JPc.columns[[7]]:
    for column2 in temp_JPc.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_JPc[(column1, column2)] = lagged_correlation(temp_JPc[column1], temp_JPc[column2], max_lag)

for column1 in temp_UKc.columns[list(range(13, 15))]:
    for column2 in temp_UKc.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_UKc[(column1, column2)] = lagged_correlation(temp_UKc[column1], temp_UKc[column2], max_lag)

for column1 in temp_ITc.columns[[9] + [12]]:
    for column2 in temp_ITc.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_ITc[(column1, column2)] = lagged_correlation(temp_ITc[column1], temp_ITc[column2], max_lag)

for column1 in temp_USc.columns[[6] + [8] + list(range(10, 12))]:
    for column2 in temp_USc.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_USc[(column1, column2)] = lagged_correlation(temp_USc[column1], temp_USc[column2], max_lag)

# Plot
plot_lagged_correlations2(lagged_corr_JPc)
plot_lagged_correlations2(lagged_corr_UKc)
plot_lagged_correlations2(lagged_corr_ITc)
plot_lagged_correlations2(lagged_corr_USc)
lagged_corr_dicts_c = [lagged_corr_JPc, lagged_corr_UKc, lagged_corr_ITc, lagged_corr_USc]
plot_lagged_correlations_combined(lagged_corr_dicts_c, max_lag)

# Full Period 2019 - 2024
lagged_corr_JPf = {}
lagged_corr_UKf = {}
lagged_corr_ITf = {}
lagged_corr_USf = {}

def lagged_correlation(series1, series2, max_lag = 5):
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = series1.corr(series2)
        else:
            corr = series1[lag:].reset_index(drop = True).corr(series2[:-lag].reset_index(drop = True))
        correlations.append(corr)
    return correlations

for column1 in temp_final_5h.columns[[7]]:
    for column2 in temp_final_5h.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_JPf[(column1, column2)] = lagged_correlation(temp_final_5h[column1], temp_final_5h[column2], max_lag)

for column1 in temp_final_15h.columns[list(range(13, 15))]:
    for column2 in temp_final_15h.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_UKf[(column1, column2)] = lagged_correlation(temp_final_15h[column1], temp_final_15h[column2], max_lag)

for column1 in temp_final_17h.columns[[9] + [12]]:
    for column2 in temp_final_17h.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_ITf[(column1, column2)] = lagged_correlation(temp_final_17h[column1], temp_final_17h[column2], max_lag)

for column1 in temp_final_20h.columns[[6] + [8] + list(range(10, 12))]:
    for column2 in temp_final_20h.columns[list(range(0, 6))]:
        if column1 != column2:
            lagged_corr_USf[(column1, column2)] = lagged_correlation(temp_final_20h[column1], temp_final_20h[column2], max_lag)

# Plot
plot_lagged_correlations2(lagged_corr_JPf)
plot_lagged_correlations2(lagged_corr_UKf)
plot_lagged_correlations2(lagged_corr_ITf)
plot_lagged_correlations2(lagged_corr_USf)
lagged_corr_dicts_f = [lagged_corr_JPf, lagged_corr_UKf, lagged_corr_ITf, lagged_corr_USf]
plot_lagged_correlations_combined(lagged_corr_dicts_f, max_lag)
#%%
##################################################
##################################################
##                                              ##
##             3.3) Stationary Check            ##
##                                              ##
##################################################
##################################################
### Cryptocurrencies ###
crypto_results = analyze_stationarity(Crypto_final) # DF test
### Equity indices ###
equity_results = analyze_stationarity(Index_final)
combined_results = pd.concat([crypto_results, equity_results], keys = ['Crypto', 'Equity'], names = ['Category', 'Index']).reset_index(level = 'Category')
latex_table = combined_results.to_latex(index = False)
latex_table_path = os.path.join(directory, 'latex_table.tex')
with open(latex_table_path, 'w') as f:
    f.write(latex_table)

# Check the Autocorrelation and the Partial Autocorrelation
plot_acf(Index_final['S&P 500'], lags = 20)
plt.show()
plot_pacf(Index_final['S&P 500'], lags = 20)
plt.show()
#%%
##################################################
##################################################
##                                              ##
##                4) VAR models                 ##
##                                              ##
##################################################
##################################################
##### New VAR method #####
# Creating the structure
crypto_vars = Crypto_final.columns
index_vars = Index_final.columns

index_var_5h = Index_final.columns[1]
index_var_5h = pd.Index([index_var_5h])
h5 = [1]
index_var_15h = Index_final.columns[[7, 8]]
index_var_15h = pd.Index([index_var_15h])
index_var_15h = pd.Index(index_var_15h[0])
h15 = [7, 8]
index_var_17h = Index_final.columns[[3, 6]]
index_var_17h = pd.Index([index_var_17h])
index_var_17h = pd.Index(index_var_17h[0])
h17 = [3, 6]
index_var_20h = Index_final.columns[[0, 2, 4, 5]]
index_var_20h = pd.Index([index_var_20h])
index_var_20h = pd.Index(index_var_20h[0])
h20 = [0, 2, 4, 5]

results_final = []
results_final_5h = []
results_final_15h = []
results_final_17h = []
results_final_20h = []
ljung_box_results_5h = {}
ljung_box_results_15h = {}
ljung_box_results_17h = {}
ljung_box_results_20h = {}
jb_test_5h = {}
jb_test_15h = {}
jb_test_17h = {}
jb_test_20h = {}
arch_test_5h = {}
arch_test_15h = {}
arch_test_17h = {}
arch_test_20h = {}
fevd_5h = []
fevd_15h = []
fevd_17h = []
fevd_20h = []
p_distribution_5h = []
p_distribution_15h = []
p_distribution_17h = []
p_distribution_20h = []

forecast_steps = 10
#%%
# 5:00:00 JP
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_5h)):
        temp_index_var = h5[index_var]
        df_5h = pd.concat([Crypto_final_5h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_5h = df_5h.apply(pd.to_numeric, errors = 'coerce')
        model_5h = VAR(df_5h)
        lag_order_5h = model_5h.select_order(maxlags = 10)
        optimal_lag_5h = lag_order_5h.aic
        result_5h = model_5h.fit(maxlags = optimal_lag_5h, ic = 'aic')
        res_5h = result_5h.resid
        dw_5h = durbin_watson(res_5h)

        # IRF
        irf_5h = result_5h.irf(forecast_steps)
        irf_5h.plot(orth = True)
        plt.show()            
        # Q-Q Plot
        #plt.subplot(1, 2, 2)
        #sm.qqplot(res_5h[column], line = 's')
        #plt.title(f'Q-Q Plot of Residuals for {column}')
        #plt.tight_layout()
        #plt.show()

        for column in res_5h.columns:
            ljung_box_results_5h[column] = acorr_ljungbox(res_5h[column], lags = optimal_lag_5h, return_df = True)
            # Jarque-Bera test
            jb_test_5h = jarque_bera(res_5h[column])

        # Variance Decomposition
        fevd_5h = result_5h.fevd(10)
        
        results_final_5h.append({
            'crypto_var': crypto_var,
            'index_var': index_var,
            'aic': result_5h.aic,
            'diff_order': optimal_lag_5h,
            'fitted_model': result_5h,
            'Durbin-Watson Statistics': dw_5h,
            'Ljung Box Statistics': ljung_box_results_5h.items(),
            'Jarque Bera p-value': jb_test_5h[1],
            'Variance Decompositions': fevd_5h.decomp
        })
        decomp = fevd_5h.decomp[:, :, :forecast_steps]
        for i, col_name in enumerate(df_5h.columns):
            plt.figure(figsize = (10, 6))
            plt.stackplot(range(1, forecast_steps + 1), decomp[i].T, labels = df_5h.columns)
            plt.title(f'Variance Decomposition of {col_name}')
            plt.xlabel('Steps')
            plt.ylabel('Variance Decomposition')
            plt.legend(loc = 'upper right')
            plt.show()
result_summary_5h = pd.DataFrame(results_final_5h, columns = ['crypto_var', 'index_var', 'aic', 'diff_order', 'Durbin-Watson Statistics', 'Jarque Bera p-value', 'Variance Decompositions'])
result_summary_5h.to_csv('var_model_results_summary_5h.csv', index = False)

# 15:00:00 UK-GER
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_15h)):
        temp_index_var = h15[index_var]
        df_15h = pd.concat([Crypto_final_15h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_15h = df_15h.apply(pd.to_numeric, errors = 'coerce')
        model_15h = VAR(df_15h)
        lag_order_15h = model_15h.select_order(maxlags = 10)
        optimal_lag_15h = lag_order_15h.aic
        result_15h = model_15h.fit(maxlags = optimal_lag_15h, ic = 'aic')
        res_15h = result_15h.resid
        dw_15h = durbin_watson(res_15h)
        # Stability
        if optimal_lag_15h == 0:
            results_final_15h.append({
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_15h.aic,
                'diff_order': optimal_lag_15h,
                'fitted_model': result_15h,
                'Durbin-Watson Statistics': dw_15h,
                'Ljung Box Statistics': np.nan,
                'Jarque Bera p-value': np.nan,
                'Variance Decompositions': np.nan
            })
        else:
            # IRF
            irf_15h = result_15h.irf(30)
            irf_15h.plot(orth = True)
            plt.show()

            for column in res_15h.columns:
                ljung_box_results_15h[column] = acorr_ljungbox(res_15h[column], lags = optimal_lag_15h, return_df = True)
                # Jarque-Bera test
                jb_test_15h = jarque_bera(res_15h[column])
            # Variance Decomposition
            fevd_15h = result_15h.fevd(10)
            results_final_15h.append({
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_15h.aic,
                'diff_order': optimal_lag_15h,
                'fitted_model': result_15h,
                'Durbin-Watson Statistics': dw_15h,
                'Ljung Box Statistics': ljung_box_results_15h.items(),
                'Jarque Bera p-value': jb_test_15h[1],
                'Variance Decompositions': fevd_15h.decomp
            })
            decomp = fevd_15h.decomp[:, :, :forecast_steps]
            for i, col_name in enumerate(df_15h.columns):
                plt.figure(figsize = (10, 6))
                plt.stackplot(range(1, forecast_steps + 1), decomp[i].T, labels = df_15h.columns)
                plt.title(f'Variance Decomposition of {col_name}')
                plt.xlabel('Steps')
                plt.ylabel('Variance Decomposition')
                plt.legend(loc = 'upper right')
                plt.show()
result_summary_15h = pd.DataFrame(results_final_15h, columns = ['crypto_var', 'index_var', 'aic', 'diff_order', 'Durbin-Watson Statistics', 'Jarque Bera p-value', 'Variance Decompositions'])
result_summary_15h.to_csv('var_model_results_summary_15h.csv', index = False)

# 17:00:00 IT-FR
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_17h)):
        temp_index_var = h17[index_var]
        df_17h = pd.concat([Crypto_final_17h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_17h = df_17h.apply(pd.to_numeric, errors = 'coerce')
        model_17h = VAR(df_17h)
        lag_order_17h = model_17h.select_order(maxlags = 10)
        optimal_lag_17h = lag_order_17h.aic
        result_17h = model_17h.fit(maxlags = optimal_lag_17h, ic = 'aic')
        res_17h = result_17h.resid
        dw_17h = durbin_watson(res_17h)
        # Stability
        if optimal_lag_17h == 0:
            results_final_17h.append({                    
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_17h.aic,
                'diff_order': optimal_lag_17h,
                'fitted_model': result_17h,
                'Durbin-Watson Statistics': dw_17h,
                'Ljung Box Statistics': np.nan,
                'Jarque Bera p-value': np.nan,
                'Variance Decompositions': np.nan
                })
        else:
            # IRF
            irf_17h = result_17h.irf(30)
            irf_17h.plot(orth = True)
            plt.show()

            for column in res_17h.columns:
                ljung_box_results_17h[column] = acorr_ljungbox(res_17h[column], lags = optimal_lag_17h, return_df = True)
                # Jarque-Bera test
                jb_test_17h = jarque_bera(res_17h[column])
            # Variance Decomposition
            fevd_17h = result_17h.fevd(10)
            results_final_17h.append({
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_17h.aic,
                'diff_order': optimal_lag_17h,
                'fitted_model': result_17h,
                'Durbin-Watson Statistics': dw_17h,
                'Ljung Box Statistics': ljung_box_results_17h.items(),
                'Jarque Bera p-value': jb_test_17h[1],
                'Variance Decompositions': fevd_17h.decomp
            })
            decomp = fevd_17h.decomp[:, :, :forecast_steps]
            for i, col_name in enumerate(df_17h.columns):
                plt.figure(figsize = (10, 6))
                plt.stackplot(range(1, forecast_steps + 1), decomp[i].T, labels = df_17h.columns)
                plt.title(f'Variance Decomposition of {col_name}')
                plt.xlabel('Steps')
                plt.ylabel('Variance Decomposition')
                plt.legend(loc = 'upper right')
                plt.show()
result_summary_17h = pd.DataFrame(results_final_17h, columns = ['crypto_var', 'index_var', 'aic', 'diff_order', 'Durbin-Watson Statistics', 'Jarque Bera p-value', 'Variance Decompositions'])
result_summary_17h.to_csv('var_model_results_summary_17h.csv', index = False)

# 20:00:00 US
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_20h)):
        temp_index_var = h20[index_var]
        df_20h = pd.concat([Crypto_final_20h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_20h = df_20h.apply(pd.to_numeric, errors = 'coerce')
        model_20h = VAR(df_20h)
        lag_order_20h = model_20h.select_order(maxlags = 10)
        optimal_lag_20h = lag_order_20h.aic
        result_20h = model_20h.fit(maxlags = optimal_lag_20h, ic = 'aic')
        res_20h = result_20h.resid
        dw_20h = durbin_watson(res_20h)
        # Stability
        if optimal_lag_20h == 0:
            # Store the result
            results_final_20h.append({
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_20h.aic,
                'diff_order': optimal_lag_20h,
                'fitted_model': result_20h,
                'Durbin-Watson Statistics': dw_20h,
                'Ljung Box Statistics': np.nan,
                'Jarque Bera p-value': np.nan,
                'Variance Decompositions': np.nan
                })
        else:
            # IRF
            irf_20h = result_20h.irf(30)
            irf_20h.plot(orth = True)
            plt.show()

            for column in res_20h.columns:
                ljung_box_results_20h[column] = acorr_ljungbox(res_20h[column], lags = optimal_lag_20h, return_df = True)
                # Jarque-Bera test
                jb_test_20h = jarque_bera(res_20h[column])
            # Variance Decomposition
            fevd_20h = result_20h.fevd(10)
            # Store the result
            results_final_20h.append({
                'crypto_var': crypto_var,
                'index_var': index_var,
                'aic': result_20h.aic,
                'diff_order': optimal_lag_20h,
                'fitted_model': result_20h,
                'Durbin-Watson Statistics': dw_20h,
                'Ljung Box Statistics': ljung_box_results_20h.items(),
                'Jarque Bera p-value': jb_test_20h[1],
                'Variance Decompositions': fevd_20h.decomp
            })
            decomp = fevd_20h.decomp[:, :, :forecast_steps]
            for i, col_name in enumerate(df_20h.columns):
                plt.figure(figsize = (10, 6))
                plt.stackplot(range(1, forecast_steps + 1), decomp[i].T, labels = df_20h.columns)
                plt.title(f'Variance Decomposition of {col_name}')
                plt.xlabel('Steps')
                plt.ylabel('Variance Decomposition')
                plt.legend(loc = 'upper right')
                plt.show()
result_summary_20h = pd.DataFrame(results_final_20h, columns = ['crypto_var', 'index_var', 'aic', 'diff_order', 'Durbin-Watson Statistics', 'Jarque Bera p-value', 'Variance Decompositions'])
result_summary_20h.to_csv('var_model_results_summary_20h.csv', index = False)
# Deleting unnecessary variables
del column, equity_results, descriptive_stats_c, descriptive_stats_c_19_20, descriptive_stats_c_21_24, descriptive_stats_i, descriptive_stats_i_19_20, descriptive_stats_i_21_24, df_i, df_i1, df_i2, df_i3, df_i4, df_i5, df_i6, df_ic
#%%
##################################################
##################################################
##                                              ##
##              4.0.1) VD in the thesis         ##
##                                              ##
##################################################
##################################################
horizons = {
    '5h': {'index_vars': h5, 'df': Crypto_final_5h, 'results_final': results_final_5h},
    '15h': {'index_vars': h15, 'df': Crypto_final_15h, 'results_final': results_final_15h},
    '17h': {'index_vars': h17, 'df': Crypto_final_17h, 'results_final': results_final_17h},
    '20h': {'index_vars': h20, 'df': Crypto_final_20h, 'results_final': results_final_20h}
}

forecast_steps = 10
output_folder = r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\VD"
os.makedirs(output_folder, exist_ok = True)

# Process each horizon separately
for horizon, data in horizons.items():
    index_vars = data['index_vars']
    crypto_df = data['df']
    results_final = data['results_final']

    for crypto_var in range(len(crypto_vars)):
        for index_var in range(len(index_vars)):
            temp_index_var = index_vars[index_var]
            df = pd.concat([crypto_df.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
            df = df.apply(pd.to_numeric, errors = 'coerce')
            model = VAR(df)
            lag_order = model.select_order(maxlags = 10)
            optimal_lag = lag_order.aic
            result = model.fit(maxlags = optimal_lag, ic = 'aic')
            res = result.resid
            dw = durbin_watson(res)

            # Stability
            if optimal_lag == 0:
                results_final.append({
                    'crypto_var': crypto_var,
                    'index_var': index_var,
                    'aic': result.aic,
                    'diff_order': optimal_lag,
                    'fitted_model': result,
                    'Durbin-Watson Statistics': dw,
                    'Ljung Box Statistics': np.nan,
                    'Jarque Bera p-value': np.nan,
                    'Variance Decompositions': np.nan
                })
            else:
                # IRF
                irf = result.irf(30)
                irf.plot(orth = True)
                plt.title(f'Impulse Response Function')
                plt.close()

                ljung_box_results = {}
                for column in res.columns:
                    ljung_box_results[column] = acorr_ljungbox(res[column], lags = optimal_lag, return_df = True)
                    # Jarque-Bera test
                    jb_test = jarque_bera(res[column])

                # Variance Decomposition
                fevd = result.fevd(forecast_steps)
                results_final.append({
                    'crypto_var': crypto_var,
                    'index_var': index_var,
                    'aic': result.aic,
                    'diff_order': optimal_lag,
                    'fitted_model': result,
                    'Durbin-Watson Statistics': dw,
                    'Ljung Box Statistics': ljung_box_results.items(),
                    'Jarque Bera p-value': jb_test[1],
                    'Variance Decompositions': fevd.decomp
                })

                decomp = fevd.decomp[:, :, :forecast_steps]
                for i, col_name in enumerate(df.columns):
                    plt.figure(figsize = (10, 6))
                    plt.stackplot(range(1, forecast_steps + 1), decomp[i].T, labels = df.columns)
                    plt.title(f'Variance Decomposition of {col_name}')
                    plt.xlabel('Steps')
                    plt.ylabel('Variance Decomposition')
                    plt.legend(loc = 'upper right')
                    plt.savefig(os.path.join(output_folder, f'VD_{horizon}_{col_name}_crypto{crypto_var}_index{index_var}.png'))
                    plt.close()

    # Save the results for this horizon
    result_summary = pd.DataFrame(results_final, columns = ['crypto_var', 'index_var', 'aic', 'diff_order', 'Durbin-Watson Statistics', 'Jarque Bera p-value', 'Variance Decompositions'])
    result_summary.to_csv(f'var_model_results_summary_{horizon}.csv', index = False)

#%%
##################################################
##################################################
##                                              ##
##              4.0.1) p distribution           ##
##                                              ##
##################################################
##################################################
p_distribution_19_20 = []
p_distribution_21_24 = []
p_distribution_19_24 = []
#5:00:00
#First sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_5h)):
        temp_index_var = h5[index_var]
        df_5h = pd.concat([Crypto_19_20_5h.iloc[:, crypto_var], return_i_19_20.iloc[:, temp_index_var]], axis = 1)
        df_5h = df_5h.apply(pd.to_numeric, errors = 'coerce')
        model_5h = VAR(df_5h)
        lag_order_5h = model_5h.select_order(maxlags = 10)
        optimal_lag_5h = lag_order_5h.aic
        p_distribution_19_20.append({'Lag': optimal_lag_5h})
# Second sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_5h)):
        temp_index_var = h5[index_var]
        df_5h = pd.concat([Crypto_21_24_5h.iloc[:, crypto_var], return_i_21_24.iloc[:, temp_index_var]], axis = 1)
        df_5h = df_5h.apply(pd.to_numeric, errors = 'coerce')
        model_5h = VAR(df_5h)
        lag_order_5h = model_5h.select_order(maxlags = 10)
        optimal_lag_5h = lag_order_5h.aic
        p_distribution_21_24.append({'Lag': optimal_lag_5h})
# Full sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_5h)):
        temp_index_var = h5[index_var]
        df_5h = pd.concat([Crypto_final_5h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_5h = df_5h.apply(pd.to_numeric, errors = 'coerce')
        model_5h = VAR(df_5h)
        lag_order_5h = model_5h.select_order(maxlags = 10)
        optimal_lag_5h = lag_order_5h.aic
        p_distribution_19_24.append({'Lag': optimal_lag_5h})

#15:00:00
#First sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_15h)):
        temp_index_var = h15[index_var]
        df_15h = pd.concat([Crypto_19_20_15h.iloc[:, crypto_var], return_i_19_20.iloc[:, temp_index_var]], axis = 1)
        df_15h = df_5h.apply(pd.to_numeric, errors = 'coerce')
        model_15h = VAR(df_15h)
        lag_order_15h = model_15h.select_order(maxlags = 10)
        optimal_lag_15h = lag_order_15h.aic
        p_distribution_19_20.append({'Lag': optimal_lag_15h})
# Second sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_15h)):
        temp_index_var = h15[index_var]
        df_15h = pd.concat([Crypto_21_24_15h.iloc[:, crypto_var], return_i_21_24.iloc[:, temp_index_var]], axis = 1)
        df_15h = df_15h.apply(pd.to_numeric, errors = 'coerce')
        model_15h = VAR(df_15h)
        lag_order_15h = model_15h.select_order(maxlags = 10)
        optimal_lag_15h = lag_order_15h.aic
        p_distribution_21_24.append({'Lag': optimal_lag_15h})
# Full sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_15h)):
        temp_index_var = h15[index_var]
        df_15h = pd.concat([Crypto_final_15h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_15h = df_15h.apply(pd.to_numeric, errors = 'coerce')
        model_15h = VAR(df_15h)
        lag_order_15h = model_15h.select_order(maxlags = 10)
        optimal_lag_15h = lag_order_15h.aic
        p_distribution_19_24.append({'Lag': optimal_lag_15h})

#17:00:00
#First sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_17h)):
        temp_index_var = h17[index_var]
        df_17h = pd.concat([Crypto_19_20_17h.iloc[:, crypto_var], return_i_19_20.iloc[:, temp_index_var]], axis = 1)
        df_17h = df_17h.apply(pd.to_numeric, errors = 'coerce')
        model_17h = VAR(df_17h)
        lag_order_17h = model_17h.select_order(maxlags = 10)
        optimal_lag_17h = lag_order_17h.aic
        p_distribution_19_20.append({'Lag': optimal_lag_17h})
# Second sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_17h)):
        temp_index_var = h17[index_var]
        df_17h = pd.concat([Crypto_21_24_17h.iloc[:, crypto_var], return_i_21_24.iloc[:, temp_index_var]], axis = 1)
        df_17h = df_17h.apply(pd.to_numeric, errors = 'coerce')
        model_17h = VAR(df_17h)
        lag_order_17h = model_17h.select_order(maxlags = 10)
        optimal_lag_17h = lag_order_17h.aic
        p_distribution_21_24.append({'Lag': optimal_lag_17h})
# Full sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_17h)):
        temp_index_var = h17[index_var]
        df_17h = pd.concat([Crypto_final_17h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_17h = df_17h.apply(pd.to_numeric, errors = 'coerce')
        model_17h = VAR(df_17h)
        lag_order_17h = model_17h.select_order(maxlags = 10)
        optimal_lag_17h = lag_order_17h.aic
        p_distribution_19_24.append({'Lag': optimal_lag_17h})

#20:00:00
#First sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_20h)):
        temp_index_var = h20[index_var]
        df_20h = pd.concat([Crypto_19_20_20h.iloc[:, crypto_var], return_i_19_20.iloc[:, temp_index_var]], axis = 1)
        df_20h = df_20h.apply(pd.to_numeric, errors = 'coerce')
        model_20h = VAR(df_20h)
        lag_order_20h = model_20h.select_order(maxlags = 10)
        optimal_lag_20h = lag_order_20h.aic
        p_distribution_19_20.append({'Lag': optimal_lag_20h})
# Second sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_20h)):
        temp_index_var = h20[index_var]
        df_20h = pd.concat([Crypto_21_24_20h.iloc[:, crypto_var], return_i_21_24.iloc[:, temp_index_var]], axis = 1)
        df_20h = df_20h.apply(pd.to_numeric, errors = 'coerce')
        model_20h = VAR(df_20h)
        lag_order_20h = model_20h.select_order(maxlags = 10)
        optimal_lag_20h = lag_order_20h.aic
        p_distribution_21_24.append({'Lag': optimal_lag_20h})
# Full sample
for crypto_var in range(len(crypto_vars)):
    for index_var in range(len(index_var_20h)):
        temp_index_var = h20[index_var]
        df_20h = pd.concat([Crypto_final_20h.iloc[:, crypto_var], Index_final.iloc[:, temp_index_var]], axis = 1)
        df_20h = df_20h.apply(pd.to_numeric, errors = 'coerce')
        model_20h = VAR(df_20h)
        lag_order_20h = model_20h.select_order(maxlags = 10)
        optimal_lag_20h = lag_order_20h.aic
        p_distribution_19_24.append({'Lag': optimal_lag_20h})
#%%
df_p_distribution_19_20 = pd.DataFrame(p_distribution_19_20)
df_p_distribution_21_24 = pd.DataFrame(p_distribution_21_24)
df_p_distribution_19_24 = pd.DataFrame(p_distribution_19_24)
merged = pd.concat([df_p_distribution_19_20, df_p_distribution_21_24, df_p_distribution_19_24])
p_distribution = merged['Lag'].value_counts().reset_index()
p_distribution.columns = ["Lag", "Frequency"]
p_distribution = p_distribution.sort_values(by = "Lag")
p_distribution = p_distribution.reset_index(drop = True)
p_distribution = p_distribution[1::]
freq_19_20 = df_p_distribution_19_20['Lag'].value_counts().sort_index()
freq_21_24 = df_p_distribution_21_24['Lag'].value_counts().sort_index()
freq_19_24 = df_p_distribution_19_24['Lag'].value_counts().sort_index()
index = sorted(set(df_p_distribution_19_20['Lag']) | set(df_p_distribution_21_24['Lag']) | set(df_p_distribution_19_24['Lag']))
freq_19_20 = freq_19_20.reindex(index, fill_value = 0)
freq_21_24 = freq_21_24.reindex(index, fill_value = 0)
freq_19_24 = freq_19_24.reindex(index, fill_value = 0)
cumulative_21_24 = freq_19_20 + freq_21_24

# Plotting the stacked bars
plt.figure(figsize = (12, 8))
plt.bar(index, freq_19_20, color = 'skyblue', edgecolor = 'black', label = '2019-2020')
plt.bar(index, freq_21_24, bottom = freq_19_20, color = 'lightgreen', edgecolor = 'black', label = '2021-2024')
plt.bar(index, freq_19_24, bottom = cumulative_21_24, color = 'salmon', edgecolor = 'black', label = '2019-2024')

plt.xlabel('Lag', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.title('Histogram of p from the VAR(p) model', fontsize = 16)
plt.xticks(index, fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend()
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.7)
plt.savefig('histogram_VAR_p_model.png')
plt.show()
#%%
##################################################
##################################################
##                                              ##
##             5) Granger Causality             ##
##                                              ##
##################################################
##################################################
# Creating matrix
variables = df_final.columns
df_gc_19_20_ctoe = pd.DataFrame(0, columns = names_crypto, index = names_equity)
df_gc_21_24_ctoe = pd.DataFrame(0, columns = names_crypto, index = names_equity)
df_gc_cryptotoequity = pd.DataFrame(0, columns = names_crypto, index = names_equity)
df_gc_19_20_etoc = pd.DataFrame(0, columns = names_equity, index = names_crypto)
df_gc_21_24_etoc = pd.DataFrame(0, columns = names_equity, index = names_crypto)
df_gc_equitytocrypto = pd.DataFrame(0, columns = names_equity, index = names_crypto)

def granger_causality_matrix(data, variables, maxlag = 10, test = 'ssr_chi2test'):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns = variables, index = variables)
    for c in df.columns:
        for r in df.index:
            if c != r:
                # r = effect, c = cause
                model = VAR(data[[r, c]].apply(pd.to_numeric, errors = 'coerce'))
                lag_order = model.select_order(maxlags = 10)
                if lag_order.aic > 0:
                    optimal_lag = lag_order.aic
                    test_result = grangercausalitytests(data[[r, c]], maxlag = optimal_lag, verbose = False)
                    p_values = [round(test_result[i+1][0][test][1], 4) for i in range(optimal_lag)]
                    min_p_value = np.min(p_values)
                    df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
# Perform Granger causality tests
# 2019 - 2021
# Nikkei vs Crypto (5:00)
granger_results_JP_19_20 = granger_causality_matrix(temp_JP, variables, maxlag = 10)
#print(granger_results_JP_19_20)
df_gc_19_20_ctoe.iloc[1, :] = granger_results_JP_19_20.iloc[7, :6].values
df_gc_19_20_etoc.iloc[:, 1] = granger_results_JP_19_20.iloc[:6, 7].values
#DAX and FTSE 100 vs Crypto:
granger_results_UK_19_20 = granger_causality_matrix(temp_UK, variables, maxlag = 10)
#print(granger_results_UK_19_20)
df_gc_19_20_ctoe.iloc[7:9, :] = granger_results_UK_19_20.iloc[13:15, :6].values
df_gc_19_20_etoc.iloc[:, 7:9] = granger_results_UK_19_20.iloc[:6, 13:15].values
# FTSE MIB and CAC 40 vs Crypto
granger_results_IT_19_20 = granger_causality_matrix(temp_IT, variables, maxlag = 10)
#print(granger_results_IT_19_20)
df_gc_19_20_ctoe.iloc[3, :] = granger_results_IT_19_20.iloc[9, :6].values
df_gc_19_20_etoc.iloc[:, 3] = granger_results_IT_19_20.iloc[:6, 9].values
df_gc_19_20_ctoe.iloc[6, :] = granger_results_IT_19_20.iloc[12, :6].values
df_gc_19_20_etoc.iloc[:, 6] = granger_results_IT_19_20.iloc[:6, 12].values
# US vs Crypto
granger_results_US_19_20 = granger_causality_matrix(temp_US, variables, maxlag = 10)
#print(granger_results_US_19_20)
df_gc_19_20_ctoe.iloc[0, :] = granger_results_US_19_20.iloc[6, :6].values
df_gc_19_20_etoc.iloc[:, 0] = granger_results_US_19_20.iloc[:6, 6].values
df_gc_19_20_ctoe.iloc[2, :] = granger_results_US_19_20.iloc[8, :6].values
df_gc_19_20_etoc.iloc[:, 2] = granger_results_US_19_20.iloc[:6, 8].values
df_gc_19_20_ctoe.iloc[4:6, :] = granger_results_US_19_20.iloc[10:12, :6].values
df_gc_19_20_etoc.iloc[:, 4:6] = granger_results_US_19_20.iloc[:6, 10:12].values

#2022 - 2024
# Nikkei vs Crypto (5:00)
granger_results_JP_21_24 = granger_causality_matrix(temp_JPc, variables, maxlag = 10)
#print(granger_results_JP_21_24)
df_gc_21_24_ctoe.iloc[1, :] = granger_results_JP_21_24.iloc[7, :6].values
df_gc_21_24_etoc.iloc[:, 1] = granger_results_JP_21_24.iloc[:6, 7].values
#DAX and FTSE 100 vs Crypto:
granger_results_UK_21_24 = granger_causality_matrix(temp_UKc, variables, maxlag = 10)
#print(granger_results_UK_21_24)
df_gc_21_24_ctoe.iloc[7:9, :] = granger_results_UK_21_24.iloc[13:15, :6].values
df_gc_21_24_etoc.iloc[:, 7:9] = granger_results_UK_21_24.iloc[:6, 13:15].values
# FTSE MIB and CAC 40 vs Crypto
granger_results_IT_21_24 = granger_causality_matrix(temp_ITc, variables, maxlag = 10)
#print(granger_results_IT_21_24)
df_gc_21_24_ctoe.iloc[3, :] = granger_results_IT_21_24.iloc[9, :6].values
df_gc_21_24_etoc.iloc[:, 3] = granger_results_IT_21_24.iloc[:6, 9].values
df_gc_21_24_ctoe.iloc[6, :] = granger_results_IT_21_24.iloc[12, :6].values
df_gc_21_24_etoc.iloc[:, 6] = granger_results_IT_21_24.iloc[:6, 12].values
# US vs Crypto
granger_results_US_21_24 = granger_causality_matrix(temp_USc, variables, maxlag = 10)
#print(granger_results_US_21_24)
df_gc_21_24_ctoe.iloc[0, :] = granger_results_US_21_24.iloc[6, :6].values
df_gc_21_24_etoc.iloc[:, 0] = granger_results_US_21_24.iloc[:6, 6].values
df_gc_21_24_ctoe.iloc[2, :] = granger_results_US_21_24.iloc[8, :6].values
df_gc_21_24_etoc.iloc[:, 2] = granger_results_US_21_24.iloc[:6, 8].values
df_gc_21_24_ctoe.iloc[4:6, :] = granger_results_US_21_24.iloc[10:12, :6].values
df_gc_21_24_etoc.iloc[:, 4:6] = granger_results_US_21_24.iloc[:6, 10:12].values

# Full sample
# Nikkei vs Crypto (5:00)
granger_causality_full_JP = granger_causality_matrix(temp_final_5h, variables, maxlag = 10)
#print(granger_causality_full_JP)
df_gc_cryptotoequity.iloc[1, :] = granger_causality_full_JP.iloc[7, :6].values
df_gc_equitytocrypto.iloc[:, 1] = granger_causality_full_JP.iloc[:6, 7].values

#DAX and FTSE 100 vs Crypto:
granger_results_full_UK = granger_causality_matrix(temp_final_15h, variables, maxlag = 10)
#print(granger_results_full_UK)
df_gc_cryptotoequity.iloc[7:9, :] = granger_results_full_UK.iloc[13:15, :6].values
df_gc_equitytocrypto.iloc[:, 7:9] = granger_results_full_UK.iloc[:6, 13:15].values
# FTSE MIB and CAC 40 vs Crypto
granger_results_full_IT = granger_causality_matrix(temp_final_17h, variables, maxlag = 10)
#print(granger_results_full_IT)
df_gc_cryptotoequity.iloc[3, :] = granger_results_full_IT.iloc[9, :6].values
df_gc_equitytocrypto.iloc[:, 3] = granger_results_full_IT.iloc[:6, 9].values
df_gc_cryptotoequity.iloc[6, :] = granger_results_full_IT.iloc[12, :6].values
df_gc_equitytocrypto.iloc[:, 6] = granger_results_full_IT.iloc[:6, 12].values
# US vs Crypto
granger_results_full_US = granger_causality_matrix(temp_final_20h, variables, maxlag = 10)
#print(granger_results_full_US)
df_gc_cryptotoequity.iloc[0, :] = granger_results_full_US.iloc[6, :6].values
df_gc_equitytocrypto.iloc[:, 0] = granger_results_full_US.iloc[:6, 6].values
df_gc_cryptotoequity.iloc[2, :] = granger_results_full_US.iloc[8, :6].values
df_gc_equitytocrypto.iloc[:, 2] = granger_results_full_US.iloc[:6, 8].values
df_gc_cryptotoequity.iloc[4:6, :] = granger_results_full_US.iloc[10:12, :6].values
df_gc_equitytocrypto.iloc[:, 4:6] = granger_results_full_US.iloc[:6, 10:12].values

with pd.ExcelWriter(r"C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\3. Deliverables\Granger Causality.xlsx", engine = 'xlsxwriter') as writer:
    df_gc_19_20_ctoe.to_excel(writer, sheet_name = 'df_gc_19_20_ctoe', index = True)
    df_gc_19_20_etoc.to_excel(writer, sheet_name = 'df_gc_19_20_etoc', index = True)
    df_gc_21_24_ctoe.to_excel(writer, sheet_name = 'df_gc_21_24_ctoe', index = True)
    df_gc_21_24_etoc.to_excel(writer, sheet_name = 'df_gc_21_24_etoc', index = True)
    df_gc_cryptotoequity.to_excel(writer, sheet_name = 'df_gc_cryptotoequity', index = True)
    df_gc_equitytocrypto.to_excel(writer, sheet_name = 'df_gc_equitytocrypto', index = True)

# Create the count-link table
def apply_threshold(df):
    return df.applymap(lambda x: 1 if x < 0.05 else 0)

df_gc_19_20_ctoe_new = apply_threshold(df_gc_19_20_ctoe)
df_gc_21_24_ctoe_new = apply_threshold(df_gc_21_24_ctoe)
df_gc_cryptotoequity_new = apply_threshold(df_gc_cryptotoequity)
df_gc_19_20_etoc_new = apply_threshold(df_gc_19_20_etoc)
df_gc_21_24_etoc_new = apply_threshold(df_gc_21_24_etoc)
df_gc_equitytocrypto_new = apply_threshold(df_gc_equitytocrypto)
#%%
def calculate_total(df):
    col_total = df.sum(axis = 0) / len(df)
    row_total = df.sum(axis = 1) / len(df.columns)
    df['Total'] = row_total
    total_row = col_total.to_dict()
    total_row['Total'] = row_total.mean()
    df.loc['Total'] = total_row
    return df

df_gc_19_20_ctoe_new_total = calculate_total(df_gc_19_20_ctoe_new.copy())
df_gc_21_24_ctoe_new_total = calculate_total(df_gc_21_24_ctoe_new.copy())
df_gc_cryptotoequity_new_total = calculate_total(df_gc_cryptotoequity_new.copy())
df_gc_19_20_etoc_new_total = calculate_total(df_gc_19_20_etoc_new.copy())
df_gc_21_24_etoc_new_total = calculate_total(df_gc_21_24_etoc_new.copy())
df_gc_equitytocrypto_new_total = calculate_total(df_gc_equitytocrypto_new.copy())

total_19_20_ctoe = df_gc_19_20_ctoe_new_total.iloc[[-1]]
total_21_24_ctoe = df_gc_21_24_ctoe_new_total.iloc[[-1]]
total_cryptotoequity = df_gc_cryptotoequity_new_total.iloc[[-1]]
total_19_20_etoc = df_gc_19_20_etoc_new_total.iloc[[-1]]
total_21_24_etoc = df_gc_21_24_etoc_new_total.iloc[[-1]]
total_equitytocrypto = df_gc_equitytocrypto_new_total.iloc[[-1]]


combined_table_19_20 = pd.concat([total_19_20_ctoe, total_19_20_etoc], axis = 1)
combined_table_21_24 = pd.concat([total_21_24_ctoe, total_21_24_etoc], axis = 1)
combined_table_all = pd.concat([total_cryptotoequity, total_equitytocrypto], axis = 1)
final_combined_table = pd.concat([combined_table_19_20, combined_table_21_24, combined_table_all], keys = ['2019-2020', '2021-2024', 'Combined'])
transposed_final_combined_table = final_combined_table.T
formatted_table = transposed_final_combined_table.reset_index()
formatted_table = formatted_table.drop(index = 6)
formatted_table.columns = ['Assets', 'Sub-dataset 2019 - 2020', 'Sub-dataset 2021 - 2024', 'Full dataset 2019 - 2024']

mean_values = formatted_table[['Sub-dataset 2019 - 2020', 'Sub-dataset 2021 - 2024', 'Full dataset 2019 - 2024']].mean()
formatted_table.loc[16, ['Sub-dataset 2019 - 2020', 'Sub-dataset 2021 - 2024', 'Full dataset 2019 - 2024']] = mean_values.values
plt.figure(figsize = (14, 10))
sns.heatmap(formatted_table.set_index('Assets'), annot = True, cmap = "coolwarm", cbar = True, linewidths = .5)
plt.title('Granger causality link ratio for the 3 datasets')
plt.ylabel("Assets")
plt.savefig('Granger causality link ratio.png')
plt.show()



















#%%
cryptos = df_gc_19_20_ctoe.columns
equities = df_gc_19_20_ctoe.index

# Network Graph
def create_network_graph(granger_matrix, threshold = 0.05):
    G = nx.DiGraph()
    for row in equities:
        for col in cryptos:
            p_value = granger_matrix.loc[row, col]
            if p_value < threshold:
                G.add_edge(row, col, weight = p_value)
    return G

G = create_network_graph(df_gc_19_20_ctoe, threshold = 0.05)
plt.figure(figsize = (12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray', arrows = True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
plt.title('Granger Causality Network')
plt.show()
#%%



#%%
##################################################
##################################################
##                                              ##
##             5.1) GC final graphs             ##
##                                              ##
##################################################
##################################################
# Crypto to Equity (2019 - 2020)
#Create a Bipartite graph
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_19_20_ctoe.loc[equity, crypto]
        if p_value < threshold:
            B.add_edge(crypto, equity, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (0, i)) for i, node in enumerate(cryptos))  # Cryptos on the left
pos.update((node, (1, i)) for i, node in enumerate(equities))   # Equities on the right
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Crypto -> Equity) - 2019-2021')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Crypto to Equity) - 2019-2021.png')
plt.show()

# Equity to Crypto (2019 - 2020)
#Create a Bipartite graph
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_19_20_etoc.loc[crypto, equity]
        if p_value < threshold:
            B.add_edge(equity, crypto, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (1, i)) for i, node in enumerate(cryptos))  # Cryptos on the right
pos.update((node, (0, i)) for i, node in enumerate(equities))   # Equities on the left
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Equity -> Crypto) - 2019-2021')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Equity to Crypto) - 2019-2021.png')
plt.show()

# Crypto to Equity (2021 - 2024)
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_21_24_ctoe.loc[equity, crypto]
        if p_value < threshold:
            B.add_edge(crypto, equity, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (0, i)) for i, node in enumerate(cryptos))  # Cryptos on the left
pos.update((node, (1, i)) for i, node in enumerate(equities))   # Equities on the right
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Crypto -> Equity) - 2021-2014')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Crypto to Equity) - 2021-2024.png')
plt.show()

# Equity to Crypto (2021 - 2024)
#Create a Bipartite graph
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_21_24_etoc.loc[crypto, equity]
        if p_value < threshold:
            B.add_edge(equity, crypto, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (1, i)) for i, node in enumerate(cryptos))  # Cryptos on the right
pos.update((node, (0, i)) for i, node in enumerate(equities))   # Equities on the left
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Equity -> Crypto) - 2021-2014')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Equity to Crypto) - 2021-2014.png')
plt.show()

##### Full sample #####
# Crypto to Equity (2019 - 2024)
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_cryptotoequity.loc[equity, crypto]
        if p_value < threshold:
            B.add_edge(crypto, equity, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (0, i)) for i, node in enumerate(cryptos))  # Cryptos on the left
pos.update((node, (1, i)) for i, node in enumerate(equities))   # Equities on the right
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Crypto -> Equity) - Full sample')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Crypto to Equity) - Full sample.png')
plt.show()

# Equity to Crypto (2019 - 2024)
#Create a Bipartite graph
B = nx.DiGraph()
# Add nodes
G.add_nodes_from(cryptos, bipartite = 0)
G.add_nodes_from(equities, bipartite = 1)
threshold = 0.05  # Define significance level
for equity in equities:
    for crypto in cryptos:
        p_value = df_gc_equitytocrypto.loc[crypto, equity]
        if p_value < threshold:
            B.add_edge(equity, crypto, weight = -np.log(p_value))
# Draw the bipartite graph
plt.figure(figsize = (12, 8))
pos = {}
pos.update((node, (1, i)) for i, node in enumerate(cryptos))  # Cryptos on the right
pos.update((node, (0, i)) for i, node in enumerate(equities))   # Equities on the left
edges = B.edges(data = True)
nx.draw_networkx_nodes(B, pos, nodelist = cryptos, node_color = 'lightblue', node_size = 1000, label = 'Crypto')
nx.draw_networkx_nodes(B, pos, nodelist = equities, node_color = 'lightgreen', node_size = 1000, label = 'Equity Indexes')
nx.draw_networkx_edges(B, pos, edgelist = edges, arrowstyle = '->', arrowsize = 20, edge_color = 'gray')
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize = 12, ha = 'center', va = 'center')
plt.title('Bipartite Graph of Granger Causality (Equity -> Crypto) - Full sample')
plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.01), ncol = 2)
plt.savefig('Bipartite Graph of Granger Causality (Equity to Crypto) - Full sample.png')
plt.show()
#%%
##################################################
##################################################
##                                              ##
##                  5.1) Heatmaps               ##
##                                              ##
##################################################
##################################################
with open('Granger_Causality_Matrix_p_values_Crypto_to_Equity_2019_2020.tex', 'w') as f:
    f.write(df_gc_19_20_ctoe.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Crypto to Equity - 2019-2020'))

with open('Granger_Causality_Matrix_p_values_Equity_to_Crypto_2019_2020.tex', 'w') as f:
    f.write(df_gc_19_20_etoc.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Equity to Crypto - 2019-2020'))

with open('Granger_Causality_Matrix_p_values_Crypto_to_Equity_2021_2024.tex', 'w') as f:
    f.write(df_gc_21_24_ctoe.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Crypto to Equity - 2021-2024'))

with open('Granger_Causality_Matrix_p_values_Equity_to_Crypto_2021_2024.tex', 'w') as f:
    f.write(df_gc_21_24_etoc.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Equity to Crypto - 2021-2024'))

with open('Granger_Causality_Matrix_p_values_Crypto_to_Equity_Full_sample.tex', 'w') as f:
    f.write(df_gc_cryptotoequity.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Crypto to Equity - Full sample'))

with open('Granger_Causality_Matrix_p_values_Equity_to_Crypto_Full_sample.tex', 'w') as f:
    f.write(df_gc_equitytocrypto.to_latex(index = True, caption = 'Granger Causality Matrix (p-values) - Equity to Crypto - Full sample'))

with pd.ExcelWriter('Granger_Causality_Matrices.xlsx', engine = 'xlsxwriter') as writer:
    df_gc_19_20_ctoe.to_excel(writer, sheet_name = 'Crypto to Equity 2019-2020', index = True)
    df_gc_19_20_etoc.to_excel(writer, sheet_name = 'Equity to Crypto 2019-2020', index = True)
    df_gc_21_24_ctoe.to_excel(writer, sheet_name = 'Crypto to Equity 2021-2024', index = True)
    df_gc_21_24_etoc.to_excel(writer, sheet_name = 'Equity to Crypto 2021-2024', index = True)
    df_gc_cryptotoequity.to_excel(writer, sheet_name = 'Crypto to Equity Full', index = True)
    df_gc_equitytocrypto.to_excel(writer, sheet_name = 'Equity to Crypto Full', index = True)
#%%
# Heatmap
#  Crypto to Equity - 2019-2020
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_19_20_ctoe, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Crypto to Equity - 2019-2020')
plt.savefig('Granger Causality Matrix (p-values) - Crypto to Equity - 2019-2020.png')
#  Equity to Crypto - 2019-2020
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_19_20_etoc, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Equity to Crypto - 2019-2020')
plt.savefig('Granger Causality Matrix (p-values) - Equity to Crypto - 2019-2020.png')
#  Crypto to Equity - 2021-2024
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_21_24_ctoe, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Crypto to Equity - 2021-2024')
plt.savefig('Granger Causality Matrix (p-values) - Crypto to Equity - 2021-2024.png')
#  Equity to Crypto - 2021-2024
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_21_24_etoc, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Equity to Crypto - 2021-2024')
plt.savefig('Granger Causality Matrix (p-values) - Equity to Crypto - 2021-2024.png')
#  Crypto to Equity - Full sample
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_cryptotoequity, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Crypto to Equity - Full sample')
plt.savefig('Granger Causality Matrix (p-values) - Crypto to Equity - Full sample.png')
#  Equity to Crypto - Full sample
plt.figure(figsize = (10, 8))
sns.heatmap(df_gc_equitytocrypto, annot = True, cbar = True, cmap = 'coolwarm', linewidths = 0.5, linecolor = 'black')
plt.title('Granger Causality Matrix (p-values) - Equity to Crypto - Full sample')
plt.savefig('Granger Causality Matrix (p-values) - Equity to Crypto - Full sample.png')

#%%
directory = r'C:\Users\wow10\Google Drive\Quant. Finance\Tesi\Master thesis 2023\2. Working\LaTeX'
os.makedirs(directory, exist_ok = True)
file_path_df_gc_19_20_ctoe = os.path.join(directory, 'Granger Causality 19-21 Crypto to Equity.tex')
file_path_df_gc_19_20_etoc = os.path.join(directory, 'Granger Causality 19-21 Equity to Crypto.tex')
file_path_df_gc_21_24_ctoe = os.path.join(directory, 'Granger Causality 22-24 Crypto to Equity.tex')
file_path_df_gc_21_24_etoc = os.path.join(directory, 'Granger Causality 22-24 Equity to Crypto.tex')
file_path_df_gc_19_24_ctoe = os.path.join(directory, 'Granger Causality 19-24 Crypto to Equity.tex')
file_path_df_gc_19_24_etoc = os.path.join(directory, 'Granger Causality 19-24 Equity to Crypto.tex')
with open(file_path_df_gc_19_20_ctoe, 'w') as f:
    f.write(df_gc_19_20_ctoe.to_latex())
with open(file_path_df_gc_19_20_etoc, 'w') as f:
    f.write(df_gc_19_20_etoc.to_latex())
with open(file_path_df_gc_21_24_ctoe, 'w') as f:
    f.write(df_gc_21_24_ctoe.to_latex())
with open(file_path_df_gc_21_24_etoc, 'w') as f:
    f.write(df_gc_21_24_etoc.to_latex())
with open(file_path_df_gc_19_24_ctoe, 'w') as f:
    f.write(df_gc_cryptotoequity.to_latex())
with open(file_path_df_gc_19_24_etoc, 'w') as f:
    f.write(df_gc_equitytocrypto.to_latex())

#%%
# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_19_20_ctoe.index, bipartite = 0)
G.add_nodes_from(df_gc_19_20_ctoe.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_19_20_ctoe.index:
    for col in df_gc_19_20_ctoe.columns:
        weight = df_gc_19_20_ctoe.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_19_20_ctoe.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Crypto to Equity (2019 - 2021)')
plt.savefig('Granger Causality Bipartite Network - Crypto to Equity (2019 - 2021).png')
plt.show()

# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_19_20_etoc.index, bipartite = 0)
G.add_nodes_from(df_gc_19_20_etoc.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_19_20_etoc.index:
    for col in df_gc_19_20_etoc.columns:
        weight = df_gc_19_20_etoc.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_19_20_etoc.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Equity to Crypto (2019 - 2021)')
plt.savefig('Granger Causality Bipartite Network - Equity to Crypto (2019 - 2021).png')
plt.show()

top_nodes, bottom_nodes = nx.bipartite.sets(G)

top_degrees = [G.degree(n) for n in top_nodes]
bottom_degrees = [G.degree(n) for n in bottom_nodes]

print("Top nodes degree distribution:", top_degrees)
print("Bottom nodes degree distribution:", bottom_degrees)

# Visualization: Visualize the bipartite graph to get an intuitive understanding of the connections and overall structure.
pos = nx.spring_layout(G)  # Or use nx.bipartite_layout(G, top_nodes)
nx.draw(G, pos, with_labels = True, node_color = 'lightblue', edge_color = 'gray')
plt.show()

# Bipartiteness Check
is_bipartite = nx.is_bipartite(G)
print(f"Graph is bipartite: {is_bipartite}")

# Projection Analysis: Project the bipartite graph into two one-mode networks (one for each set of nodes). This helps in analyzing relationships within each set.
top_projection = nx.bipartite.projected_graph(G, top_nodes)
bottom_projection = nx.bipartite.projected_graph(G, bottom_nodes)
# Analyze properties of the projections
print("Top projection edges:", top_projection.edges())
print("Bottom projection edges:", bottom_projection.edges())
# Plotting the top projection
plt.figure(figsize = (10, 5))
plt.title("Top Projection")
pos = nx.spring_layout(top_projection)
nx.draw(top_projection, pos, with_labels = True, node_color = 'lightblue', edge_color = 'gray', node_size = 500, font_size = 10)
plt.show()

# Plotting the bottom projection
plt.figure(figsize = (10, 5))
plt.title("Bottom Projection")
pos = nx.spring_layout(bottom_projection)
nx.draw(bottom_projection, pos, with_labels = True, node_color = 'lightgreen', edge_color = 'gray', node_size = 500, font_size = 10)
plt.show()




# Connectivity and Components: Check for the shortest paths between nodes, which can give insights into the efficiency of connectivity within the graph.
components = list(nx.connected_components(G))
print(f"Connected components: {components}")
path_lengths = dict(nx.all_pairs_shortest_path_length(G))
print("Shortest path lengths:", path_lengths)

# Centrality Measures: Measure the importance of nodes based on their degree.
degree_centrality = nx.degree_centrality(G)
print("Degree centrality:", degree_centrality)

# Bipartite Clustering Coefficient: Calculate the clustering coefficient for bipartite networks.
clustering_coefficient = nx.bipartite.clustering(G)
print("Bipartite clustering coefficient:", clustering_coefficient)

# Assortativity: Check for assortative mixing, which can indicate if nodes tend to connect to other nodes that are similar (e.g., by degree).
assortativity = nx.degree_assortativity_coefficient(G)
print("Degree assortativity:", assortativity)

#Community Detection: Identify communities or clusters within the bipartite graph using algorithms like the Girvan-Newman algorithm or modularity-based methods.
communities = girvan_newman(G)
top_level_communities = next(communities)
print("Top-level communities:", top_level_communities)

#####################################################################################################################################
#%%
##################################################
##################################################
##                                              ##
##             4.1) Rolling window              ##
##                                              ##
##################################################
##################################################
# Equity to Crypto
rolling_window = 60 # 2 months
shift = 20 # 2 weeks
p = 6
test = 'ssr_chi2test'
# Adjust the dataset
Crypto_final_5h_rw = Crypto_final_5h.iloc[:-5]
Crypto_final_15h_rw = Crypto_final_15h.iloc[:-5]
Crypto_final_17h_rw = Crypto_final_17h.iloc[:-5]
Crypto_final_20h_rw = Crypto_final_20h.iloc[:-5]
Index_final_rw = Index_final.iloc[:-5]

def grangercausality_index_to_crypto(df1, df2, name_index, h_array): # df1 = Indexes df2 = Crypto
    ratios = []
    for i in range(0, len(df1) - rolling_window, shift):
        p_values = []
        for crypto_var in range(len(crypto_vars)):
            for index_var in range(len(name_index)):
                temp_index_var = h_array[index_var]
                crypto = df2.iloc[i:(i + rolling_window), crypto_var]
                indexes = df1.iloc[i:(i + rolling_window), temp_index_var]
                df_rw = pd.concat([crypto, indexes], axis = 1)
                df_rw = df_rw.apply(pd.to_numeric, errors = 'coerce')
                if df_rw.nunique().min() == 1:
                    p_values.append(1)
                    continue
                try:                
                    test_result_rw = grangercausalitytests(df_rw, maxlag = [p], verbose = False)
                    p_value_rw = round(test_result_rw[p][0][test][1], 3)
                    if p_value_rw < 0.05:
                        p_values.append(0)
                    else:
                        p_values.append(1)
                except (ValueError, np.linalg.LinAlgError, InfeasibleTestError) as e:
                    p_values.append(1)
                    continue
        # Calculate the ratio of acceteptance for the current window
        ratio = p_values.count(0) / len(p_values)
        ratios.append(ratio)   
    return ratios

granger_rw_5h = grangercausality_index_to_crypto(df1 = Index_final_rw, df2 = Crypto_final_5h_rw, name_index = index_var_5h, h_array = h5)
granger_rw_15h = grangercausality_index_to_crypto(df1 = Index_final_rw, df2 = Crypto_final_15h_rw, name_index = index_var_15h, h_array = h15)
granger_rw_17h = grangercausality_index_to_crypto(df1 = Index_final_rw, df2 = Crypto_final_17h_rw, name_index = index_var_17h, h_array = h17)
granger_rw_20h = grangercausality_index_to_crypto(df1 = Index_final_rw, df2 = Crypto_final_20h_rw, name_index = index_var_20h, h_array = h20)
granger_rw_5h = pd.DataFrame(granger_rw_5h, columns = ["C1"])
granger_rw_15h = pd.DataFrame(granger_rw_15h, columns = ["C1"])
granger_rw_17h = pd.DataFrame(granger_rw_17h, columns = ["C1"])
granger_rw_20h = pd.DataFrame(granger_rw_20h, columns = ["C1"])

# Equity to Crypto
rollingwindow_count_etoc = pd.DataFrame(0, columns = names_equity + ["Total"], index = names_crypto + ["Total"])
#5h
#NIKKEI
index_values1 = granger_rw_5h.iloc[::6]
index_values2 = granger_rw_5h.iloc[1::6]
index_values3 = granger_rw_5h.iloc[2::6]
index_values4 = granger_rw_5h.iloc[3::6]
index_values5 = granger_rw_5h.iloc[4::6]
index_values6 = granger_rw_5h.iloc[5::6]
count_NIKKEI = (granger_rw_5h["C1"] < 0.05).sum() / 6 * len(index_values1)
rollingwindow_count_etoc.iloc[6, 1] = count_NIKKEI
rollingwindow_count_etoc.iloc[0, 1] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 1] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 1] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 1] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 1] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 1] = 100 * (index_values6 < 0.05).sum() / len(index_values1)

# 15h
#DAX
index_values1 = granger_rw_15h.iloc[::12]
index_values2 = granger_rw_15h.iloc[1::12]
index_values3 = granger_rw_15h.iloc[2::12]
index_values4 = granger_rw_15h.iloc[3::12]
index_values5 = granger_rw_15h.iloc[4::12]
index_values6 = granger_rw_15h.iloc[5::12]
rollingwindow_count_etoc.iloc[0, 7] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 7] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 7] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 7] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 7] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 7] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 7] = rollingwindow_count_etoc.iloc[0:6, 7].sum() / 6
#FTSE 100
index_values1 = granger_rw_15h.iloc[6::12]
index_values2 = granger_rw_15h.iloc[7::12]
index_values3 = granger_rw_15h.iloc[8::12]
index_values4 = granger_rw_15h.iloc[9::12]
index_values5 = granger_rw_15h.iloc[10::12]
index_values6 = granger_rw_15h.iloc[11::12]
rollingwindow_count_etoc.iloc[0, 8] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 8] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 8] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 8] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 8] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 8] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 8] = rollingwindow_count_etoc.iloc[0:6, 8].sum() / 6

#17h
#FTSE MIB
index_values1 = granger_rw_17h.iloc[::12]
index_values2 = granger_rw_17h.iloc[1::12]
index_values3 = granger_rw_17h.iloc[2::12]
index_values4 = granger_rw_17h.iloc[3::12]
index_values5 = granger_rw_17h.iloc[4::12]
index_values6 = granger_rw_17h.iloc[5::12]
rollingwindow_count_etoc.iloc[0, 3] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 3] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 3] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 3] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 3] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 3] = 100 * (index_values6 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[6, 3] =rollingwindow_count_etoc.iloc[0:6, 3].sum() / 6
#CAC 40
index_values1 = granger_rw_17h.iloc[6::12]
index_values2 = granger_rw_17h.iloc[7::12]
index_values3 = granger_rw_17h.iloc[8::12]
index_values4 = granger_rw_17h.iloc[9::12]
index_values5 = granger_rw_17h.iloc[10::12]
index_values6 = granger_rw_17h.iloc[11:12]
rollingwindow_count_etoc.iloc[0, 6] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 6] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 6] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 6] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 6] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 6] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 6] = rollingwindow_count_etoc.iloc[0:6, 6].sum() / 6

#20h
#S&P 500
index_values1 = granger_rw_20h.iloc[::24]
index_values2 = granger_rw_20h.iloc[1::24]
index_values3 = granger_rw_20h.iloc[2::24]
index_values4 = granger_rw_20h.iloc[3::24]
index_values5 = granger_rw_20h.iloc[4::24]
index_values6 = granger_rw_20h.iloc[5::24]
rollingwindow_count_etoc.iloc[0, 0] = 100 * (index_values1 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[1, 0] = 100 * (index_values2 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[2, 0] = 100 * (index_values3 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[3, 0] = 100 * (index_values4 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[4, 0] = 100 * (index_values5 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[5, 0] = 100 * (index_values6 < 0.05).sum() / len(index_values1) 
rollingwindow_count_etoc.iloc[6, 0] = rollingwindow_count_etoc.iloc[0:6, 0].sum() / 6

#RUSSELL 2000
index_values1 = granger_rw_20h.iloc[6::24]
index_values2 = granger_rw_20h.iloc[7::24]
index_values3 = granger_rw_20h.iloc[8::24]
index_values4 = granger_rw_20h.iloc[9::24]
index_values5 = granger_rw_20h.iloc[10::24]
index_values6 = granger_rw_20h.iloc[11::24]
rollingwindow_count_etoc.iloc[0, 2] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 2] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 2] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 2] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 2] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 2] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 2] = rollingwindow_count_etoc.iloc[0:6,2].sum() / 6
#NASDAQ
index_values1 = granger_rw_20h.iloc[12::24]
index_values2 = granger_rw_20h.iloc[13::24]
index_values3 = granger_rw_20h.iloc[14::24]
index_values4 = granger_rw_20h.iloc[15::24]
index_values5 = granger_rw_20h.iloc[16::24]
index_values6 = granger_rw_20h.iloc[17::24]
rollingwindow_count_etoc.iloc[0, 4] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 4] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 4] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 4] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 4] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 4] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 4] = rollingwindow_count_etoc.iloc[0:6, 4].sum() / 6
#Dow Jones
index_values1 = granger_rw_20h.iloc[18::24]
index_values2 = granger_rw_20h.iloc[19::24]
index_values3 = granger_rw_20h.iloc[20::24]
index_values4 = granger_rw_20h.iloc[21::24]
index_values5 = granger_rw_20h.iloc[22::24]
index_values6 = granger_rw_20h.iloc[23::24]
rollingwindow_count_etoc.iloc[0, 5] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[1, 5] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[2, 5] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[3, 5] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[4, 5] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[5, 5] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_etoc.iloc[6, 5] = rollingwindow_count_etoc.iloc[0:6, 5].sum() / 6

rollingwindow_count_etoc.iloc[0, 9] = rollingwindow_count_etoc.iloc[0, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[1, 9] = rollingwindow_count_etoc.iloc[1, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[2, 9] = rollingwindow_count_etoc.iloc[2, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[3, 9] = rollingwindow_count_etoc.iloc[3, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[4, 9] = rollingwindow_count_etoc.iloc[4, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[5, 9] = rollingwindow_count_etoc.iloc[5, 0:9].sum() / 9
rollingwindow_count_etoc.iloc[6, 9] = rollingwindow_count_etoc.iloc[6, 0:9].sum() / 9
latex_code = rollingwindow_count_etoc.to_latex()
with open('rollingwindow_count_etoc.tex', 'w') as file:
    file.write(latex_code)

granger_equity_to_crypto_rw = (np.array(granger_rw_5h) * (1/9)) + (np.array(granger_rw_15h) * (2/9)) + (np.array(granger_rw_17h) * (2/9)) + (np.array(granger_rw_20h) * (4/9))
granger_equity_to_crypto_rw = pd.DataFrame(granger_equity_to_crypto_rw, columns = ['Ratio'])
granger_equity_to_crypto_rw.index = Index_final_rw.index[(rolling_window)::(shift)]
def grangercausality_crypto_to_equity(df1, df2, name_index, h_array): # df1 = Crypto, df2 = Indexes
    ratios = []
    for i in range(0, len(df1) - rolling_window, shift):
        p_values = []
        for crypto_var in range(len(crypto_vars)):
            for index_var in range(len(name_index)):
                temp_index_var = h_array[index_var]
                crypto = df2.iloc[i:(i + rolling_window), crypto_var]
                indexes = df1.iloc[i:(i + rolling_window), temp_index_var]
                df_rw = pd.concat([indexes, crypto], axis = 1)
                df_rw = df_rw.apply(pd.to_numeric, errors = 'coerce')
                if df_rw.nunique().min() == 1:
                    p_values.append(1)
                    continue
                try:                
                    test_result_rw = grangercausalitytests(df_rw, maxlag = [p], verbose = False)
                    p_value_rw = round(test_result_rw[p][0][test][1], 3)
                    if p_value_rw < 0.05:
                        p_values.append(0)
                    else:
                        p_values.append(1)
                except (ValueError, np.linalg.LinAlgError, InfeasibleTestError) as e:
                    p_values.append(np.nan)
                    continue
        # Calculate the ratio of acceptance for the current window
        ratio = p_values.count(0) / len(p_values)
        ratios.append(ratio)             
    return ratios

granger_rw_5h_ctoe = grangercausality_crypto_to_equity(df1 = Index_final_rw, df2 = Crypto_final_5h_rw, name_index = index_var_5h, h_array = h5)
granger_rw_15h_ctoe = grangercausality_crypto_to_equity(df1 = Index_final_rw, df2 = Crypto_final_15h_rw, name_index = index_var_15h, h_array = h15)
granger_rw_17h_ctoe = grangercausality_crypto_to_equity(df1 = Index_final_rw, df2 = Crypto_final_17h_rw, name_index = index_var_17h, h_array = h17)
granger_rw_20h_ctoe = grangercausality_crypto_to_equity(df1 = Index_final_rw, df2 = Crypto_final_20h_rw, name_index = index_var_20h, h_array = h20)

granger_rw_5h_ctoe = pd.DataFrame(granger_rw_5h_ctoe, columns = ["C1"])
granger_rw_15h_ctoe = pd.DataFrame(granger_rw_15h_ctoe, columns = ["C1"])
granger_rw_17h_ctoe = pd.DataFrame(granger_rw_17h_ctoe, columns = ["C1"])
granger_rw_20h_ctoe = pd.DataFrame(granger_rw_20h_ctoe, columns = ["C1"])

granger_crypto_to_equity_rw = (np.array(granger_rw_5h_ctoe) * (1/9)) + (np.array(granger_rw_15h_ctoe) * (2/9)) + (np.array(granger_rw_17h_ctoe) * (2/9)) + (np.array(granger_rw_20h_ctoe) * (4/9))
granger_crypto_to_equity_rw = pd.DataFrame(granger_crypto_to_equity_rw, columns = ['Ratio'])
granger_crypto_to_equity_rw.index = Index_final_rw.index[(rolling_window)::(shift)]

# Crypto to Equity
rollingwindow_count_ctoe = pd.DataFrame(0, columns = names_equity + ["Total"], index = names_crypto + ["Total"])
#5h
#NIKKEI
index_values1 = granger_rw_5h_ctoe.iloc[::6]
index_values2 = granger_rw_5h_ctoe.iloc[1::6]
index_values3 = granger_rw_5h_ctoe.iloc[2::6]
index_values4 = granger_rw_5h_ctoe.iloc[3::6]
index_values5 = granger_rw_5h_ctoe.iloc[4::6]
index_values6 = granger_rw_5h_ctoe.iloc[5::6]
count_NIKKEI = (granger_rw_5h_ctoe["C1"] < 0.05).sum() / 6 * len(index_values1)
rollingwindow_count_ctoe.iloc[6, 1] = count_NIKKEI
rollingwindow_count_ctoe.iloc[0, 1] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 1] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 1] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 1] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 1] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 1] = 100 * (index_values6 < 0.05).sum() / len(index_values1)

# 15h
#DAX
index_values1 = granger_rw_15h_ctoe.iloc[::12]
index_values2 = granger_rw_15h_ctoe.iloc[1::12]
index_values3 = granger_rw_15h_ctoe.iloc[2::12]
index_values4 = granger_rw_15h_ctoe.iloc[3::12]
index_values5 = granger_rw_15h_ctoe.iloc[4::12]
index_values6 = granger_rw_15h_ctoe.iloc[5::12]
rollingwindow_count_ctoe.iloc[0, 7] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 7] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 7] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 7] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 7] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 7] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 7] = rollingwindow_count_ctoe.iloc[0:6, 7].sum() / 6
#FTSE 100
index_values1 = granger_rw_15h_ctoe.iloc[6::12]
index_values2 = granger_rw_15h_ctoe.iloc[7::12]
index_values3 = granger_rw_15h_ctoe.iloc[8::12]
index_values4 = granger_rw_15h_ctoe.iloc[9::12]
index_values5 = granger_rw_15h_ctoe.iloc[10::12]
index_values6 = granger_rw_15h_ctoe.iloc[11::12]
rollingwindow_count_ctoe.iloc[0, 8] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 8] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 8] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 8] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 8] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 8] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 8] = rollingwindow_count_ctoe.iloc[0:6, 8].sum() / 6

#17h
#FTSE MIB
index_values1 = granger_rw_17h_ctoe.iloc[::12]
index_values2 = granger_rw_17h_ctoe.iloc[1::12]
index_values3 = granger_rw_17h_ctoe.iloc[2::12]
index_values4 = granger_rw_17h_ctoe.iloc[3::12]
index_values5 = granger_rw_17h_ctoe.iloc[4::12]
index_values6 = granger_rw_17h_ctoe.iloc[5::12]
rollingwindow_count_ctoe.iloc[0, 3] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 3] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 3] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 3] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 3] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 3] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 3] =rollingwindow_count_ctoe.iloc[0:6, 3].sum() / 6
#CAC 40
index_values1 = granger_rw_17h_ctoe.iloc[6::12]
index_values2 = granger_rw_17h_ctoe.iloc[7::12]
index_values3 = granger_rw_17h_ctoe.iloc[8::12]
index_values4 = granger_rw_17h_ctoe.iloc[9::12]
index_values5 = granger_rw_17h_ctoe.iloc[10::12]
index_values6 = granger_rw_17h_ctoe.iloc[11:12]
rollingwindow_count_ctoe.iloc[0, 6] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 6] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 6] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 6] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 6] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 6] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 6] = rollingwindow_count_ctoe.iloc[0:6, 6].sum() / 6

#20h
#S&P 500
index_values1 = granger_rw_20h_ctoe.iloc[::24]
index_values2 = granger_rw_20h_ctoe.iloc[1::24]
index_values3 = granger_rw_20h_ctoe.iloc[2::24]
index_values4 = granger_rw_20h_ctoe.iloc[3::24]
index_values5 = granger_rw_20h_ctoe.iloc[4::24]
index_values6 = granger_rw_20h_ctoe.iloc[5::24]
rollingwindow_count_ctoe.iloc[0, 0] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 0] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 0] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 0] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 0] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 0] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 0] = rollingwindow_count_ctoe.iloc[0:6, 0].sum() / 6
#RUSSELL 2000
index_values1 = granger_rw_20h_ctoe.iloc[6::24]
index_values2 = granger_rw_20h_ctoe.iloc[7::24]
index_values3 = granger_rw_20h_ctoe.iloc[8::24]
index_values4 = granger_rw_20h_ctoe.iloc[9::24]
index_values5 = granger_rw_20h_ctoe.iloc[10::24]
index_values6 = granger_rw_20h_ctoe.iloc[11:24]
rollingwindow_count_ctoe.iloc[0, 2] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 2] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 2] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 2] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 2] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 2] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 2] = rollingwindow_count_ctoe.iloc[0:6,2].sum() / 6
#NASDAQ
index_values1 = granger_rw_20h_ctoe.iloc[12::24]
index_values2 = granger_rw_20h_ctoe.iloc[13::24]
index_values3 = granger_rw_20h_ctoe.iloc[14::24]
index_values4 = granger_rw_20h_ctoe.iloc[15::24]
index_values5 = granger_rw_20h_ctoe.iloc[16::24]
index_values6 = granger_rw_20h_ctoe.iloc[17::24]
rollingwindow_count_ctoe.iloc[0, 4] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 4] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 4] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 4] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 4] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 4] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 4] = rollingwindow_count_ctoe.iloc[0:6, 4].sum() / 6
#Dow Jones
index_values1 = granger_rw_20h_ctoe.iloc[18::24]
index_values2 = granger_rw_20h_ctoe.iloc[19::24]
index_values3 = granger_rw_20h_ctoe.iloc[20::24]
index_values4 = granger_rw_20h_ctoe.iloc[21::24]
index_values5 = granger_rw_20h_ctoe.iloc[22::24]
index_values6 = granger_rw_20h_ctoe.iloc[23::24]
rollingwindow_count_ctoe.iloc[0, 5] = 100 * (index_values1 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[1, 5] = 100 * (index_values2 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[2, 5] = 100 * (index_values3 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[3, 5] = 100 * (index_values4 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[4, 5] = 100 * (index_values5 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[5, 5] = 100 * (index_values6 < 0.05).sum() / len(index_values1)
rollingwindow_count_ctoe.iloc[6, 5] = rollingwindow_count_ctoe.iloc[0:6, 5].sum() / 6
#
rollingwindow_count_ctoe.iloc[0, 9] = rollingwindow_count_ctoe.iloc[0, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[1, 9] = rollingwindow_count_ctoe.iloc[1, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[2, 9] = rollingwindow_count_ctoe.iloc[2, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[3, 9] = rollingwindow_count_ctoe.iloc[3, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[4, 9] = rollingwindow_count_ctoe.iloc[4, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[5, 9] = rollingwindow_count_ctoe.iloc[5, 0:9].sum() / 9
rollingwindow_count_ctoe.iloc[6, 9] = rollingwindow_count_ctoe.iloc[6, 0:9].sum() / 9

latex_code = rollingwindow_count_ctoe.to_latex()
with open('rollingwindow_count_ctoe.tex', 'w') as file:
    file.write(latex_code)
#%%
# Plotting
plt.figure(figsize = (12, 6))
plt.plot(granger_equity_to_crypto_rw.index, granger_equity_to_crypto_rw['Ratio'], label = 'Equity to Crypto', color = 'blue', marker = 'o')
plt.plot(granger_crypto_to_equity_rw.index, granger_crypto_to_equity_rw['Ratio'], label = 'Crypto to Equity', color = 'red', marker = 'x')
# Add custom lines
plt.axhline(y = 0.10, color = 'green', linestyle = '--', label = 'y=0.10 full range')
plt.axhline(y = 0.2, color = 'orange', linestyle = '--', label = 'y=0.2 (2019-2020)')
plt.axhline(y = 0.25, color = 'purple', linestyle = '--', label = 'y=0.25 (2021-Apr 2024)')

# Define date ranges for custom lines
start_2019 = pd.to_datetime('2019-01-01')
end_2020 = pd.to_datetime('2020-12-31')
start_2021 = pd.to_datetime('2021-01-01')
end_2024 = pd.to_datetime('2024-04-30')

plt.plot([start_2019, end_2020], [0.2, 0.2], color = 'orange', linestyle = '--')
plt.plot([start_2021, end_2024], [0.25, 0.25], color = 'purple', linestyle = '--')

plt.ylabel('Granger Causality Networks ratio')
plt.title('Granger Causality Netowrks ratio evolution')
plt.legend()
plt.grid(True)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig('Granger Causality ratiorw60sh20p6.png')
plt.show()

#%%
# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_21_24_ctoe.index, bipartite = 0)
G.add_nodes_from(df_gc_21_24_ctoe.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_21_24_ctoe.index:
    for col in df_gc_21_24_ctoe.columns:
        weight = df_gc_21_24_ctoe.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_21_24_ctoe.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Crypto to Equity (2022 - 2024)')
plt.savefig('Granger Causality Bipartite Network - Crypto to Equity (2022 - 2024).png')
plt.show()

# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_21_24_etoc.index, bipartite = 0)
G.add_nodes_from(df_gc_21_24_etoc.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_21_24_etoc.index:
    for col in df_gc_21_24_etoc.columns:
        weight = df_gc_21_24_etoc.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_21_24_etoc.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Equity to Crypto (2022 - 2024)')
plt.savefig('Granger Causality Bipartite Network - Equity to Crypto (2022 - 2024).png')
plt.show()

# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_cryptotoequity.index, bipartite = 0)
G.add_nodes_from(df_gc_cryptotoequity.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_cryptotoequity.index:
    for col in df_gc_cryptotoequity.columns:
        weight = df_gc_cryptotoequity.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_cryptotoequity.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Crypto to Equity (2019 - 2024)')
plt.savefig('Granger Causality Bipartite Network - Crypto to Equity (2019 - 2024).png')
plt.show()

# Create a bipartite graph
G = nx.Graph()
G.add_nodes_from(df_gc_equitytocrypto.index, bipartite = 0)
G.add_nodes_from(df_gc_equitytocrypto.columns, bipartite = 1)
# Add nodes and edges to the graph
for row in df_gc_equitytocrypto.index:
    for col in df_gc_equitytocrypto.columns:
        weight = df_gc_equitytocrypto.loc[row, col]
        if weight > 0.05:
            G.add_edge(row, col, weight = weight)

# Define the layout (circular or oval)
pos = nx.bipartite_layout(G, df_gc_equitytocrypto.index)
# Draw the graph
plt.figure(figsize = (12, 8))
nx.draw(G, pos, with_labels = True, node_size = 700, node_color = 'lightblue', font_size = 12, edge_color = 'gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = {k: f'{v:.4f}' for k, v in labels.items()})
# Display the graph
plt.title('Granger Causality Bipartite Network - Equity to Crypto (2019 - 2024)')
plt.savefig('Granger Causality Bipartite Network - Equity to Crypto (2019 - 2024).png')
plt.show()
#%%