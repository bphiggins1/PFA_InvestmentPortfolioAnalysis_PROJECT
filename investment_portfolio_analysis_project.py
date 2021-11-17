# -*- coding: utf-8 -*-
"""
Investment Analysis Project

1. Using five years of historical data from Yahoo! Finance, (ended 6/30/21),
please construct and analyze the performance of a portfolio that fits the
following description:
   a. Begins with the following securities: TLT, VTI, VTV, VBR, AGG
   b. Rebalances the portfolio to equal weight each security in the
      portfolio at the beginning of each month
   c. Reinvests dividends into the securities that paid them
   d. Midway during the time period, (12/31/19), the portfolio replaces
      AGG with SHV during an otherwise normal rebalance
   
2. Using this constructed track record, please calculate the following
performance metrics for the portfolio:
   a. Cumulative Return
   b. Sharpe Ratio (vs. a RF of your choice)
   c. Information Ratio (vs. a benchmark of your choice)
   d. Sortino Ratio (vs. a benchmark of your choice)
   e. CAPM Beta (vs. a benchmark of your choice)
   
In your deliverable, please provide:
   1. The metrics both since inception and over rolling six-month timeframes.
   2. An explanation of how each metric can and can't be interpreted, in
      this context.
   3. A short explanation of why you chose the benchmark(s).
   4. Any other analysis/visualization of the portfolio you find valuable.
      This is your opportunity to impress us!
      
      Examples below:
      a. Turnover analysis
      b. Style analysis
      c. Macro exposure analysis

"""

# import libraries needed for program
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np


###################### PART 1 ######################
# Initializes begin and end date covering 5 years 
# beginning 6/30/2016 and ending 6/30/21
beg = datetime(2016,6,30)
end = datetime(2021,6,30)
# Universe of securities for this project
# Note: ^GSPC is the S&P 500 Index on Yahoo Finance - portfolio benchmark
univ = ['TLT', 'VTI', 'VTV', 'VBR', 'AGG', 'SHV']
benchmark = ['^GSPC']
# Portfolio of 5 securities are equally weighted at 20%
weights = 0.20

###################### PART 1a ######################
# Retrieves Adjusted Closing Price for the universe of securities
# from Yahoo! Finance into DataFrame1 from start date to end date.
# Does the same for Benchmark DataFrame1.
###################### PART 1c ######################
# Note: The adjusted closing price amends a stock's closing price
#       to reflect that stock's value after accounting for corporate
#       actions such as stock splits, dividends, and rights offerings.
df1 = yf.download(univ, start=beg, end=end)['Adj Close']
bm1 = pd.DataFrame(yf.download(benchmark, start=beg, end=end)['Adj Close'])

###################### PART 1b, 1d ##################
# Creates DataFrame2 by computing the percent change between each current
# and prior element of DataFrame1. Drops all DataFrame 2 elements with
# missing values. If all values are NA, drop that row or column.
# Does the same for Benchmark DataFrame2.
df2 = df1.pct_change(1).dropna(how='all')
bm2 = bm1.pct_change(1).dropna(how='all')

# Creates a new Month index for DataFrame2 which is the Month End date
# for each Date index. Does the same for Benchmark DataFrame2.
df2 = df2.set_index(df2.index.rename('Month') + pd.tseries.offsets.MonthEnd(1), append=True)
bm2 = pd.DataFrame(bm2.set_index(bm2.index.rename('Month') + pd.tseries.offsets.MonthEnd(1), append=True))

# Computes monthly returns for each security by adding 1 to DataFrame2
# elements (percent change), grouping elements by month, multiplying monthly
# groups together and subtracting 1. Does the same for Benchmark DataFrame2.
df2 = df2.add(1).groupby('Month').apply(np.prod).subtract(1)
bm2 = bm2.add(1).groupby('Month').apply(np.prod).subtract(1)

# Creates DataFame3 by making a copy of DataFrame2 and assigns the 20%
# weights to each element. Not necessary for Benchmark DataFrame since
# it is only 1 security
df3 = df2.copy()
df3.loc[:,:] = weights

# [PART 1d] Midway during the time period, (12/31/19), the portfolio replaces
# AGG with SHV during an otherwise normal rebalance. Weights for SHV are
# set to 0 through the end of 2019. They will be 20% after. Weights for
# AGG are set to 0 after 2019. They will be 20% before.
df3.loc[df3.index.year <= 2019, 'SHV'] = 0
df3.loc[df3.index.year  > 2019, 'AGG'] = 0

# [PART 1b] Creates DataFrame4 by computing portfolio returns by multiplying
# DataFrame2 return by DataFrame3 weights and summing by month.
# Monthly rebalance is achieved by muliplying by monthly return by weight.
# Benchmark DataFrame4 is created holding SP500 monthly returns
df4 = df2.multiply(df3).sum(axis=1).rename('Portfolio Monthly Return').to_frame()
bm4 = bm2.sum(axis=1).rename('SP500 Monthly Return').to_frame()

################ PLOTS #######################
# Monthly Returns for Tickers and SP500 Index
ticker_mthy_returns = df2.multiply(df3)
all_mthy_returns = pd.merge(ticker_mthy_returns, bm2.sum(axis=1).rename('^GSPC').to_frame(), on="Month")
all_mthy_returns.plot(figsize=(20,10), title="Ticker Monthly Returns")

# Portfolio Monthly Returns vs SP500 Index Returns
portf_sp500_returns = pd.merge(df4['Portfolio Monthly Return'], bm4['SP500 Monthly Return'], on="Month")
portf_sp500_returns.plot(figsize=(20,10), title="Portfolio vs SP500 Monthly Returns")

################################ PART 2 ################################

###################### PART 2a ######################
# Computes Cumulative Return by adding 1 to Monthly Returns, taking cumulative
# product, and subtracting 1
df4['Cumulative Portfolio Monthly Return'] = df4['Portfolio Monthly Return'].add(1).cumprod().subtract(1)
bm4['Cumulative SP500 Monthly Return'] = bm4['SP500 Monthly Return'].add(1).cumprod().subtract(1)
print("2a. The Cumulative Portfolio Return is", round(df4['Cumulative Portfolio Monthly Return'].iloc[-1], 4))
print("    The Cumulative SP500 Return is", round(bm4['Cumulative SP500 Monthly Return'].iloc[-1], 4))

# Compute 6M rolling Cumulative Return
df4['Cumulative Portfolio Return 6M'] = df4['Portfolio Monthly Return'].add(1).rolling(window=6).agg(lambda x : x.prod()) -1
bm4['Cumulative SP500 Return 6M'] = bm4['SP500 Monthly Return'].add(1).rolling(window=6).agg(lambda x : x.prod()) -1

################ PLOTS #######################
# 6M Rolling Monthly Returns for Tickers and SP500 Index
portf_sp500_6M_returns = pd.merge(df4['Cumulative Portfolio Return 6M'], bm4['Cumulative SP500 Return 6M'], on="Month")
portf_sp500_6M_returns.plot(figsize=(20,10), title="Portfolio vs SP500 6M Rolling Cumulative Monthly Returns")


###################### PART 2b ######################
# Computes Sharpe Ratio (vs. a RF of your choice)
# 
# Since the calculation uses monthly returns, then the risk-free rate
# should also be on instruments that are risk-free at the one-month horizon,
# the 30 day T-Bills which is 0.05% on 2021-6-30 (https://ycharts.com/indicators/1_month_treasury_rate)
risk_free_rate = 0.005
Sharpe_Ratio = (df4['Portfolio Monthly Return'].mean() - risk_free_rate) / df4['Portfolio Monthly Return'].std()
df4['Portfolio Sharpe Ratio'] = Sharpe_Ratio
SP500_Sharpe_Ratio = (bm4['SP500 Monthly Return'].mean() - risk_free_rate) / bm4['SP500 Monthly Return'].std()
df4['SP500 Sharpe Ratio'] = SP500_Sharpe_Ratio
print("2b. The Portfolio Sharpe Ratio is", round(Sharpe_Ratio, 4))
print("    The SP500 Sharpe Ratio is", round(SP500_Sharpe_Ratio, 4))

# Compute 6M rolling Sharpe Ratio
df4['Portfolio Sharpe Ratio 6M'] = df4['Portfolio Monthly Return'].rolling(window=6).apply(lambda x: (x.mean() - risk_free_rate) / x.std(), raw = True)
bm4['SP500 Sharpe Ratio 6M'] = bm4['SP500 Monthly Return'].rolling(window=6).apply(lambda x: (x.mean() - risk_free_rate) / x.std(), raw = True)

################ PLOTS #######################
# 6M Rolling Sharpe Ratio for Tickers and SP500 Index
portf_sp500_6M_sharpe = pd.merge(df4['Portfolio Sharpe Ratio 6M'], bm4['SP500 Sharpe Ratio 6M'], on="Month")
portf_sp500_6M_sharpe.plot(figsize=(20,10), title="Portfolio vs SP500 6M Rolling Sharpe Ratio")


###################### PART 2c ######################
# Computes Information Ratio vs. SP500 benchmark
# 
Information_Ratio = df4['Portfolio Monthly Return'].subtract(bm4['SP500 Monthly Return']).mean() / df4['Portfolio Monthly Return'].subtract(bm4['SP500 Monthly Return']).std()
df4['Portfolio Information Ratio'] = Information_Ratio 
print("2c. The Information Ratio is", round(Information_Ratio, 4))

# Compute 6M rolling Information Ratio
Info_Num_6M = (df4['Portfolio Monthly Return'].subtract(bm4['SP500 Monthly Return'])).rolling(window=6).mean()
Info_Denom_6M = (df4['Portfolio Monthly Return'].subtract(bm4['SP500 Monthly Return'])).rolling(window=6).std()
df4['Portfolio Information Ratio 6M'] = Info_Num_6M.div(Info_Denom_6M)

################ PLOTS #######################
# 6M Rolling Information Ratio
df4['Portfolio Information Ratio 6M'].plot(figsize=(20,10), title="Portfolio 6M Rolling Information Ratio")

###################### PART 2d ######################
# Computes Sortino Ratio vs. SP500 benchmark
#
Sortino_Ratio = df4['Portfolio Monthly Return'].mean() - risk_free_rate / (df4['Portfolio Monthly Return'] < 0).std()
df4['Portfolio Sortino Ratio'] = Sortino_Ratio
SP500_Sortino_Ratio = bm4['SP500 Monthly Return'].mean() - risk_free_rate / (bm4['SP500 Monthly Return'] < 0).std()
bm4['SP500 Sortino Ratio'] = SP500_Sortino_Ratio
print("2d. The Portfolio Sortino Ratio is", round(Sortino_Ratio, 4))
print("    The SP500 Sortino Ratio is", round(SP500_Sortino_Ratio, 4))

# Compute 6M rolling Sortino Ratio
Sortino_Num_6M = df4['Portfolio Monthly Return'].rolling(window=6).mean().subtract(risk_free_rate)
Sortino_Denom_6M = (df4['Portfolio Monthly Return'] < 0).rolling(window=6).std()
df4['Portfolio Sortino Ratio 6M'] = Sortino_Num_6M.div(Sortino_Denom_6M)
SP500_Sortino_Num_6M = bm4['SP500 Monthly Return'].rolling(window=6).mean().subtract(risk_free_rate)
SP500_Sortino_Denom_6M = (bm4['SP500 Monthly Return'] < 0).rolling(window=6).std()
bm4['SP500 Sortino Ratio 6M'] = SP500_Sortino_Num_6M.div(SP500_Sortino_Denom_6M)

################ PLOTS #######################
# 6M Rolling Sortino Ratio for Tickers and SP500 Index
df4['Portfolio Sortino Ratio 6M'].plot(figsize=(20,10), title="Portfolio 6M Rolling Sortino Ratio")
bm4['SP500 Sortino Ratio 6M'].plot(figsize=(20,10), title="SP500 6M Rolling Sortino Ratio")
portf_sp500_6M_sortino = pd.merge(df4['Portfolio Sortino Ratio 6M'], bm4['SP500 Sortino Ratio 6M'], on="Month")
portf_sp500_6M_sortino.plot(figsize=(20,10), title="Portfolio vs SP500 6M Rolling Sortino Ratio")

###################### PART 2e ######################
# Computes CAPM Beta vs. SP500 benchmark
# 
CAPM_Beta_Covariance = df4['Portfolio Monthly Return'].cov(bm4['SP500 Monthly Return'])
CAPM_Beta_Variance = df4['Portfolio Monthly Return'].var()
CAPM_Beta = CAPM_Beta_Covariance / CAPM_Beta_Variance
df4['CAPM Beta'] = CAPM_Beta_Covariance / CAPM_Beta_Variance
print("2e. The CAPM Beta is", round(CAPM_Beta, 4))

# Compute 6M rolling CAPM Beta
CAPM_Beta_Covariance_6M = df4['Portfolio Monthly Return'].rolling(window=6).cov(bm4['SP500 Monthly Return'])
CAPM_Beta_Variance_6M = df4['Portfolio Monthly Return'].rolling(window=6).var()
df4['CAPM Beta 6M'] = CAPM_Beta_Covariance_6M / CAPM_Beta_Variance_6M

################ PLOTS #######################
# 6M Rolling Information Ratio
df4['CAPM Beta 6M'].plot(figsize=(20,10), title="Portfolio 6M Rolling CAPM Beta")


#############################################################################
############### EXTRA ANALYSIS ##############################################


#############################################################################
# EXTRA 1 - Portfolio Drawdown
df4['Portfolio Current Drawdown'] = df4['Cumulative Portfolio Monthly Return'].add(1) / np.maximum.accumulate(df4['Cumulative Portfolio Monthly Return'].add(1), axis=0) - 1
df4['Portfolio Max Drawdown'] = np.minimum.accumulate(df4['Portfolio Current Drawdown'] , axis=0)
bm4['SP500 Current Drawdown'] = bm4['Cumulative SP500 Monthly Return'].add(1) / np.maximum.accumulate(bm4['Cumulative SP500 Monthly Return'].add(1), axis=0) - 1
bm4['SP500 Max Drawdown'] = np.minimum.accumulate(bm4['SP500 Current Drawdown'] , axis=0)
print("EXTRA. The Portfolio Max Drawdown is", round(df4['Portfolio Max Drawdown'].iloc[-1], 4))
print("       The SP500 Max Drawdown is", round(bm4['SP500 Max Drawdown'].iloc[-1], 4))


portf_sp500_6M_drawdown = pd.merge(df4['Portfolio Current Drawdown'], bm4['SP500 Current Drawdown'], on="Month")
portf_sp500_6M_drawdown.plot(figsize=(20,10), title="Portfolio vs SP500 Current Drawdown")

#############################################################################
# EXTRA 2 - Portfolio Ticker Comparison vs Benchmark

all_mthy_returns_bef2020 = all_mthy_returns[df3.index.year <= 2019]
del all_mthy_returns_bef2020['SHV']
all_mthy_returns_from2020 = all_mthy_returns[df3.index.year > 2019]
del all_mthy_returns_from2020['AGG']

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

N = 12 #12 months in a year
rf = risk_free_rate
sharpes = all_mthy_returns.apply(sharpe_ratio, args=(N,rf,),axis=0)
sharpes.plot.bar(title="Ticker Annualized Sharpe Ratio Comparison")
sharpes_bef2020 = all_mthy_returns_bef2020.apply(sharpe_ratio, args=(N,rf,),axis=0)
sharpes_bef2020.plot.bar(title="(Before 2020) Ticker Annualized Sharpe Ratio Comparison")
sharpes_from2020 = all_mthy_returns_from2020.apply(sharpe_ratio, args=(N,rf,),axis=0)
sharpes_from2020.plot.bar(title="(From 2020) Ticker Annualized Sharpe Ratio Comparison")

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg

sortinos = all_mthy_returns.apply(sortino_ratio, args=(N,rf,), axis=0 )
sortinos.plot.bar(title="Ticker Annualized Sortino Ratio Comparison")
sortinos_bef2020 = all_mthy_returns_bef2020.apply(sortino_ratio, args=(N,rf,), axis=0 )
sortinos_bef2020.plot.bar(title="(Before 2020) Ticker Annualized Sortino Ratio Comparison")
sortinos_from2020 = all_mthy_returns_from2020.apply(sortino_ratio, args=(N,rf,), axis=0 )
sortinos_from2020.plot.bar(title="(From 2020) Ticker Annualized Sortino Ratio Comparison")

def max_drawdown(return_series):
    comp_ret = (return_series+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

max_drawdowns = all_mthy_returns.apply(max_drawdown,axis=0)
max_drawdowns.plot.bar(title="Ticker Max Drawdown Comparison")
max_drawdowns_bef2020 = all_mthy_returns_bef2020.apply(max_drawdown,axis=0)
max_drawdowns_bef2020.plot.bar(title="(Before 2020) Ticker Max Drawdown Comparison")
max_drawdowns_from2020 = all_mthy_returns_from2020.apply(max_drawdown,axis=0)
max_drawdowns_from2020.plot.bar(title="(From 2020) Ticker Max Drawdown Comparison")

calmars = all_mthy_returns.mean()*12/abs(max_drawdowns)
calmars.plot.bar(title="Ticker Annualized Calmar Ratio Comparison")
calmars_bef2020 = all_mthy_returns_bef2020.mean()*12/abs(max_drawdowns_bef2020)
calmars_bef2020.plot.bar(title="(Before 2020) Ticker Annualized Calmar Ratio Comparison")
calmars_from2020 = all_mthy_returns_from2020.mean()*12/abs(max_drawdowns_from2020)
calmars_from2020.plot.bar(title="(From 2020) Ticker Annualized Calmar Ratio Comparison")

# Portfolio Risk vs SP500
all_mthy_returns.plot(kind = "box", figsize = (20, 10), title="Portfolio Risk")
