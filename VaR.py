"""Calculating given VaR for all tickers in FTSE100."""

import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

df = pd.read_excel('FTSE100.xlsx', index_col=0, parse_dates=True, skiprows=[0])

# Calculate daily returns
returns = df.pct_change().dropna()
returns.head()

# Calculate mean and standard deviation 
mean = np.mean(returns['Closing Price'])
stdev = np.std(returns['Closing Price'])

rolling_std = returns.rolling(21).std()
sigten = np.sqrt(10*rolling_std**2)
muten = mean*10

var = muten - sigten*2.33

plt.figure()
plt.plot(var)
plt.title("VaR with Rolling 21 Day Std. Dev.")
plt.xlabel("Date")
plt.ylabel("VaR")
plt.show()

def garch(ret, omega=0.000001, alpha=0.047, beta=0.9466):
    
    var = []
    for i in range(len(ret)):
        if i==0:
            var.append(stdev**2)
        else:
            var.append(omega + alpha * ret.iloc[i-1,0]**2 + beta * var[i-1])
            
    return np.array(var)

variance = garch(ret=returns)
var_garch = muten - (np.sqrt(10*variance))*2.33

plt.figure()
plt.plot(var_garch)
plt.title("GARCH VaR")
plt.xlabel("Date")
plt.ylabel("VaR")
plt.show()

tenday = np.log(df) - np.log(df.shift(10))
tenday = tenday.drop(tenday.index[:21], axis=0)

var = var.drop(var.index[:20], axis=0)
var_garch = pd.DataFrame(var_garch,columns=['Closing Price'])
var_garch = var_garch.drop(var_garch.index[:20], axis=0)

# VaR to 10 day log ret
tenday = tenday.drop(tenday.index[:10], axis=0)
var = var.drop(var.index[-10:], axis=0)
var_garch = var_garch.drop(var_garch.index[-10:], axis=0)

breaches = pd.DataFrame()
breaches['VaR'] = var
breaches['Garch VaR'] = var_garch

var_temp = var.reset_index(drop=True)
var_garch_temp = var_garch.reset_index(drop=True)
tenday_temp = tenday.reset_index(drop=True)

var_breach = pd.DataFrame(var_temp>tenday_temp)
garch_var_breach = pd.DataFrame(var_garch_temp>tenday_temp)

print(len(var_breach[var_breach['Closing Price']]))
print(len(garch_var_breach[garch_var_breach['Closing Price']]))


fig, ax = plt.subplots()
ax.plot(var_temp, 'b')
ax.plot(var_garch_temp, 'r')
ax.plot(tenday_temp, 'k-')
ax.legend(['VaR', 'GARCH VaR', '10 Day Mean'])
plt.title("Comparison of VaR, GARCH VaR and 10 Day Mean Returns")
ax.set_xlabel("Days")
ax.set_ylabel("VaR/Mean Returns")
plt.show()
