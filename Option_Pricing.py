"""Performing option pricing under binomial method."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import warnings
warnings.filterwarnings('ignore')

def binomial_option(spot, strike, rate, sigma, time, steps, output=0):
    
    """
    binomial_option(spot, strike, rate, sigma, time, steps, output=0)
    
    Function for building binomial option tree for european call option payoff. 
    
    Parameters
    ----------
    spot        int or float   - spot price
    strike      int or float   - strike price 
    rate        float          - interest rate
    sigma       float          - volatility
    time        int or float   - expiration time
    steps       int            - number of time steps
    output      int            - [0: price, 1: payoff, 2: option value, 3: option delta]
    
    Returns
    ----------
    out : ndarray
    An array object of price, payoff, option value and delta as specified by the output parameter

    """
    
    # define parameters
    ts = time/steps                                 # ts is time steps, dt
    u = 1 + sigma*np.sqrt(ts)                          # u is up factor
    v = 1 - sigma*np.sqrt(ts)                          # v is down factor
    p = 0.5 + rate*np.sqrt(ts)/(2*sigma)               # p here is risk neutral probability (p') - for ease of use
    df = 1/(1+rate*ts)                              # df is discount factor

    # initialize arrays
    px = np.zeros((steps+1, steps+1))                  # price path
    cp = np.zeros((steps+1, steps+1))                  # call intrinsic payoff
    V = np.zeros((steps+1, steps+1))                   # option value
    d = np.zeros((steps+1, steps+1))                   # delta value
    
    # binomial loop
    for j in range(steps+1):
        for i in range(j+1):
            px[i,j] = spot * np.power(v,i) * np.power(u,j-i)
            cp[i,j] = np.maximum(px[i,j] - strike,0)
            
    for j in range(steps+1, 0, -1):
        for i in range(j):
            if (j == steps+1):
                V[i,j-1] = cp[i,j-1]                # terminal payoff
                d[i,j-1] = 0                        # terminal delta
            else:
                V[i,j-1] = df*(p*V[i,j]+(1-p)*V[i+1,j])
                d[i,j-1] = (V[i,j]-V[i+1,j])/(px[i,j]-px[i+1,j])
                    
    results = np.around(px,2), np.around(cp,2), np.around(V,2), np.around(d,4)
    
    return results[output]

option_value = []
vols = []
for i in range(101):
    option_value += [binomial_option(spot=100, strike=100, rate=0.05, sigma=i/100, time=1, steps=4, output=2)[0][0]]
    vols += [i/100]

plt.figure()
plt.plot(vols,option_value)
plt.title("Option Value as Volatility Increases")
plt.xlabel("Volatility")
plt.ylabel("Option Value")
plt.show()


upper_range = 300
ts_option_value = [binomial_option(spot=100, strike=100, rate=0.05, sigma=0.2, time=1, steps=i, output=2)[0][0] for i in range(4, upper_range+1)]
plt.figure()
plt.plot(np.linspace(4,upper_range, upper_range-3) ,ts_option_value)
plt.title("Option Value as Time Steps Increase")
plt.xlabel("Time Steps")
plt.ylabel("Option Value")
plt.show()
