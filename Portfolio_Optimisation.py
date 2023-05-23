"""Optimising portfolio for assets with given correlations, and price mean and volatility."""

# Import dependencies
import numpy as np
from numpy.linalg import multi_dot
import pandas as pd

import cufflinks as cf
cf.set_config_file(offline=True, dimensions=((1000,600)))

import plotly.express as px
px.defaults.template, px.defaults.width, px.defaults.height = "plotly_white", 1000, 600

import warnings
warnings.filterwarnings('ignore')


R = np.array([
    [1, 0.2, 0.5, 0.3],
    [0.2, 1, 0.7, 0.4],
    [0.5, 0.7, 1, 0.9],
    [0.3 ,0.4, 0.9, 1]
])

investment_universe = {
    "A": {
        "mu": 0.04,
        "sigma": 0.07
    },
    "B": {
        "mu": 0.08,
        "sigma": 0.12
    },
    "C": {
        "mu": 0.12,
        "sigma": 0.18
    },
    "D": {
        "mu": 0.15,
        "sigma": 0.26
    }
}

asset_rets = [[investment_universe[asset]['mu']] for asset in investment_universe]
asset_vols = [[investment_universe[asset]['sigma']] for asset in investment_universe]

# Question 1
number_of_assets = len(investment_universe)

starting_weights = np.array(number_of_assets * [1./number_of_assets])[:, np.newaxis]
one = np.ones(number_of_assets)


S = np.diagflat(asset_vols)
cov_matrix = multi_dot([S, R, S])

def lagrange_multipliers(cov_matrix, asset_rets):
    cov_matrix_i = np.linalg.inv(cov_matrix)
    asset_rets_t = np.transpose(asset_rets)

    A = multi_dot([one, cov_matrix_i, one])
    B = multi_dot([asset_rets_t, cov_matrix_i, one])
    C = multi_dot([asset_rets_t, cov_matrix_i, asset_rets])
    return A, B, C

def lambda_and_gamma(m):
    multipliers = lagrange_multipliers(cov_matrix, asset_rets)
    A = multipliers[0]
    B = multipliers[1]
    C = multipliers[2]

    lmda = (A*m - B)/(A*C - B**2)
    gmma = (C - B*m)/(A*C - B**2)
    return lmda, gmma

def optimal_allocation(cov_matrix, asset_rets, m):
    cov_matrix_i = np.linalg.inv(cov_matrix)

    greeks = lambda_and_gamma(m)
    lmda = greeks[0]
    gmma = greeks[1][0]

    dot_sum = lmda*np.array(asset_rets) + gmma*one
    optimal_weights = multi_dot([cov_matrix_i, dot_sum])
    return [_w[0] for _w in optimal_weights]

def portfolio_simulation(asset_rets, cov_matrix, m, optimal = True):

    # Initialize the lists
    rets = []
    vols = []
    wts = []

    # Simulate 5,000 portfolios
    for _ in range(number_of_portfolios):
        # Generate random weights
        if optimal:
            weights = optimal_allocation(cov_matrix, asset_rets, m)
        else:
            weights = np.random.random(number_of_assets)[:, np.newaxis]
        weights /= sum(weights)
        # Portfolio statistics
        rets.append((multi_dot([np.transpose(weights), asset_rets])*252)[:, np.newaxis])
        vols.append(np.sqrt(multi_dot([weights.T, cov_matrix, weights]))*252)
        wts.append(weights.flatten())

    if optimal:
        return wts[0], np.sqrt(vols[0])
    # Create a dataframe for analysis
    portdf = pd.DataFrame({
        'port_rets': np.array(rets).flatten(),
        'port_vols': np.array(vols).flatten(),
        'weights': list(np.array(wts))
    })

    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']

    return round(portdf,2)


def min_var_portfolio(asset_rets, cov_matrix, optimal=True):
    cov_matrix_i = np.linalg.inv(cov_matrix) #

    multipliers = lagrange_multipliers(cov_matrix, asset_rets) #
    A = multipliers[0]
    B = multipliers[1]

    rets = []
    vols = []
    wts = []

    for _ in range(number_of_portfolios):
        if optimal:
            w = multi_dot([cov_matrix_i, one])/A
            w /= sum(w)
            rets.append(B/A)
            vols.append(1/A)
        else:
            w = np.random.random(number_of_assets)[:, np.newaxis]
            w /= sum(w)
            rets.append((multi_dot([np.transpose(w), asset_rets])*252)[:, np.newaxis])
            vols.append(np.sqrt(multi_dot([np.transpose(w), cov_matrix, w]))*252)
        wts.append(w.flatten())

    if optimal:
        return wts[0], np.sqrt(vols[0])
    # Create a dataframe for analysis
    portdf = pd.DataFrame({
        'port_rets': np.array(rets).flatten(),
        'port_vols': np.array(vols).flatten(),
        'weights': list(np.array(wts))
    })

    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']

    return round(portdf,2)



number_of_portfolios = 50000
asset_rets = [[investment_universe[asset]['mu']] for asset in investment_universe]

temp = portfolio_simulation(asset_rets=asset_rets, cov_matrix=cov_matrix, m=0.1, optimal=False)

fig = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title="Monte Carlo Simulated Portfolio"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))

# Plot max sharpe
fig.add_scatter(
    mode='markers', 
    x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
    y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
    marker=dict(color='Red', size=20, symbol='star'),
    name = 'Max Sharpe'
).update(layout_showlegend=False)

# Show spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


temp = min_var_portfolio(asset_rets=asset_rets, cov_matrix=cov_matrix, optimal = False)

fig = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title="Monte Carlo Simulated Min-Variance Portfolio"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))

# Plot max sharpe
fig.add_scatter(
    mode='markers', 
    x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
    y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
    marker=dict(color='Red', size=20, symbol='star'),
    name = 'Max Sharpe'
).update(layout_showlegend=False)

# Show spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()

port_opt = portfolio_simulation(asset_rets=asset_rets, cov_matrix=cov_matrix, m=0.1, optimal =True)
print(port_opt[0], port_opt[1])
minvar_opt = min_var_portfolio(asset_rets=asset_rets, cov_matrix=cov_matrix, optimal = True)
print(minvar_opt[0], minvar_opt[1])
