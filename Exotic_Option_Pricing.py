"""Exotic option (binary, lookback, asian) pricing using Euler-Maruyama, Milstein and Closed Form methods."""

import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(10000)

S0 = 100
E = 100
T = 1
sig = 0.2
r = 0.05

N = 252
dt = T / N
t = np.arange(dt, 1 + dt, dt)


def all_strategies(S0=S0, E=E, mu=r, dt=dt, sigma=sig, N=N):
    asian_payoff = []
    binary_payoff = []
    lookback_max_payoff = []
    lookback_min_payoff = []
    call_payoff = []
    for _ in range(sims):
        X_em = []
        X_mil = []
        X_cf = []
        X = S0
        Y = S0
        Z = S0
        for _ in range(N):
            phi = random.gauss(0, 1)
            X += mu * X * dt + sigma * X * phi * np.sqrt(dt)
            Z = Z * np.exp((mu - 0.5 * sigma**2) * dt + sigma * phi * np.sqrt(dt))
            Y += (
                mu * Y * dt
                + sigma * Y * phi * np.sqrt(dt)
                + 0.5 * sigma**2 * Y * (phi**2 - 1) * dt
            )
            X_em.append(X)
            X_mil.append(Y)
            X_cf.append(Z)

        call_payoff.append(
            [
                np.maximum(X_em[-1] - E, 0),
                np.maximum(X_mil[-1] - E, 0),
                np.maximum(X_cf[-1] - E, 0),
            ]
        )
        asian_payoff.append(
            [
                np.maximum(np.average(X_em) - E, 0),
                np.maximum(np.average(X_mil) - E, 0),
                np.maximum(np.average(X_cf) - E, 0),
            ]
        )
        lookback_max_payoff.append(
            [
                np.max(X_em) - X_em[-1],
                np.max(X_mil) - X_mil[-1],
                np.max(X_cf) - X_cf[-1],
            ]
        )
        lookback_min_payoff.append(
            [
                X_em[-1] - np.min(X_em),
                X_mil[-1] - np.min(X_mil),
                X_cf[-1] - np.min(X_cf),
            ]
        )
        binary_payoff.append([X, Y, Z])

    asian_price = [
        np.average([p[i] for p in asian_payoff]) * np.exp(-r * T) for i in range(3)
    ]
    lookback_max_price = [
        np.average([p[i] for p in lookback_max_payoff]) * np.exp(-r * T)
        for i in range(3)
    ]
    lookback_min_price = [
        np.average([p[i] for p in lookback_min_payoff]) * np.exp(-r * T)
        for i in range(3)
    ]
    count = [
        sum(map(lambda x: x >= E, [p[i] for p in binary_payoff])) for i in range(3)
    ]
    binary_price = [count[i] / sims * np.exp(-r * T) for i in range(3)]
    call_price = [np.average([p[i] for p in call_payoff]) for i in range(3)]

    return asian_price, lookback_max_price, lookback_min_price, binary_price, call_price


sims = 1000

sigmas = np.arange(0, 1, 0.01)
strikes = np.arange(50, 200, 10)
rfr = np.arange(0.01, 0.2, 0.01)
stock = np.arange(0, 200, 10)
expiry = np.arange(1, N * 5, 30)

S0 = 100
E = 100
T = 1
sigma = 0.2
r = 0.05

N = 252
dt = T / N
t = np.arange(dt, 1 + dt, dt)

initial = all_strategies(S0=S0, mu=r, dt=dt, sigma=sigma)

variable_name = "Time to expiry"
variable = expiry

# vary = [all_strategies(S0=S0, mu=r, dt=dt, sigma=sigma) for sigma in sigmas]
# vary = [all_strategies(S0=S0, mu=r, E=strike, dt=dt, sigma=sigma) for strike in strikes]
# vary = [all_strategies(S0=S0, mu=r, E=E, dt=dt, sigma=sigma) for r in rfr]
# vary = [all_strategies(S0=s, mu=r, E=E, dt=dt, sigma=sigma) for s in stock]
vary = [all_strategies(S0=S0, mu=r, E=E, dt=dt, sigma=sigma, N=n) for n in expiry]

### Figures

plt.figure()
plt.plot(variable, [v[0][0] for v in vary], label="Euler-Maruyama")
plt.plot(variable, [v[0][1] for v in vary], label="Milstein")
plt.plot(variable, [v[0][2] for v in vary], label="Closed Form")
plt.legend(loc="upper left")
plt.title(f"Asian Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(variable, [v[1][0] for v in vary], label="Euler-Maruyama")
plt.plot(variable, [v[1][1] for v in vary], label="Milstein")
plt.plot(variable, [v[1][2] for v in vary], label="Closed Form")
plt.legend(loc="upper left")
plt.title(f"Lookback (max) Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(variable, [v[2][0] for v in vary], label="Euler-Maruyama")
plt.plot(variable, [v[2][1] for v in vary], label="Milstein")
plt.plot(variable, [v[2][2] for v in vary], label="Closed Form")
plt.legend(loc="upper left")
plt.title(f"Lookback (min) Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(variable, [v[3][0] for v in vary], label="Euler-Maruyama")
plt.plot(variable, [v[3][1] for v in vary], label="Milstein")
plt.plot(variable, [v[3][2] for v in vary], label="Closed Form")
plt.legend(loc="upper left")
plt.title(f"Binary Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

### Error figures

plt.figure()
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[0][0] for v in vary], [v[0][2] for v in vary])],
    label="Euler-Maruyama Error",
)
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[0][1] for v in vary], [v[0][2] for v in vary])],
    label="Milstein Error",
)
plt.legend(loc="upper left")
plt.title(f"Asian Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[1][0] for v in vary], [v[1][2] for v in vary])],
    label="Euler-Maruyama Error",
)
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[1][1] for v in vary], [v[1][2] for v in vary])],
    label="Milstein Error",
)
plt.legend(loc="upper left")
plt.title(f"Lookback max Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[2][0] for v in vary], [v[2][2] for v in vary])],
    label="Euler-Maruyama Error",
)
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[2][1] for v in vary], [v[2][2] for v in vary])],
    label="Milstein Error",
)
plt.legend(loc="upper left")
plt.title(f"Lookback min Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()

plt.figure()
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[3][0] for v in vary], [v[3][2] for v in vary])],
    label="Euler-Maruyama Error",
)
plt.plot(
    variable,
    [b_i - a_i for a_i, b_i in zip([v[3][1] for v in vary], [v[3][2] for v in vary])],
    label="Milstein Error",
)
plt.legend(loc="upper left")
plt.title(f"Binary Option Price for Varying {variable_name}")
plt.xlabel(f"{variable_name}")
plt.ylabel("Price")
plt.show()
