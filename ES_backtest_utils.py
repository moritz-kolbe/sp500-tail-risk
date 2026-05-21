import numpy as np
from scipy.stats import genpareto

losses_z_arr = None
sigma_arr = None
returns_arr = None
window = None


def init_worker(losses_z, sigma, returns, w):
    global losses_z_arr, sigma_arr, returns_arr, window
    losses_z_arr = losses_z
    sigma_arr = sigma
    returns_arr = returns
    window = w


def backtest_single(i):
    train = losses_z_arr[i - window : i]

    u = np.quantile(train, 0.95)
    exceedances = train[train > u] - u
    if len(exceedances) < 20:
        return np.nan, np.nan, False
    try:
        xi, _, beta = genpareto.fit(exceedances, floc=0)
        f_bar_u = len(exceedances) / len(train)
        var_z = u + (beta / xi) * (((1 - 0.99) / f_bar_u) ** (-xi) - 1)
        es_z = var_z / (1 - xi) + (beta - xi * u) / (1 - xi)

        sigma_i = sigma_arr[i] / 100
        var_return = var_z * sigma_i
        es_return = es_z * sigma_i

        actual_loss = -returns_arr[i]
        violation = actual_loss > var_return

        return var_return, es_return, violation
    except:
        return np.nan, np.nan, False
