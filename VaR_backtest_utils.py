import numpy as np
from scipy.stats import genpareto
from arch import arch_model

losses_arr = None
window = None


def init_worker(data, w):
    global losses_arr, window
    losses_arr = data
    window = w


def backtest_single(i):
    train = losses_arr[i - window : i]
    try:
        garch = arch_model(train * 100, vol="GARCH", p=1, q=1)
        result = garch.fit(disp="off")

        std_resid = result.resid / result.conditional_volatility
        losses_z = -std_resid[~np.isnan(std_resid)]

        u = np.quantile(losses_z, 0.95)
        exceedances = losses_z[losses_z > u] - u
        if len(exceedances) < 20:
            return np.nan, np.nan, False

        xi, _, beta = genpareto.fit(exceedances, floc=0)
        f_bar_u = len(exceedances) / len(losses_z)
        var_z = u + (beta / xi) * (((1 - 0.99) / f_bar_u) ** (-xi) - 1)
        es_z = (var_z + beta - xi * u) / (1 - xi)

        forecast = result.forecast(horizon=1)
        sigma_next = np.sqrt(forecast.variance.values[-1, 0]) / 100

        var_return = var_z * sigma_next
        es_return = es_z * sigma_next
        actual_loss = -losses_arr[i]
        violation = actual_loss > var_return

        return var_return, es_return, violation
    except:
        return np.nan, np.nan, False
