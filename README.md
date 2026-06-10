# sp500-tail-risk
📄 [View rendered analysis](https://moritz-kolbe.github.io/sp500-tail-risk)

Extreme value analysis of S&P 500 tail risk: GARCH prefiltering, POT/GPD 
estimation, Bayesian inference via MCMC, rolling-window backtesting.

## Motivation

Standard risk models assume normally distributed returns and static volatility. 
Both assumptions fail for equity index data. This project combines GARCH 
pre-filtering with Peaks-over-Threshold (POT) extreme value analysis to produce 
tail risk estimates that account for volatility clustering and heavy-tailed 
innovations. A complementary unconditional analysis characterises long-run tail 
behaviour via return levels.

## Methodology

**Conditional analysis (short-term risk management)**

1. GARCH(1,1) pre-filtering on S&P 500 log-returns to extract standardised 
   residuals
2. POT/GPD estimation on residuals (threshold u = 2.2 standard deviations) 
   via MCMC using `emcee`, yielding full posterior distributions for ξ and σ
3. Credible intervals for next-day VaR and ES via posterior propagation and 
   rescaling by conditional volatility
4. 10-day forecast via HS-GARCH-EVT simulation (McNeil & Frey, 2000), compared 
   against naive square-root-of-time scaling
5. Rolling-window backtest (window = 1000 days) with Kupiec and Christoffersen 
   tests

**Unconditional analysis (long-run tail characterisation)**

POT/GPD estimation directly on raw log-losses, producing return levels for 
horizons of 1, 5, 10, 20, and 50 years with Bayesian credible intervals.

## Main Results

The conditional model produces an empirical violation rate of 1.03% against a 
nominal 1% over a 43-year out-of-sample window (110/10,691 observations). The 
Kupiec test does not reject correct unconditional coverage (p = 0.77). The 
Christoffersen independence test rejects at the 5% level (p = 0.034), indicating 
residual violation clustering — a known limitation of GARCH(1,1) that motivates 
extensions such as GJR-GARCH or Student-t innovations.

The 10-day cumulative VaR from simulation is 6.46%, compared to 7.03% from naive 
scaling — confirming that the square-root-of-time rule overstates tail risk under 
GARCH dynamics.

The 50-year return level from the unconditional model is approximately 15% 
(95% CI: 10.8%–25.2%), reflecting substantial parameter uncertainty at long 
horizons.

## How to Run

```bash
pip install -r requirements.txt
quarto render SP500_POT_Analysis.qmd --to html
```

The backtest uses `multiprocessing.Pool` and is parallelised across available 
CPU cores. Adjust the number of workers in the Pool calls to match your machine.

## References

McNeil, A.J. & Frey, R. (2000). Estimation of tail-related risk measures for 
heteroscedastic financial time series: an extreme value approach. Journal of 
Empirical Finance, 7, 271–300.

McNeil, A.J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management. 
Princeton University Press.