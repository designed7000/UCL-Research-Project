# Scalable Bayesian Model Selection for Time-Series Volatility Using Sequential Monte Carlo

UCL MSc Dissertation Project

---

## Overview

This repository contains the research code for a UCL dissertation project investigating Bayesian approaches to financial forecasting. The project implements Stochastic Volatility (SV) models using Sequential Monte Carlo (SMC) and Hamiltonian Monte Carlo (HMC) methods to model and forecast asset price dynamics.

The research explores both univariate (1D) and bivariate (2D) cases, applying these methods to real financial data from the S&P 500 and NASDAQ indices.

---

## Project Structure

```
.
├── 1d case/                              # Univariate stochastic volatility analysis
│   ├── univariate_sv_analysis.ipynb      # Main notebook: HMC estimation and forecasting
│   ├── univariate_sv_draft.ipynb         # Development notebook (exploratory work)
│   └── sv_model.py                       # Core Python module with model implementations
│
├── 2d case/                              # Bivariate stochastic volatility analysis
│   ├── bivariate_sv_sp500_nasdaq.ipynb   # Main notebook: S&P 500 and NASDAQ analysis
│   ├── bivariate_sv_spy_eem.ipynb        # SPY and Emerging Markets (EEM) analysis
│   └── bivariate_sv_spy_eem_alt.ipynb    # Alternative model specification for SPY/EEM
│
├── pyproject.toml                        # Project dependencies and metadata
└── README.md                             # This file
```

---

## Methodology

### Stochastic Volatility Model

The project implements the standard log-volatility stochastic volatility model:

- **Observation equation**: Returns are modelled as normally distributed with time-varying volatility
- **State equation**: Log-volatility follows an AR(1) process

### Inference Methods

1. **Sequential Monte Carlo (SMC)**: Particle filtering for online state estimation
2. **Hamiltonian Monte Carlo (HMC)**: NUTS sampler via NumPyro for Bayesian parameter estimation
3. **Bootstrap Particle Filter**: For likelihood evaluation and predictive distributions

### Key Features

- Bayesian parameter estimation with diagnostic tools (ESS, trace plots, ACF)
- One-step-ahead predictive distributions
- Negative Log Predictive Density (NLPD) for model comparison
- Extension to multivariate (correlated) assets in the 2D case

---

## Dependencies

The project requires Python 3.9 or higher. Key dependencies include:

| Package | Purpose |
|---------|---------|
| NumPyro | HMC/NUTS inference |
| JAX | Automatic differentiation and GPU acceleration |
| particles | Sequential Monte Carlo methods |
| yfinance | Financial data retrieval |
| matplotlib, seaborn | Visualisation |
| pandas, numpy, scipy | Data manipulation and numerical computing |

Install all dependencies using:

```bash
pip install -e .
```

Or install directly from pyproject.toml:

```bash
pip install numpyro jax particles yfinance matplotlib seaborn pandas numpy scipy arviz statsmodels
```

---

## Usage

### 1D Case (Univariate Analysis)

Open and run the notebook `1d case/univariate_sv_analysis.ipynb` to:

1. Fetch S&P 500 data via yfinance
2. Estimate SV model parameters using HMC
3. Run particle filtering for volatility estimation
4. Generate one-step-ahead forecasts
5. Evaluate model performance using NLPD

### 2D Case (Bivariate Analysis)

Open and run the notebook `2d case/bivariate_sv_sp500_nasdaq.ipynb` to:

1. Fetch S&P 500 and NASDAQ data
2. Fit a bivariate SV model with correlated volatilities
3. Analyse cross-asset dependencies
4. Generate joint forecasts

---

## Data

The analysis uses daily log-returns from:

- **S&P 500** (ticker: ^GSPC or SPY)
- **NASDAQ Composite** (ticker: ^IXIC)

Data is fetched automatically via the `yfinance` package. Default date ranges can be modified in the data fetching functions.

---

## References

- Doucet, A., de Freitas, N., and Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
- Kim, S., Shephard, N., and Chib, S. (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models. *Review of Economic Studies*, 65(3), 361-393.
- Hoffman, M. D. and Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593-1623.
- Chopin, N., Papaspiliopoulos, O., et al. (2020). *An Introduction to Sequential Monte Carlo*. Springer.
- Harvey, A., Ruiz, E., and Shephard, N. (1994). Multivariate Stochastic Variance Models. *Review of Economic Studies*, 61(2), 247-264.
- Phan, D., Pradhan, N., and Jankowiak, M. (2019). Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro. *arXiv preprint arXiv:1912.11554*.

---

## Author

Alexandros Chortis

Supervised by Professor Alexandros Beskos and Christopher Stanton  
UCL Department of Statistical Science

---

## License

This project was developed for academic research purposes at University College London.
