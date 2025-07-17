class StochasticVolatilityModel(ssm.StateSpaceModel):
    """
    Stochastic Volatility Model with logit-transformed φ parameter
    Parameters (μ, logit_phi, σ^2):
    - mu: long-run log-volatility level
    - logit_phi: logit-transformed persistence parameter
    - sigma_x: volatility of log-volatility
    
    Note: phi = exp(logit_phi) / (1 + exp(logit_phi))
    """
    
    def __init__(self, mu=-2.5, logit_phi=2.94, sigma_x=0.3):
        self.mu = mu
        self.logit_phi = logit_phi
        self.sigma_x = sigma_x
        
        # Convert logit_phi back to phi for internal calculations
        self.phi = np.exp(logit_phi) / (1 + np.exp(logit_phi))
        
    def PX0(self):
        """Initial distribution of log-volatility"""
        # Stationary distribution: X_0 ~ N(mu, sigma_x^2 / (1 - phi^2))
        var_stat = self.sigma_x**2 / (1 - self.phi**2)
        return particles.distributions.Normal(loc=self.mu, scale=np.sqrt(var_stat))
    
    def PX(self, t, xp):
        # State transition: X_t | X_{t-1}
        # X_{t-1} = xp
        mean = self.mu + self.phi * (xp - self.mu)
        return particles.distributions.Normal(loc=mean, scale=self.sigma_x)
    
    def PY(self, t, xp, x):
        """Observation distribution: Y_t | X_t"""
        # Y_t = exp(X_t/2) * eta_t where eta_t ~ N(0,1)
        volatility = np.exp(x / 2)
        return particles.distributions.Normal(loc=0.0, scale=volatility)

class BlackScholesModel:
    """
    Black-Scholes model with constant volatility
    """
    def __init__(self, volatility=0.2):
        self.volatility = volatility
    
    def fit(self, returns):
        """Fit the model to returns data"""
        # Calculate realized volatility (annualized)
        self.volatility = returns.std() * np.sqrt(252)
        return self
    
    def predict_volatility(self, n_periods):
        """Predict constant volatility for n periods"""
        return np.full(n_periods, self.volatility)
    
    def log_likelihood(self, returns):
        """Calculate log-likelihood of returns under constant volatility"""
        n = len(returns)
        vol_daily = self.volatility / np.sqrt(252)
        
        # Log-likelihood of normal distribution
        ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(vol_daily) - \
             0.5 * np.sum((returns / vol_daily) ** 2)
        return ll

def plot_acf(chain, param_names, max_lags=50):
    """Plot autocorrelation function for MCMC chains"""
    
    
    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 4))
    if len(param_names) == 1:
        axes = [axes]
    
    for i, param in enumerate(param_names):
        # Calculate ACF
        autocorr = acf(chain[:, i], nlags=max_lags, fft=True)
        lags = np.arange(len(autocorr))
        
        axes[i].plot(lags, autocorr, 'b-', linewidth=2)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% threshold')
        axes[i].axhline(y=-0.05, color='r', linestyle='--', alpha=0.7)
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('Autocorrelation')
        axes[i].set_title(f'ACF for {param}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def effective_sample_size(chain):
    """Calculate effective sample size for MCMC chains"""
    from statsmodels.tsa.stattools import acf
    
    n_samples, n_params = chain.shape
    ess_values = []
    
    for i in range(n_params):
        # Calculate autocorrelation
        max_lags = min(n_samples // 4, 200)  # Don't use too many lags
        try:
            autocorrs = acf(chain[:, i], nlags=max_lags, fft=True)
            
            # Find first negative autocorrelation or where it drops below 0.05
            cutoff = 1
            for lag in range(1, len(autocorrs)):
                if autocorrs[lag] <= 0.05:
                    cutoff = lag
                    break
            
            # Calculate integrated autocorrelation time
            tau_int = 1 + 2 * np.sum(autocorrs[1:cutoff+1])
            
            # Effective sample size
            ess = n_samples / (2 * tau_int)
            ess_values.append(max(ess, 1))  # Ensure ESS is at least 1
            
        except:
            # Fallback calculation
            ess_values.append(n_samples / 10)  # Conservative estimate
    
    return np.array(ess_values)

def mcmc_diagnostics(chain, param_names, burned_chain):
    """Comprehensive MCMC diagnostics"""
    print("\n📊 MCMC Convergence Diagnostics:")
    print("=" * 40)
    
    # Basic chain statistics
    n_total, n_params = chain.shape
    n_burned = len(burned_chain)
    
    print(f"Total iterations: {n_total}")
    print(f"Burn-in samples: {n_total - n_burned}")
    print(f"Post-burn samples: {n_burned}")
    
    # Effective sample size
    ess = effective_sample_size(burned_chain)
    print(f"\nEffective Sample Sizes:")
    for i, param in enumerate(param_names):
        print(f"  {param}: {ess[i]:.1f} ({ess[i]/n_burned:.1%} of total)")
    
    # Plot traces and ACF
    fig, axes = plt.subplots(2, n_params, figsize=(15, 8))
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    
    for i, param in enumerate(param_names):
        # Trace plot
        axes[0, i].plot(chain[:, i], alpha=0.7)
        axes[0, i].axvline(x=n_total-n_burned, color='r', linestyle='--', 
                          label='Burn-in end')
        axes[0, i].set_title(f'Trace: {param}')
        axes[0, i].set_ylabel('Value')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()
        
        # Density plot
        axes[1, i].hist(burned_chain[:, i], bins=50, density=True, alpha=0.7)
        axes[1, i].set_title(f'Posterior: {param}')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ACF plots
    plot_acf(burned_chain, param_names)
    
    return ess
def bayesian_parameter_estimation_numpyro(returns, n_warmup=1000, n_samples=2000, n_chains=4, n_burn=500):
    """
    Bayesian parameter estimation for SV model using NumPyro with NUTS (HMC variant).
    Uses non-centered reparameterization for better mixing.
    """
    print(f"🎯 Running NumPyro NUTS estimation ({n_chains} chains, {n_warmup} warmup, {n_samples} samples)...")
    
    # Convert returns to numpy array first (fix for pandas Series)
    returns_np = returns.values.astype(np.float32)  # Use float32 for efficiency
    
    # Empirical mu for prior centering (adjustment for log-returns in decimal)
    emp_mu = np.mean(np.log(returns_np**2 + 1e-16)) + 1.2704  # Bias correction
    print(f"Empirical mu estimate: {emp_mu:.2f}")
    
    # Convert to JAX array
    returns_jax = jnp.array(returns_np)
    
    def sv_model(returns):
        mu = numpyro.sample('mu', dist.Normal(loc=emp_mu, scale=2.0))
        
        # Logit reparam for phi in (0,1)
        logit_phi = numpyro.sample('logit_phi', dist.Normal(loc=2.94, scale=1.0))
        phi = numpyro.deterministic('phi', jnp.exp(logit_phi) / (1 + jnp.exp(logit_phi)))
        
        sigma_x = numpyro.sample('sigma_x', dist.InverseGamma(3, 0.5))
        
        T = returns.shape[0]
        
        # Non-centered: sample i.i.d innovations
        eta = numpyro.sample('eta', dist.Normal(0, 1), sample_shape=(T,))
        
        # Deterministically build h (log-vol) using scan for efficiency
        def build_h(carry, xs):
            t, eta_t = xs
            h_t = mu + phi * (carry - mu) + sigma_x * eta_t
            return h_t, h_t
        
        # Initial h_0 from stationary dist
        h_init = mu + (sigma_x / jnp.sqrt(1 - phi**2 + 1e-10)) * eta[0]  # Avoid div by zero
        
        # Scan over t=1 to T-1
        _, h_rest = jax.lax.scan(build_h, h_init, (jnp.arange(1, T), eta[1:]))
        h = jnp.concatenate([h_init[None], h_rest])
        
        numpyro.deterministic('h', h)
        
        # Observations (decimal returns)
        numpyro.sample('obs', dist.Normal(0.0, jnp.exp(h / 2)), obs=returns)
    
    # Set up NUTS sampler
    nuts_kernel = NUTS(sv_model)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains, progress_bar=True)
    
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, returns=returns_jax)
    
    # Get samples (dict with 'mu', 'logit_phi', 'sigma_x', etc.; phi is deterministic)
    samples = mcmc.get_samples()
    
    # Compute phi samples from logit_phi
    phi_samples = jnp.exp(samples['logit_phi']) / (1 + jnp.exp(samples['logit_phi']))
    
    # Handle single vs multi-chain shaping
    is_single_chain = (samples['mu'].ndim == 1)
    if is_single_chain:
        burned_mu = samples['mu'][n_burn:]
        burned_phi = phi_samples[n_burn:]
        burned_sigma_x = samples['sigma_x'][n_burn:]
        # Reshape to (1, -1) for consistency in ESS etc.
        burned_mu = burned_mu[None, :]
        burned_phi = burned_phi[None, :]
        burned_sigma_x = burned_sigma_x[None, :]
    else:
        burned_mu = samples['mu'][:, n_burn:]
        burned_phi = phi_samples[:, n_burn:]
        burned_sigma_x = samples['sigma_x'][:, n_burn:]
    
    burned_samples = {
        'mu': burned_mu,
        'phi': burned_phi,
        'sigma_x': burned_sigma_x
    }
    
    # Chain for output (flatten chains)
    burned_chain = jnp.stack([burned_mu, burned_phi, burned_sigma_x], axis=-1).reshape(-1, 3)
    
    param_names = ['mu', 'phi', 'sigma_x']
    
    # Diagnostics: ESS on burned samples (expects shape (chains, samples))
    ess = {p: numpyro_ess(burned_samples[p]) for p in param_names}
    ess_values = [ess[p] for p in param_names]
    print("\n📊 MCMC Convergence Diagnostics:")
    print(f"Effective Sample Sizes: {ess}")
    
    # Traces and ACF use full samples (adjust for single chain)
    if is_single_chain:
        mu_full = samples['mu'][None, :]
        phi_full = phi_samples[None, :]
        sigma_x_full = samples['sigma_x'][None, :]
    else:
        mu_full = samples['mu']
        phi_full = phi_samples
        sigma_x_full = samples['sigma_x']
    
    full_samples_for_plot = {
        'mu': mu_full,
        'phi': phi_full,
        'sigma_x': sigma_x_full
    }
    
    for i, param in enumerate(param_names):
        param_samples = full_samples_for_plot[param]
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(param_samples.T, alpha=0.5)
        plt.title(f'Trace: {param}')
        plt.axvline(n_burn, color='r', linestyle='--')
        
        plt.subplot(1, 2, 2)
        lags = np.arange(50)
        autocorr = acf(param_samples.mean(axis=0), nlags=49)  # Avg over chains
        plt.plot(lags, autocorr)
        plt.axhline(0.05, color='r', linestyle='--')
        plt.axhline(-0.05, color='r', linestyle='--')
        plt.title(f'ACF: {param}')
        plt.show()
    
    # Parameter estimates from burned_chain
    param_estimates = {
        param: {'mean': jnp.mean(burned_chain[:, i]), 'std': jnp.std(burned_chain[:, i])}
        for i, param in enumerate(param_names)
    }
    
    # Also store logit_phi
    if is_single_chain:
        logit_phi_burned = samples['logit_phi'][n_burn:]
    else:
        logit_phi_burned = samples['logit_phi'][:, n_burn:].reshape(-1)
    logit_phi_estimates = {
        'mean': jnp.mean(logit_phi_burned),
        'std': jnp.std(logit_phi_burned)
    }
    
    return {
        'chain': jnp.stack([burned_mu, burned_phi, burned_sigma_x], axis=-1),
        'burned_chain': burned_chain,
        'estimates': param_estimates,
        'logit_phi_estimates': logit_phi_estimates,
        'effective_sample_size': ess_values,
        'param_names': param_names,
        'mcmc': mcmc  # For further inspection
    }


# Usage (replace your PMMH call)

# Your existing execution flow:
data = fetch_sp500_data(start_date="2022-01-01", end_date="2025-01-01")
train_data, test_data = split_data(data, train_ratio=0.8)
returns_train = train_data['Returns']
returns_test = test_data['Returns']


# Convert percentage returns to decimal returns
train_data_decimal = train_data.copy()
train_data_decimal['Returns'] = returns_train / 100
# Convert percentage returns to decimal returns
returns_train_decimal = returns_train / 100
returns_test_decimal = returns_test / 100
test_data_decimal = test_data.copy()
test_data_decimal['Returns'] = returns_test / 100


model = StochasticVolatilityModel()
filtering_results = run_particle_filter(model, returns_train, N=1000)
smoothing_results = run_particle_smoother(model, returns_train, N=1000)

mcmc_results = bayesian_parameter_estimation_numpyro(returns_train_decimal, n_warmup=5000, n_samples=20000)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from jax import random
import jax.numpy as jnp

def compute_sv_predictive_nlpd(mcmc_results, train_data_decimal, test_data_decimal):
    """
    Optimized: Vectorized NLPD for SV using posterior latents.
    """
    # Hardcode defaults
    n_burn = 500
    n_chains = 4
    
    # Get posterior samples (flatten)
    samples = mcmc_results['mcmc'].get_samples(group_by_chain=False)
    n_burn_total = n_burn * n_chains
    burned_indices = slice(n_burn_total, None)
    
    mu = samples['mu'][burned_indices]
    logit_phi = samples['logit_phi'][burned_indices]
    phi = jnp.exp(logit_phi) / (1 + jnp.exp(logit_phi))
    sigma_x = samples['sigma_x'][burned_indices]
    eta = samples['eta'][burned_indices]  # (n_samples, T_train)
    
    # Vectorized h reconstruction
    def compute_h_single(mu_i, phi_i, sigma_x_i, eta_i):
        h_init = mu_i + (sigma_x_i / jnp.sqrt(1 - phi_i**2 + 1e-10)) * eta_i[0]
        def scan_fn(carry, eta_j):
            h_j = mu_i + phi_i * (carry - mu_i) + sigma_x_i * eta_j
            return h_j, h_j
        _, h_rest = jax.lax.scan(scan_fn, h_init, eta_i[1:])
        return jnp.concatenate([h_init[None], h_rest])
    
    compute_h = jax.vmap(compute_h_single)  # Vectorize over samples
    h = compute_h(mu, phi, sigma_x, eta)  # Shape (n_samples, T_train)
    
    test_returns = jnp.array(test_data_decimal['Returns'].values)
    sv_nlpd_scores = []
    
    rng_key = random.PRNGKey(0)
    for test_return in test_returns:
        rng_key, subkey = random.split(rng_key)
        eps = random.normal(subkey, shape=(len(mu),))
        h_future = mu + phi * (h[:, -1] - mu) + sigma_x * eps
        pred_scale = jnp.exp(h_future / 2)
        log_probs = dist.Normal(0.0, pred_scale).log_prob(test_return)
        mean_log_prob = jnp.mean(log_probs)
        nlpd = -mean_log_prob
        sv_nlpd_scores.append(nlpd.item())
    
    mean_nlpd = np.mean(sv_nlpd_scores)
    print(f"Mean Predictive NLPD (SV): {mean_nlpd:.4f}")
    return sv_nlpd_scores

# Calculate Black-Scholes NLPD for comparison
def calculate_black_scholes_nlpd(train_data_decimal, test_data_decimal):
    """
    Calculate NLPD scores for Black-Scholes model
    """
    # Fit Black-Scholes (constant volatility) on training data
    train_returns = train_data_decimal['Returns']
    bs_volatility = train_returns.std()  # Daily volatility (decimal)
    
    # Calculate NLPD for each test return
    test_returns = test_data_decimal['Returns'].values
    bs_nlpd_scores = []
    
    for return_val in test_returns:
        # Black-Scholes: assume normal distribution N(0, bs_volatility^2)
        log_prob = norm.logpdf(return_val, loc=0, scale=bs_volatility)
        nlpd = -log_prob  # Negative log probability density
        bs_nlpd_scores.append(nlpd)
    
    print(f"Mean NLPD (Black-Scholes): {np.mean(bs_nlpd_scores):.4f}")
    
    return bs_nlpd_scores

# Compare NLPD scores
def compare_nlpd_scores(sv_nlpd, bs_nlpd):
    """
    Compare NLPD scores between models
    """
    # Align lengths (in case of different sizes due to None values)
    min_length = min(len(sv_nlpd), len(bs_nlpd))
    sv_aligned = np.array(sv_nlpd[:min_length]).flatten()
    bs_aligned = np.array(bs_nlpd[:min_length])
    
    # Calculate summary statistics
    sv_mean = np.mean(sv_aligned)
    bs_mean = np.mean(bs_aligned)
    
    improvement = (bs_mean - sv_mean) / bs_mean * 100
    print(f"Debug: SV array shape = {sv_aligned.shape}, BS array shape = {bs_aligned.shape}")
    print(f"Debug: SV length = {len(sv_aligned)}, BS length = {len(bs_aligned)}")

    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(bs_aligned, sv_aligned)
    
    print(f"\n📊 NLPD COMPARISON RESULTS:")
    print(f"=" * 40)
    print(f"Stochastic Volatility NLPD: {sv_mean:.4f}")
    print(f"Black-Scholes NLPD:        {bs_mean:.4f}")
    print(f"Improvement:                {improvement:.1f}%")
    print(f"Statistical significance:    p = {p_value:.4f}")
    
    if improvement > 0:
        print(f"✅ Stochastic Volatility model is BETTER (lower NLPD)")
    else:
        print(f"❌ Black-Scholes model is better (lower NLPD)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Time series of NLPD scores
    dates = test_data_decimal.index[:min_length]
    axes[0].plot(dates, sv_aligned, 'b-', label='Stochastic Volatility', linewidth=2)
    axes[0].plot(dates, bs_aligned, 'r-', label='Black-Scholes', linewidth=2)
    axes[0].set_title('NLPD Scores Over Time')
    axes[0].set_ylabel('NLPD (lower is better)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1].boxplot([sv_aligned, bs_aligned], labels=['Stochastic Vol', 'Black-Scholes'])
    axes[1].set_title('NLPD Distribution Comparison')
    axes[1].set_ylabel('NLPD (lower is better)')
    axes[1].grid(True, alpha=0.3)
    
    # Bar chart of means
    models = ['Stochastic Vol', 'Black-Scholes']
    means = [sv_mean, bs_mean]
    colors = ['blue', 'red']
    bars = axes[2].bar(models, means, color=colors, alpha=0.7)
    axes[2].set_title('Mean NLPD Comparison')
    axes[2].set_ylabel('Mean NLPD (lower is better)')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, means):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'sv_nlpd': sv_aligned,
        'bs_nlpd': bs_aligned,
        'sv_mean': sv_mean,
        'bs_mean': bs_mean,
        'improvement': improvement,
        'p_value': p_value
    }

# Run the NLPD analysis
print("📊 Extracting and comparing NLPD scores...")

# Compute SV NLPD scores
sv_nlpd_scores = compute_sv_predictive_nlpd(mcmc_results, train_data_decimal, test_data_decimal)

# Calculate BS NLPD scores
bs_nlpd_scores = calculate_black_scholes_nlpd(train_data_decimal, test_data_decimal)

# Compare the models
nlpd_comparison = compare_nlpd_scores(sv_nlpd_scores, bs_nlpd_scores)

print(f"\n✅ NLPD analysis completed!")
print(f"Results stored in 'nlpd_comparison' dictionary")