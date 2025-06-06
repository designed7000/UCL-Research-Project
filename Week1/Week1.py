print("hello")
"""
S&P 500 Stochastic Volatility Analysis using Particles Package
============================================================

This application implements:
1. Data fetching from Yahoo Finance API
2. Stochastic volatility model using particles package
3. Filtering (real-time volatility estimation)
4. Smoothing (hindsight volatility estimation)
5. Bayesian parameter estimation via Particle MCMC

Model:
- State: X_t = log-volatility
- Evolution: X_t = mu + phi*(X_{t-1} - mu) + sigma_x * epsilon_t
- Observation: Y_t = exp(X_t/2) * eta_t

Author: Generated for S&P 500 volatility analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import particles package components
try:
    import particles
    from particles import state_space_models as ssm
    from particles import mcmc
    from particles import distributions
    from particles.collectors import Moments
    print("✓ Particles package imported successfully")
except ImportError:
    print("❌ Please install particles package: pip install particles")
    exit()

class StochasticVolatilityModel(ssm.StateSpaceModel):
    """
    Stochastic Volatility Model for S&P 500
    
    State equation: X_t = mu + phi*(X_{t-1} - mu) + sigma_x * epsilon_t
    Observation equation: Y_t = exp(X_t/2) * eta_t
    
    Parameters:
    - mu: long-run log-volatility level
    - phi: persistence parameter (0 < phi < 1)
    - sigma_x: volatility of log-volatility
    """
    
    def __init__(self, mu=-2.5, phi=0.95, sigma_x=0.3):
        self.mu = mu
        self.phi = phi 
        self.sigma_x = sigma_x
        
    def PX0(self):
        """Initial distribution of log-volatility"""
        # Stationary distribution: X_0 ~ N(mu, sigma_x^2 / (1 - phi^2))
        var_stat = self.sigma_x**2 / (1 - self.phi**2)
        return particles.distributions.Normal(loc=self.mu, scale=np.sqrt(var_stat))
    
    def PX(self, t, xp):
        """State transition: X_t | X_{t-1}"""
        mean = self.mu + self.phi * (xp - self.mu)
        return particles.distributions.Normal(loc=mean, scale=self.sigma_x)
    
    def PY(self, t, xp, x):
        """Observation distribution: Y_t | X_t"""
        # Y_t = exp(X_t/2) * eta_t where eta_t ~ N(0,1)
        volatility = np.exp(x / 2)
        return particles.distributions.Normal(loc=0.0, scale=volatility)

def fetch_sp500_data(start_date="2023-01-01", end_date="2024-11-01"):
    """
    Fetch S&P 500 data from Yahoo Finance and compute log returns
    """
    print(f"📊 Fetching S&P 500 data from {start_date} to {end_date}...")
    
    # Download S&P 500 data
    ticker = "^GSPC"
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Calculate log returns (skip first day as we need previous close)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    data = data.dropna()
    
    print(f"✓ Downloaded {len(data)} trading days")
    print(f"✓ Return statistics: Mean={data['Returns'].mean():.3f}%, Std={data['Returns'].std():.2f}%")
    
    return data

def run_particle_filter(model, returns, N=1000):
    """
    Run particle filter for stochastic volatility estimation
    """
    print(f"🔄 Running particle filter with {N} particles...")
    
    # Convert returns to numpy array
    y_data = returns.values
    T = len(y_data)
    
    # Create the Feynman-Kac model
    fk_model = ssm.Bootstrap(ssm=model, data=y_data)
    
    # Run particle filter
    pf = particles.SMC(fk=fk_model, N=N, collect=[Moments()])
    pf.run()
    
    # Extract filtering means and variances
    filtering_means = np.array([pf.summaries.moments[t]['mean'] for t in range(T)])
    filtering_vars = np.array([pf.summaries.moments[t]['var'] for t in range(T)])
    
    # Convert log-volatility to volatility percentage
    vol_estimates = np.exp(filtering_means / 2) * np.sqrt(252)  # Annualized volatility
    vol_std = np.sqrt(filtering_vars) * np.exp(filtering_means / 2) * np.sqrt(252)
    
    print("✓ Particle filtering completed")
    
    return {
        'log_vol_mean': filtering_means,
        'log_vol_var': filtering_vars,
        'vol_estimates': vol_estimates,
        'vol_std': vol_std,
        'particle_filter': pf
    }

def run_particle_smoother(model, returns, N=1000):
    """
    Run particle smoother for refined volatility estimates
    """
    print(f"🔄 Running particle smoother with {N} particles...")
    
    # Convert returns to numpy array  
    y_data = returns.values
    T = len(y_data)
    
    # Run forward filter with history storage
    fk = ssm.Bootstrap(ssm=model, data=y_data)
    pf = particles.SMC(fk=fk, N=N, store_history=True)  # Enable history storage
    pf.run()
    
    # Check if history is available
    if pf.hist is None:
        print("⚠️  History not available, using filtering estimates as smoothing approximation")
        # Fallback: use filtering means as approximation
        filtering_means = np.array([np.mean(pf.X) for _ in range(T)])
        filtering_vars = np.array([np.var(pf.X) for _ in range(T)])
        
        smooth_vol_estimates = np.exp(filtering_means / 2) * np.sqrt(252)
        smooth_vol_std = np.sqrt(filtering_vars) * np.exp(filtering_means / 2) * np.sqrt(252)
        
        return {
            'log_vol_mean': filtering_means,
            'log_vol_var': filtering_vars, 
            'vol_estimates': smooth_vol_estimates,
            'vol_std': smooth_vol_std,
            'trajectories': None
        }
    
    try:
        # Run backward sampling for smoothing
        M = min(N//10, 100)  # Number of trajectories to sample
        paths = pf.hist.backward_sampling(M=M)
        
        # Compute smoothed estimates
        smoothed_means = np.mean(paths, axis=0)
        smoothed_vars = np.var(paths, axis=0)
        
        # Convert to volatility percentage
        smooth_vol_estimates = np.exp(smoothed_means / 2) * np.sqrt(252)
        smooth_vol_std = np.sqrt(smoothed_vars) * np.exp(smoothed_means / 2) * np.sqrt(252)
        
        print("✓ Particle smoothing completed")
        
        return {
            'log_vol_mean': smoothed_means,
            'log_vol_var': smoothed_vars, 
            'vol_estimates': smooth_vol_estimates,
            'vol_std': smooth_vol_std,
            'trajectories': paths
        }
        
    except Exception as e:
        print(f"⚠️  Backward sampling failed: {e}")
        print("   Using alternative smoothing approach...")
        
        # Alternative: Fixed-lag smoother approximation
        # Use weighted average of nearby filtering estimates
        lag = 5  # Look ahead/behind 5 steps
        
        # Re-run filter with moments collection
        pf_moments = particles.SMC(fk=fk, N=N, collect=[Moments()])
        pf_moments.run()
        
        filtering_means = np.array([pf_moments.summaries.moments[t]['mean'] for t in range(T)])
        filtering_vars = np.array([pf_moments.summaries.moments[t]['var'] for t in range(T)])
        
        # Simple smoothing: moving average
        smoothed_means = np.copy(filtering_means)
        smoothed_vars = np.copy(filtering_vars)
        
        for t in range(T):
            start_idx = max(0, t - lag)
            end_idx = min(T, t + lag + 1)
            smoothed_means[t] = np.mean(filtering_means[start_idx:end_idx])
            smoothed_vars[t] = np.mean(filtering_vars[start_idx:end_idx])
        
        # Convert to volatility percentage
        smooth_vol_estimates = np.exp(smoothed_means / 2) * np.sqrt(252)
        smooth_vol_std = np.sqrt(smoothed_vars) * np.exp(smoothed_means / 2) * np.sqrt(252)
        
        print("✓ Alternative smoothing completed")
        
        return {
            'log_vol_mean': smoothed_means,
            'log_vol_var': smoothed_vars, 
            'vol_estimates': smooth_vol_estimates,
            'vol_std': smooth_vol_std,
            'trajectories': None
        }

def bayesian_parameter_estimation(returns, n_iter=2000, n_burn=500):
    """
    Bayesian parameter estimation using Particle MCMC
    """
    print(f"🎯 Running Bayesian parameter estimation ({n_iter} iterations)...")
    
    # Define parameter priors
    class SVPrior:
        def __init__(self):
            # mu ~ N(-2.5, 1^2)  
            self.mu_prior = stats.norm(loc=-2.5, scale=1.0)
            # phi ~ Beta(20, 2) scaled to (0,1) for stationarity
            self.phi_prior = stats.beta(a=20, b=2)
            # sigma_x ~ InvGamma(3, 0.5) 
            self.sigma_x_prior = stats.invgamma(a=3, scale=0.5)
            
        def logpdf(self, theta):
            mu, phi, sigma_x = theta
            if not (0 < phi < 1 and sigma_x > 0):
                return -np.inf
            return (self.mu_prior.logpdf(mu) + 
                   self.phi_prior.logpdf(phi) +
                   self.sigma_x_prior.logpdf(sigma_x))
        
        def rvs(self, size=None):
            mu = self.mu_prior.rvs(size)
            phi = self.phi_prior.rvs(size) 
            sigma_x = self.sigma_x_prior.rvs(size)
            if size is None:
                return np.array([mu, phi, sigma_x])
            else:
                return np.column_stack([mu, phi, sigma_x])
    
    # MCMC sampler
    class SVPosterior:
        def __init__(self, data):
            self.data = data
            self.prior = SVPrior()
            
        def logpdf(self, theta):
            mu, phi, sigma_x = theta
            
            # Prior
            log_prior = self.prior.logpdf(theta)
            if np.isinf(log_prior):
                return log_prior
                
            # Likelihood via particle filter
            try:
                model = StochasticVolatilityModel(mu=mu, phi=phi, sigma_x=sigma_x)
                fk = ssm.Bootstrap(ssm=model, data=self.data)
                pf = particles.SMC(fk=fk, N=100)  # Smaller N for MCMC
                pf.run()
                log_likelihood = pf.logLt
                return log_prior + log_likelihood
            except:
                return -np.inf
    
    # Run MCMC
    y_data = returns.values
    posterior = SVPosterior(y_data)
    
    # Initial value
    theta0 = np.array([-2.5, 0.95, 0.3])
    
    # MCMC chain
    print("🔗 Running MCMC chain...")
    chain = []
    current_theta = theta0
    current_logpdf = posterior.logpdf(current_theta)
    n_accept = 0
    
    # Proposal covariance (tuned)
    prop_cov = np.diag([0.1, 0.01, 0.02])**2
    
    for i in range(n_iter):
        if i % 500 == 0:
            print(f"   Iteration {i}/{n_iter}")
            
        # Propose new state
        proposal = np.random.multivariate_normal(current_theta, prop_cov)
        proposal_logpdf = posterior.logpdf(proposal)
        
        # Accept/reject
        log_alpha = proposal_logpdf - current_logpdf
        if np.log(np.random.rand()) < log_alpha:
            current_theta = proposal
            current_logpdf = proposal_logpdf
            n_accept += 1
            
        chain.append(current_theta.copy())
    
    print(f"✓ MCMC completed. Acceptance rate: {n_accept/n_iter:.2%}")
    
    # Process results
    chain = np.array(chain)
    burned_chain = chain[n_burn:]
    
    # Parameter estimates
    param_estimates = {
        'mu': {'mean': np.mean(burned_chain[:, 0]), 'std': np.std(burned_chain[:, 0])},
        'phi': {'mean': np.mean(burned_chain[:, 1]), 'std': np.std(burned_chain[:, 1])},
        'sigma_x': {'mean': np.mean(burned_chain[:, 2]), 'std': np.std(burned_chain[:, 2])}
    }
    
    return {
        'chain': chain,
        'burned_chain': burned_chain,
        'estimates': param_estimates,
        'acceptance_rate': n_accept/n_iter
    }

def create_comprehensive_plots(data, filtering_results, smoothing_results, mcmc_results):
    """
    Create comprehensive visualization of results
    """
    print("📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('S&P 500 Stochastic Volatility Analysis', fontsize=16, fontweight='bold')
    
    dates = data.index
    returns = data['Returns']
    
    # 1. Returns time series
    axes[0,0].plot(dates, returns, 'b-', alpha=0.7, linewidth=0.8)
    axes[0,0].set_title('S&P 500 Daily Returns')
    axes[0,0].set_ylabel('Returns (%)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Filtering vs Smoothing
    vol_filter = filtering_results['vol_estimates']
    vol_smooth = smoothing_results['vol_estimates']
    
    axes[0,1].plot(dates, vol_filter, 'r-', label='Filtered', alpha=0.8)
    axes[0,1].plot(dates, vol_smooth, 'g-', label='Smoothed', alpha=0.8)
    axes[0,1].fill_between(dates, 
                          vol_filter - 1.96*filtering_results['vol_std'],
                          vol_filter + 1.96*filtering_results['vol_std'],
                          alpha=0.2, color='red', label='95% CI (Filter)')
    axes[0,1].set_title('Volatility Estimates: Filtering vs Smoothing')
    axes[0,1].set_ylabel('Annualized Volatility (%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Returns vs Volatility
    axes[0,2].scatter(vol_filter[:-1], np.abs(returns[1:]), alpha=0.6, s=20)
    axes[0,2].set_xlabel('Estimated Volatility (%)')
    axes[0,2].set_ylabel('Absolute Returns (%)')
    axes[0,2].set_title('Volatility vs Absolute Returns')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. MCMC trace plots
    chain = mcmc_results['burned_chain']
    param_names = ['μ (Long-run log-vol)', 'φ (Persistence)', 'σₓ (Vol-of-vol)']
    
    for i, param in enumerate(param_names):
        axes[1,i].plot(chain[:, i], alpha=0.7)
        axes[1,i].set_title(f'MCMC Trace: {param}')
        axes[1,i].set_ylabel('Parameter Value')
        axes[1,i].set_xlabel('Iteration')
        axes[1,i].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(chain[:, i])
        axes[1,i].axhline(y=mean_val, color='red', linestyle='--', 
                         label=f'Mean: {mean_val:.3f}')
        axes[1,i].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics table
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📈 DATA SUMMARY:")
    print(f"   Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"   Trading days: {len(returns)}")
    print(f"   Average return: {returns.mean():.3f}%")
    print(f"   Return volatility: {returns.std():.2f}%")
    print(f"   Annualized volatility: {returns.std() * np.sqrt(252):.1f}%")
    
    print(f"\n🎯 VOLATILITY ESTIMATES:")
    print(f"   Average filtered volatility: {vol_filter.mean():.1f}%")
    print(f"   Average smoothed volatility: {vol_smooth.mean():.1f}%")
    print(f"   Max volatility (filtered): {vol_filter.max():.1f}%")
    print(f"   Min volatility (filtered): {vol_filter.min():.1f}%")
    
    print(f"\n⚙️  PARAMETER ESTIMATES:")
    estimates = mcmc_results['estimates']
    print(f"   μ (long-run log-vol): {estimates['mu']['mean']:.3f} ± {estimates['mu']['std']:.3f}")
    print(f"   φ (persistence): {estimates['phi']['mean']:.3f} ± {estimates['phi']['std']:.3f}")  
    print(f"   σₓ (vol-of-vol): {estimates['sigma_x']['mean']:.3f} ± {estimates['sigma_x']['std']:.3f}")
    print(f"   MCMC acceptance rate: {mcmc_results['acceptance_rate']:.1%}")
    
    # Business interpretation
    long_run_vol = np.exp(estimates['mu']['mean']/2) * np.sqrt(252)
    persistence_days = -1 / np.log(estimates['phi']['mean'])
    
    print(f"\n💼 BUSINESS INTERPRETATION:")
    print(f"   Long-run volatility: {long_run_vol:.1f}% (annualized)")
    print(f"   Volatility shock half-life: {persistence_days:.1f} trading days")
    print(f"   Volatility clustering: {'Strong' if estimates['phi']['mean'] > 0.9 else 'Moderate'}")
    
    return fig

def main():
    """
    Main application workflow
    """
    print("🚀 S&P 500 Stochastic Volatility Analysis")
    print("=" * 50)
    
    # 1. Data fetching
    data = fetch_sp500_data(start_date="2023-01-01", end_date="2024-11-01")
    returns = data['Returns']
    
    # 2. Initial model (will be updated via MCMC)
    model = StochasticVolatilityModel(mu=-2.5, phi=0.95, sigma_x=0.3)
    
    # 3. Particle filtering
    filtering_results = run_particle_filter(model, returns, N=1000)
    
    # 4. Particle smoothing  
    smoothing_results = run_particle_smoother(model, returns, N=1000)
    
    # 5. Bayesian parameter estimation
    mcmc_results = bayesian_parameter_estimation(returns, n_iter=2000, n_burn=500)
    
    # 6. Visualization and summary
    create_comprehensive_plots(data, filtering_results, smoothing_results, mcmc_results)
    
    print("\n✅ Analysis completed successfully!")
    
    return {
        'data': data,
        'filtering': filtering_results,
        'smoothing': smoothing_results,
        'mcmc': mcmc_results
    }

if __name__ == "__main__":
    # Run the complete analysis
    results = main()