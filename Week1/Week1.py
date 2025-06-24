import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import particles package components
import particles
from particles import state_space_models as ssm
from particles import mcmc
from particles import distributions
from particles.collectors import Moments

def fetch_sp500_data(start_date="2023-01-01", end_date="2024-11-01"):
    # Download S&P 500 data
    ticker = "^GSPC"
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Calculate log returns (skip first day as we need previous close)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1)) * 100
    data = data.dropna() #! remove any potential na's
    
    print(f"Downloaded {len(data)} trading days")
    print(f"Some basic stats: Mean={data['Returns'].mean():.3f}%, Std={data['Returns'].std():.2f}%")
    
    return data


# Fetch the data
data = fetch_sp500_data()

class StochasticVolatilityModel(ssm.StateSpaceModel):
    """
    Stochastic Volatility Model for S&P 500
    Parameters (μ, φ, σ^2):
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
        # State transition: X_t | X_{t-1}
        # X_{t-1} = xp
        mean = self.mu + self.phi * (xp - self.mu)
        return particles.distributions.Normal(loc=mean, scale=self.sigma_x)
    
    def PY(self, t, xp, x):
        """Observation distribution: Y_t | X_t"""
        # Y_t = exp(X_t/2) * eta_t where eta_t ~ N(0,1)
        volatility = np.exp(x / 2)
        return particles.distributions.Normal(loc=0.0, scale=volatility)

def run_particle_filter(model, returns, N=1000):
    """
    Run particle filter for stochastic volatility estimation
    """
    # Convert returns to numpy array
    y_data = returns.values
    T = len(y_data)
    
    # Create bootstrap filter
    fk_model = ssm.Bootstrap(ssm=model, data=y_data)
    
    # Run the algorithm
    pf = particles.SMC(fk=fk_model, N=N, collect=[Moments()])
    pf.run()
    
    # Filtering means and variances
    filtering_means = np.array([pf.summaries.moments[t]['mean'] for t in range(T)])
    filtering_vars = np.array([pf.summaries.moments[t]['var'] for t in range(T)])
    
    # Convert log-volatility to volatility percentage
    vol_estimates = np.exp(filtering_means / 2) * np.sqrt(252)  # Annualized volatility
    vol_std = np.sqrt(filtering_vars) * np.exp(filtering_means / 2) * np.sqrt(252)

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
    print(f" Running particle smoother with {N} particles...")
    
    # Convert returns to numpy array  
    y_data = returns.values
    T = len(y_data)
    
    # Run forward filter with history storage
    fk = ssm.Bootstrap(ssm=model, data=y_data)
    pf = particles.SMC(fk=fk, N=N, store_history=True)
    
    print(" Running forward pass...")
    pf.run()
    print("✓ Forward pass completed")
    
    # Check if history is available
    if pf.hist is None:
        print(" History not available, using filtering estimates as smoothing approximation")
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
        # Try different backward sampling methods based on particles version
        M = min(N//10, 100)  # Number of trajectories to sample
        print(f" Running backward sampling ({M} trajectories)...")
        
        # Method 1: Try the standard backward_sampling method
        if hasattr(pf.hist, 'backward_sampling'):
            paths = pf.hist.backward_sampling(M=M)
        
        # Method 2: Try backward_sampling_mcmc 
        elif hasattr(pf.hist, 'backward_sampling_mcmc'):
            paths = pf.hist.backward_sampling_mcmc(M=M)
        
        # Method 3: Use particles.backward_sampling function directly
        else:
            print(" Using direct backward sampling function...")
            # Use the particles.backward_sampling function
            paths = particles.backward_sampling(pf, M=M)
        
        print("✓ Backward sampling completed")
        
        # CRITICAL FIX: Convert to numpy array if it's a list
        if isinstance(paths, list):
            print("🔧 Converting list to numpy array...")
            paths = np.array(paths)
        
        print(f"   Paths shape: {paths.shape}")
        print(f"   Paths type: {type(paths)}")
        
        # IMPORTANT: Check the shape and transpose if needed
        if paths.shape[0] == M and paths.shape[1] == T:
            # Correct shape: M trajectories × T time points
            smoothed_means = np.mean(paths, axis=0)  # Average over trajectories
            smoothed_vars = np.var(paths, axis=0)
        elif paths.shape[0] == T and paths.shape[1] == M:
            # Transposed shape: T time points × M trajectories  
            smoothed_means = np.mean(paths, axis=1)  # Average over trajectories
            smoothed_vars = np.var(paths, axis=1)
        else:
            raise ValueError(f"Unexpected paths shape: {paths.shape}, expected ({M}, {T}) or ({T}, {M})")
        
        # Ensure we have the right length
        if len(smoothed_means) != T:
            raise ValueError(f"Smoothed means length {len(smoothed_means)} != data length {T}")
        
        # Compute smoothed estimates
        print(" Computing smoothed estimates...")
        
        # Convert to volatility percentage
        smooth_vol_estimates = np.exp(smoothed_means / 2) * np.sqrt(252)
        smooth_vol_std = np.sqrt(smoothed_vars) * np.exp(smoothed_means / 2) * np.sqrt(252)
        
        print("✓ Particle smoothing completed")
        print(f"   Output length: {len(smooth_vol_estimates)}")
        
        return {
            'log_vol_mean': smoothed_means,
            'log_vol_var': smoothed_vars, 
            'vol_estimates': smooth_vol_estimates,
            'vol_std': smooth_vol_std,
            'trajectories': paths
        }
        
    except Exception as e:
        print(f"  Backward sampling failed: {e}")
        print("   Using alternative smoothing approach...")
        
        # Alternative: Fixed-lag smoother approximation with LARGER window
        lag = 15  # Look ahead/behind 15 steps (increased from 5)
        
        # Re-run filter with moments collection
        print("🔄 Re-running filter for alternative smoothing...")
        pf_moments = particles.SMC(fk=fk, N=N, collect=[Moments()])
        pf_moments.run()
        
        filtering_means = np.array([pf_moments.summaries.moments[t]['mean'] for t in range(T)])
        filtering_vars = np.array([pf_moments.summaries.moments[t]['var'] for t in range(T)])
        
        # Enhanced smoothing: weighted moving average with exponential weights
        print("🔄 Computing enhanced smoothed estimates...")
        smoothed_means = np.copy(filtering_means)
        smoothed_vars = np.copy(filtering_vars)
        
        # Use exponential weights for better smoothing
        for t in range(T):
            start_idx = max(0, t - lag)
            end_idx = min(T, t + lag + 1)
            
            # Create exponential weights (more weight to closer observations)
            window_size = end_idx - start_idx
            weights = np.exp(-0.1 * np.abs(np.arange(window_size) - (t - start_idx)))
            weights = weights / np.sum(weights)  # Normalize
            
            # Apply weighted average
            smoothed_means[t] = np.average(filtering_means[start_idx:end_idx], weights=weights)
            smoothed_vars[t] = np.average(filtering_vars[start_idx:end_idx], weights=weights)
        
        # Convert to volatility percentage
        smooth_vol_estimates = np.exp(smoothed_means / 2) * np.sqrt(252)
        smooth_vol_std = np.sqrt(smoothed_vars) * np.exp(smoothed_means / 2) * np.sqrt(252)
        
        print("✓ Enhanced alternative smoothing completed")
        print(f"   Output length: {len(smooth_vol_estimates)}")
        
        return {
            'log_vol_mean': smoothed_means,
            'log_vol_var': smoothed_vars, 
            'vol_estimates': smooth_vol_estimates,
            'vol_std': smooth_vol_std,
            'trajectories': None
        }

def bayesian_parameter_estimation(returns, n_iter=2000, n_burn=500):
    """
    Bayesian parameter estimation using particles.mcmc.PMMH
    """
    print(f"🎯 Running PMMH parameter estimation ({n_iter} iterations)...")
    
    from particles import mcmc, distributions as dists
    
    # Define parameter priors (matching your original priors)
    prior_dict = {
        'mu': dists.Normal(loc=-2.5, scale=1.0),
        'phi': dists.Beta(a=20, b=2), 
        'sigma_x': dists.InvGamma(a=3, b=0.5)
    }
    my_prior = dists.StructDist(prior_dict)
    
    # Run PMMH using particles library
    print("🔗 Running PMMH chain...")
    pmmh = mcmc.PMMH(ssm_cls=StochasticVolatilityModel, 
                     prior=my_prior, 
                     data=returns.values, 
                     Nx=100,  # 100 particles (matching your original)
                     niter=n_iter)
    pmmh.run()
    
    print(f"✓ PMMH completed. Acceptance rate: {pmmh.acc_rate:.2%}")
    
    # Process results (matching your original output format)
    chain = np.array([[pmmh.chain.theta['mu'][i], 
                       pmmh.chain.theta['phi'][i], 
                       pmmh.chain.theta['sigma_x'][i]] 
                      for i in range(n_iter)])
    
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
        'acceptance_rate': pmmh.acc_rate
    }
    

# Run particle filtering for real-time volatility estimation
filtering_results = run_particle_filter(model, returns, N=1000)

print(f"\nFiltering Results Summary:")
print(f"  Average estimated volatility: {filtering_results['vol_estimates'].mean():.1f}%")
print(f"  Volatility range: {filtering_results['vol_estimates'].min():.1f}% - {filtering_results['vol_estimates'].max():.1f}%")


# Run particle smoothing for refined volatility estimates
smoothing_results = run_particle_smoother(model, returns, N=1000)

print(f"\nSmoothing Results Summary:")
print(f"  Average smoothed volatility: {smoothing_results['vol_estimates'].mean():.1f}%")
print(f"  Smoothed volatility range: {smoothing_results['vol_estimates'].min():.1f}% - {smoothing_results['vol_estimates'].max():.1f}%")

# Run MCMC for Bayesian parameter estimation
# Note: This can take several minutes to complete
mcmc_results = bayesian_parameter_estimation(returns, n_iter=2000, n_burn=500)

print(f"\nMCMC Results Summary:")
for param, est in mcmc_results['estimates'].items():
    print(f"  {param}: {est['mean']:.3f} ± {est['std']:.3f}")


# Print comprehensive analysis summary
print_analysis_summary(data, filtering_results, smoothing_results, mcmc_results)

print("\n✅ Analysis completed successfully!")

# =============================================================================
# Chunk 15: Store Results (Optional)
# =============================================================================

# Combine all results into a single dictionary for easy access
results = {
    'data': data,
    'filtering': filtering_results,
    'smoothing': smoothing_results,
    'mcmc': mcmc_results,
    'model': model
}

# Display final message
print(f"\n🎉 All results stored in 'results' dictionary")
print(f"   Available keys: {list(results.keys())}")