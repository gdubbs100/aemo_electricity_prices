"""
Unified interface for fitting and analyzing different probability distributions
for regression error terms (Gaussian, Cauchy, Levy Stable).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levy_stable
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DistributionParams:
    """Container for distribution parameters with nice string representation"""
    params: Dict[str, float]
    name: str
    
    def __repr__(self):
        param_str = "; ".join([f"{k}={v:.4f}" for k, v in self.params.items()])
        return f"{self.name} ({param_str})"
    
    def __str__(self):
        return self.__repr__()


class DistributionFitter:
    """Unified interface for fitting different distributions to data"""
    
    @staticmethod
    def fit_gaussian(data: np.ndarray) -> DistributionParams:
        """
        Fit Gaussian (Normal) distribution to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to fit
            
        Returns
        -------
        DistributionParams
            Parameters: mu (mean), sigma (std dev)
        """
        mu, sigma = stats.norm.fit(data)
        return DistributionParams(
            params={'mu': mu, 'sigma': sigma},
            name='Gaussian'
        )
    
    @staticmethod
    def fit_cauchy(data: np.ndarray) -> DistributionParams:
        """
        Fit Cauchy distribution to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to fit
            
        Returns
        -------
        DistributionParams
            Parameters: loc (location), scale
        """
        loc, scale = stats.cauchy.fit(data)
        return DistributionParams(
            params={'loc': loc, 'scale': scale},
            name='Cauchy'
        )
    
    @staticmethod
    def fit_levy_stable(data: np.ndarray) -> DistributionParams:
        """
        Fit Levy Stable distribution to data using quantile-based estimation.
        
        Converts from internal parameterization to S parameterization.
        Reference: https://stackoverflow.com/questions/54564850
        
        Parameters
        ----------
        data : np.ndarray
            Data to fit
            
        Returns
        -------
        DistributionParams
            Parameters: alpha (stability), beta (skewness), loc, scale
        """
        # Convert to S parameterization
        def pconv(alpha, beta, mu, sigma):
            return (
                alpha, beta, 
                mu - sigma * beta * np.tan(np.pi * alpha / 2.0), 
                sigma
            )
        
        alpha, beta, loc, scale = pconv(*levy_stable._fitstart(data))
        return DistributionParams(
            params={'alpha': alpha, 'beta': beta, 'loc': loc, 'scale': scale},
            name='Levy Stable'
        )
    
    @staticmethod
    def fit(data: np.ndarray, distribution: str) -> DistributionParams:
        """
        Fit specified distribution to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data to fit
        distribution : str
            Distribution type: 'gaussian', 'cauchy', 'levy' or 'levy_stable'
            
        Returns
        -------
        DistributionParams
            Fitted distribution parameters
            
        Raises
        ------
        ValueError
            If distribution type is unknown
        """
        data_clean = np.asarray(data)
        data_clean = data_clean[~np.isnan(data_clean)]
        
        if distribution.lower() == 'gaussian':
            return DistributionFitter.fit_gaussian(data_clean)
        elif distribution.lower() == 'cauchy':
            return DistributionFitter.fit_cauchy(data_clean)
        elif distribution.lower() in ['levy', 'levy_stable']:
            return DistributionFitter.fit_levy_stable(data_clean)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")


class DistributionSampler:
    """Generate samples from fitted distributions"""
    
    @staticmethod
    def sample(dist_params: DistributionParams, size: int = int(1e5), 
               random_state: int = 42) -> np.ndarray:
        """
        Generate samples from a fitted distribution.
        
        Parameters
        ----------
        dist_params : DistributionParams
            Fitted distribution parameters
        size : int, optional
            Number of samples to generate (default: 100,000)
        random_state : int, optional
            Random seed for reproducibility (default: 42)
            
        Returns
        -------
        np.ndarray
            Generated samples
            
        Raises
        ------
        ValueError
            If distribution is not recognized
        """
        dist_name = dist_params.name.lower().replace(' ', '_')
        
        if 'gaussian' in dist_name:
            return stats.norm.rvs(
                loc=dist_params.params['mu'],
                scale=dist_params.params['sigma'],
                size=size,
                random_state=random_state
            )
        elif 'cauchy' in dist_name:
            return stats.cauchy.rvs(
                loc=dist_params.params['loc'],
                scale=dist_params.params['scale'],
                size=size,
                random_state=random_state
            )
        elif 'levy' in dist_name:
            return levy_stable.rvs(
                alpha=dist_params.params['alpha'],
                beta=dist_params.params['beta'],
                loc=dist_params.params['loc'],
                scale=dist_params.params['scale'],
                size=size,
                random_state=random_state
            )
        else:
            raise ValueError(f"Cannot sample from: {dist_params.name}")


class DistributionQuantiles:
    """Compute quantiles for fitted distributions"""
    
    @staticmethod
    def get_quantiles(dist_params: DistributionParams, quantiles: list) -> np.ndarray:
        """
        Get quantile values for a fitted distribution.
        
        Parameters
        ----------
        dist_params : DistributionParams
            Fitted distribution parameters
        quantiles : list or array
            Quantile levels (between 0 and 1), e.g., [0.025, 0.5, 0.975]
            
        Returns
        -------
        np.ndarray
            Quantile values
            
        Raises
        ------
        ValueError
            If distribution is not recognized
        """
        quantiles = np.asarray(quantiles)
        dist_name = dist_params.name.lower().replace(' ', '_')
        
        if 'gaussian' in dist_name:
            return stats.norm.ppf(
                quantiles,
                loc=dist_params.params['mu'],
                scale=dist_params.params['sigma']
            )
        elif 'cauchy' in dist_name:
            return stats.cauchy.ppf(
                quantiles,
                loc=dist_params.params['loc'],
                scale=dist_params.params['scale']
            )
        elif 'levy' in dist_name:
            return levy_stable.ppf(
                quantiles,
                alpha=dist_params.params['alpha'],
                beta=dist_params.params['beta'],
                loc=dist_params.params['loc'],
                scale=dist_params.params['scale']
            )
        else:
            raise ValueError(f"Cannot get quantiles from: {dist_params.name}")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_distribution_comparison(data_dict: Dict[str, np.ndarray], 
                                  title: str = "Distribution Comparison",
                                  xlim: Tuple[float, float] = None):
    """
    Compare CDFs of multiple distributions.
    
    Parameters
    ----------
    data_dict : Dict[str, np.ndarray]
        Dictionary mapping labels to data arrays, e.g.,
        {'Actual': data, 'Gaussian': samples, ...}
    title : str, optional
        Plot title (default: "Distribution Comparison")
    xlim : Tuple[float, float], optional
        X-axis limits
    """
    plt.figure(figsize=(10, 6))
    
    for label, data in data_dict.items():
        data_clean = data[~np.isnan(data)]
        x = np.sort(data_clean)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, label=label, where='post', alpha=0.8)
    
    plt.title(title, fontsize=12, fontweight='bold')
    if xlim:
        plt.xlim(xlim)
    plt.xlabel('Value')
    plt.ylabel('P(X ≤ x)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_quantile_qq(dist_params: DistributionParams, data: np.ndarray,
                      title: str = None):
    """
    Plot quantile-quantile (Q-Q) comparison between fitted distribution and data.
    
    Perfect fit would lie on the diagonal line y=x.
    
    Parameters
    ----------
    dist_params : DistributionParams
        Fitted distribution parameters
    data : np.ndarray
        Empirical data for comparison
    title : str, optional
        Plot title. If not provided, default title is used.
    """
    data_clean = data[~np.isnan(data)]
    quantiles = np.linspace(0.01, 0.99, 100)
    
    theoretical_q = DistributionQuantiles.get_quantiles(dist_params, quantiles)
    empirical_q = np.quantile(data_clean, quantiles)
    
    plt.figure(figsize=(8, 8))
    plt.plot(theoretical_q, empirical_q, 'o', alpha=0.6, label='Data vs Fit', markersize=4)
    
    # Add diagonal reference line (perfect fit)
    min_val = min(theoretical_q.min(), empirical_q.min())
    max_val = max(theoretical_q.max(), empirical_q.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
    
    plt.xlabel(f'{dist_params.name} Theoretical Quantiles', fontsize=11)
    plt.ylabel('Empirical Quantiles', fontsize=11)
    plt.title(title or f'Q-Q Plot: {dist_params.name}', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_prediction_ribbons(data: pd.DataFrame, x_axis: str, predictions: str, 
                            actuals: str, reversion: str, dist_params: DistributionParams,
                            quantile_levels: list = [0.025, 0.25, 0.75, 0.975],
                            title: str = None, ylim: tuple = None):
    """
    Plot predictions with uncertainty ribbons based on fitted error distribution.
    
    Creates visualization with:
    - 95% confidence interval (outer ribbon, lighter color)
    - Interquartile range (inner ribbon, darker color)
    - Predictions and actual values as points
    
    Parameters
    ----------
    x_axis : np.ndarray
        X-axis values (e.g., time indices)
    predictions : np.ndarray
        Model predictions
    actuals : np.ndarray
        Actual observed values
    dist_params : DistributionParams
        Fitted error distribution parameters
    quantile_levels : list, optional
        Quantiles to use for ribbons. Default [0.025, 0.25, 0.75, 0.975]
        corresponds to 95% CI and IQR.
    title : str, optional
        Plot title. If not provided, default title is used.
    """

    x = data[x_axis].values
    preds = data[predictions].values
    acts = data[actuals].values
    rev = data[reversion].values

    q_offsets = DistributionQuantiles.get_quantiles(dist_params, quantile_levels)

    q_lower_outer = preds + rev * q_offsets[0]
    q_lower_inner = preds + rev * q_offsets[1]
    q_upper_inner = preds + rev * q_offsets[2]
    q_upper_outer = preds + rev * q_offsets[3]

    plt.figure(figsize=(12, 6))

    # Outer ribbon (95% CI)
    plt.fill_between(x, q_lower_outer, q_upper_outer, 
                     color='royalblue', alpha=0.2, label='95% Range')

    # Inner ribbon (IQR)
    plt.fill_between(x, q_lower_inner, q_upper_inner, 
                     color='royalblue', alpha=0.4, label='IQR (25%-75%)')

    # Point estimates
    plt.scatter(x, preds, color='red', label='Prediction', 
                alpha=0.3, s=20)
    plt.scatter(x, acts, color='black', label='Actual', 
                alpha=0.3, s=20)

    plt.title(title or f"Prediction Ribbons ({dist_params.name} Errors)",
              fontsize=12, fontweight='bold')
    plt.xlabel("Time / Index")
    plt.ylabel("Value")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def get_coverage_statistics(predictions: np.ndarray, actuals: np.ndarray, rev: np.ndarray,
                           dist_params: DistributionParams) -> Dict[str, float]:
    """
    Calculate what percentage of actuals fall within different quantile ranges.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    actuals : np.ndarray
        Actual observed values
    dist_params : DistributionParams
        Fitted error distribution parameters
        
    Returns
    -------
    Dict[str, float]
        Coverage statistics, e.g., {'iqr_coverage': 0.50, 'ci_95_coverage': 0.95}
    """
    q_offsets = DistributionQuantiles.get_quantiles(
        dist_params, 
        [0.025, 0.25, 0.75, 0.975]
    )
    
    lower_iqr = predictions + rev * q_offsets[1]
    upper_iqr = predictions + rev * q_offsets[2]
    lower_95 = predictions + rev * q_offsets[0]
    upper_95 = predictions + rev * q_offsets[3]
    
    in_iqr = ((lower_iqr <= actuals) & (actuals <= upper_iqr)).sum() / len(actuals)
    in_95 = ((lower_95 <= actuals) & (actuals <= upper_95)).sum() / len(actuals)
    
    return {
        'iqr_coverage': in_iqr,
        'ci_95_coverage': in_95,
        'distribution': dist_params.name
    }
