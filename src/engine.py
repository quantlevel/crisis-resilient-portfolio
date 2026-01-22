"""
Portfolio Optimization Engine for Robust Portfolio Engineering

This module implements three portfolio optimization strategies:
1. Hierarchical Risk Parity (HRP) - Machine learning-based allocation
2. CVaR Optimization - Tail risk minimization
3. Mean-Variance Optimization (MVO) - Classic Markowitz optimization

All optimizers use Ledoit-Wolf shrinkage for robust covariance estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


class CovarianceEstimator:
    """
    Robust covariance matrix estimation using Ledoit-Wolf shrinkage.
    
    The Ledoit-Wolf shrinkage estimator optimally combines the sample
    covariance matrix with a structured estimator (identity matrix scaled
    by average variance) to produce a more stable estimate.
    
    This is critical for portfolio optimization where:
    - Estimation error in covariance leads to unstable weights
    - Small sample sizes amplify noise in sample covariance
    - Eigenvalue spreading reduces condition number problems
    
    The shrinkage formula is:
        Σ_shrunk = α * F + (1 - α) * S
    
    Where:
        α = shrinkage intensity (automatically determined)
        F = target matrix (scaled identity)
        S = sample covariance matrix
    """
    
    @staticmethod
    def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Compute the Ledoit-Wolf shrinkage covariance matrix.
        
        Args:
            returns: DataFrame of asset returns with assets as columns.
        
        Returns:
            Tuple containing:
                - Shrinkage covariance matrix as DataFrame
                - Shrinkage coefficient (0 = sample cov, 1 = structured estimator)
        
        Example:
            >>> cov_matrix, shrinkage = CovarianceEstimator.ledoit_wolf_shrinkage(returns)
            >>> print(f"Shrinkage intensity: {shrinkage:.4f}")
        """
        lw = LedoitWolf()
        lw.fit(returns.values)
        
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        
        return cov_matrix, lw.shrinkage_


class HRPOptimizer:
    """
    Hierarchical Risk Parity (HRP) portfolio optimizer.
    
    HRP is a machine learning-based portfolio allocation method that:
    1. Uses hierarchical clustering to group similar assets
    2. Reorders the covariance matrix via quasi-diagonalization
    3. Allocates weights using recursive bisection
    
    Key advantages over traditional MVO:
    - Does not require matrix inversion (more stable)
    - Naturally diversifies across asset clusters
    - More robust to estimation error
    
    Reference: López de Prado, M. (2016). "Building Diversified Portfolios 
    that Outperform Out-of-Sample"
    
    Attributes:
        returns (pd.DataFrame): Historical return data.
        cov_matrix (pd.DataFrame): Covariance matrix (Ledoit-Wolf shrunk).
        linkage_matrix (np.ndarray): Hierarchical clustering result.
        sorted_indices (list): Asset indices after quasi-diagonalization.
        weights (pd.Series): Final portfolio weights.
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize the HRP optimizer.
        
        Args:
            returns: DataFrame of asset returns with assets as columns.
        """
        self.returns = returns
        self.cov_matrix = None
        self.corr_matrix = None
        self.linkage_matrix = None
        self.sorted_indices = None
        self.weights = None
    
    def fit(self) -> pd.Series:
        """
        Fit the HRP model and compute optimal weights.
        
        This method executes the three stages of HRP:
        1. Tree Clustering - Build hierarchical tree of assets
        2. Quasi-Diagonalization - Reorder assets by cluster proximity
        3. Recursive Bisection - Allocate weights inversely to variance
        
        Returns:
            Series of portfolio weights indexed by asset names.
        """
        # Get shrinkage covariance matrix
        self.cov_matrix, shrinkage = CovarianceEstimator.ledoit_wolf_shrinkage(self.returns)
        self.corr_matrix = self.returns.corr()
        
        # Stage 1: Tree Clustering
        self.linkage_matrix = self._tree_clustering()
        
        # Stage 2: Quasi-Diagonalization
        self.sorted_indices = self._quasi_diagonalization()
        
        # Stage 3: Recursive Bisection
        self.weights = self._recursive_bisection()
        
        return self.weights
    
    def _tree_clustering(self) -> np.ndarray:
        """
        Stage 1: Build hierarchical tree using correlation distance.
        
        Converts correlation matrix to distance matrix using:
            d(i,j) = sqrt(0.5 * (1 - ρ_ij))
        
        This transformation ensures:
        - d(i,j) = 0 when ρ = 1 (perfect correlation)
        - d(i,j) = 1 when ρ = -1 (perfect negative correlation)
        - d(i,j) = 0.707 when ρ = 0 (no correlation)
        
        Uses Ward's method for agglomerative clustering, which minimizes
        within-cluster variance at each merge step.
        
        Returns:
            Linkage matrix from scipy hierarchical clustering.
        """
        # Correlation distance: d = sqrt(0.5 * (1 - correlation))
        dist_matrix = np.sqrt(0.5 * (1 - self.corr_matrix.values))
        
        # Convert to condensed form for scipy
        condensed_dist = squareform(dist_matrix, checks=False)
        
        # Hierarchical clustering using Ward's method
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        return linkage_matrix
    
    def _quasi_diagonalization(self) -> list:
        """
        Stage 2: Reorder covariance matrix to place similar assets together.
        
        Quasi-diagonalization reorders rows and columns of the covariance
        matrix so that the largest values lie along the diagonal. This 
        groups correlated assets together, making the recursive bisection
        step more effective at distributing risk.
        
        Uses the dendrogram leaf ordering from hierarchical clustering.
        
        Returns:
            List of asset indices in the reordered (quasi-diagonal) sequence.
        """
        # Get the order of leaves from the dendrogram
        sorted_indices = leaves_list(self.linkage_matrix)
        return sorted_indices.tolist()
    
    def _get_cluster_variance(self, cluster_items: list) -> float:
        """
        Calculate the variance of a cluster of assets.
        
        For a portfolio with inverse-variance weights within the cluster,
        the portfolio variance is:
            σ²_p = 1 / Σ(1/σ²_i) for uncorrelated assets
        
        For correlated assets, we use the full covariance matrix.
        
        Args:
            cluster_items: List of asset indices in the cluster.
        
        Returns:
            Portfolio variance of the cluster.
        """
        sub_cov = self.cov_matrix.iloc[cluster_items, cluster_items]
        
        # Inverse-variance weights within cluster
        inv_var = 1.0 / np.diag(sub_cov.values)
        inv_var_weights = inv_var / inv_var.sum()
        
        # Portfolio variance: w' * Σ * w
        cluster_var = np.dot(inv_var_weights, np.dot(sub_cov.values, inv_var_weights))
        
        return cluster_var
    
    def _recursive_bisection(self) -> pd.Series:
        """
        Stage 3: Allocate weights using recursive bisection.
        
        This algorithm recursively splits the assets into two clusters
        and allocates weights inversely proportional to cluster variance:
        
        1. Start with full allocation (weight = 1) for all assets
        2. Split assets into two clusters using the dendrogram
        3. Calculate variance of each sub-cluster
        4. Allocate to each cluster inversely proportional to variance
        5. Recursively repeat for each sub-cluster
        
        The inverse-variance allocation ensures that:
        - Riskier clusters receive smaller allocations
        - Similar-risk clusters receive similar allocations
        - Total portfolio risk is minimized without matrix inversion
        
        Returns:
            Series of portfolio weights indexed by asset names.
        """
        # Initialize weights
        weights = pd.Series(1.0, index=self.returns.columns)
        
        # List of clusters to process (start with all assets in sorted order)
        clusters = [self.sorted_indices]
        
        while clusters:
            # Split each cluster that has more than one element
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # Split at the midpoint
                    mid = len(cluster) // 2
                    left_cluster = cluster[:mid]
                    right_cluster = cluster[mid:]
                    
                    # Calculate cluster variances
                    left_var = self._get_cluster_variance(left_cluster)
                    right_var = self._get_cluster_variance(right_cluster)
                    
                    # Inverse-variance allocation
                    alloc_factor = 1 - left_var / (left_var + right_var)
                    
                    # Update weights
                    left_assets = self.returns.columns[left_cluster]
                    right_assets = self.returns.columns[right_cluster]
                    
                    weights[left_assets] *= alloc_factor
                    weights[right_assets] *= (1 - alloc_factor)
                    
                    # Add to new clusters for further processing
                    if len(left_cluster) > 1:
                        new_clusters.append(left_cluster)
                    if len(right_cluster) > 1:
                        new_clusters.append(right_cluster)
            
            clusters = new_clusters
        
        return weights
    
    def get_dendrogram_data(self) -> Dict:
        """
        Get data needed to plot the dendrogram.
        
        Returns:
            Dictionary with linkage matrix and labels for plotting.
        """
        return {
            'linkage_matrix': self.linkage_matrix,
            'labels': self.returns.columns.tolist()
        }


class CVaROptimizer:
    """
    Conditional Value at Risk (CVaR) portfolio optimizer.
    
    CVaR, also known as Expected Shortfall (ES), measures the expected
    loss given that losses exceed the VaR threshold. Minimizing CVaR
    creates portfolios that are robust to tail risk.
    
    Mathematical definition:
        CVaR_α = E[Loss | Loss > VaR_α]
    
    At 95% confidence (α = 0.95):
        CVaR = average of the worst 5% of losses
    
    Advantages over variance-based optimization:
    - Accounts for tail risk (fat tails in return distributions)
    - Coherent risk measure (satisfies subadditivity)
    - More appropriate for non-normal returns (like crypto)
    
    The optimization problem:
        minimize: CVaR(portfolio)
        subject to: Σw_i = 1 (fully invested)
                   0 ≤ w_i ≤ 1 (long only)
    
    Attributes:
        returns (pd.DataFrame): Historical return data.
        confidence (float): Confidence level for CVaR (default 0.95).
        weights (pd.Series): Optimal portfolio weights.
    """
    
    def __init__(self, returns: pd.DataFrame, confidence: float = 0.95):
        """
        Initialize the CVaR optimizer.
        
        Args:
            returns: DataFrame of asset returns with assets as columns.
            confidence: Confidence level for CVaR calculation (default 0.95).
        """
        self.returns = returns
        self.confidence = confidence
        self.weights = None
    
    def _calculate_cvar(self, weights: np.ndarray) -> float:
        """
        Calculate CVaR for a given weight vector.
        
        CVaR is computed as the average of returns below the VaR threshold.
        Since we're working with returns (not losses), we look at the
        left tail of the distribution.
        
        Args:
            weights: Array of portfolio weights.
        
        Returns:
            CVaR (Expected Shortfall) as a positive number (loss).
        """
        # Calculate portfolio returns
        portfolio_returns = self.returns.values @ weights
        
        # VaR is the (1 - confidence) percentile
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence) * 100)
        
        # CVaR is the mean of returns below VaR
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        # Return as positive number (loss)
        cvar = -np.mean(tail_returns)
        
        return cvar
    
    def fit(self) -> pd.Series:
        """
        Fit the CVaR optimization model.
        
        Uses SLSQP (Sequential Least Squares Programming) to minimize
        CVaR subject to:
        - Sum of weights = 1 (fully invested)
        - 0 ≤ weight_i ≤ 1 (long only constraint)
        
        Returns:
            Series of optimal portfolio weights indexed by asset names.
        """
        n_assets = len(self.returns.columns)
        
        # Initial guess: equal weights
        init_weights = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Fully invested
        ]
        
        # Bounds: 0 to 1 for each weight (long only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            fun=self._calculate_cvar,
            x0=init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: CVaR optimization may not have converged: {result.message}")
        
        self.weights = pd.Series(result.x, index=self.returns.columns)
        
        # Normalize to ensure sum = 1 (handle numerical precision)
        self.weights = self.weights / self.weights.sum()
        
        return self.weights


class MVOOptimizer:
    """
    Mean-Variance Optimization (MVO) - Maximum Sharpe Ratio.
    
    Classic Markowitz optimization that maximizes the Sharpe Ratio:
        Sharpe = (E[R_p] - R_f) / σ_p
    
    Where:
        E[R_p] = expected portfolio return = Σw_i * μ_i
        σ_p = portfolio standard deviation = sqrt(w' * Σ * w)
        R_f = risk-free rate (assumed 0 for simplicity)
    
    The optimization problem (negative because we minimize):
        minimize: -Sharpe(w) = -(w'μ) / sqrt(w'Σw)
        subject to: Σw_i = 1 (fully invested)
                   0 ≤ w_i ≤ 1 (long only)
    
    Note: MVO is known to be sensitive to estimation error in expected
    returns and covariance. This implementation uses Ledoit-Wolf shrinkage
    for the covariance matrix to improve stability.
    
    Attributes:
        returns (pd.DataFrame): Historical return data.
        risk_free_rate (float): Annualized risk-free rate.
        weights (pd.Series): Optimal portfolio weights.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize the MVO optimizer.
        
        Args:
            returns: DataFrame of asset returns with assets as columns.
            risk_free_rate: Annualized risk-free rate (default 0.0).
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.cov_matrix = None
        self.expected_returns = None
    
    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for minimization.
        
        Args:
            weights: Array of portfolio weights.
        
        Returns:
            Negative Sharpe ratio (we minimize this).
        """
        # Annualized expected return (252 trading days)
        portfolio_return = np.sum(self.expected_returns.values * weights) * 252
        
        # Annualized portfolio volatility
        portfolio_var = np.dot(weights, np.dot(self.cov_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var * 252)
        
        # Sharpe ratio
        if portfolio_vol == 0:
            return 0
        
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return -sharpe  # Negative because we minimize
    
    def fit(self) -> pd.Series:
        """
        Fit the MVO model to maximize Sharpe ratio.
        
        Uses Ledoit-Wolf shrinkage for covariance estimation and
        SLSQP optimization with long-only constraints.
        
        Returns:
            Series of optimal portfolio weights indexed by asset names.
        """
        # Estimate covariance with Ledoit-Wolf shrinkage
        self.cov_matrix, _ = CovarianceEstimator.ledoit_wolf_shrinkage(self.returns)
        
        # Expected returns (historical mean)
        self.expected_returns = self.returns.mean()
        
        n_assets = len(self.returns.columns)
        
        # Initial guess: equal weights
        init_weights = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Fully invested
        ]
        
        # Bounds: 0 to 1 for each weight (long only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            fun=self._negative_sharpe,
            x0=init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Warning: MVO optimization may not have converged: {result.message}")
        
        self.weights = pd.Series(result.x, index=self.returns.columns)
        
        # Normalize to ensure sum = 1
        self.weights = self.weights / self.weights.sum()
        
        return self.weights


class PortfolioAnalytics:
    """
    Analytics toolkit for portfolio performance evaluation.
    
    Provides methods to calculate:
    - Cumulative returns
    - Rolling metrics (volatility, Sharpe)
    - Maximum drawdown
    - Performance summary statistics
    """
    
    @staticmethod
    def calculate_portfolio_returns(
        returns: pd.DataFrame, 
        weights: pd.Series
    ) -> pd.Series:
        """
        Calculate portfolio returns given asset returns and weights.
        
        Args:
            returns: DataFrame of asset returns.
            weights: Series of portfolio weights.
        
        Returns:
            Series of portfolio returns.
        """
        # Align weights with returns columns
        aligned_weights = weights.reindex(returns.columns).fillna(0)
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        return portfolio_returns
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns from period returns.
        
        Uses the formula: (1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1
        
        For log returns, this is equivalent to sum of log returns.
        
        Args:
            returns: Series of period returns.
        
        Returns:
            Series of cumulative returns starting from 0.
        """
        cumulative = (1 + returns).cumprod() - 1
        return cumulative
    
    @staticmethod
    def calculate_drawdown(returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series from returns.
        
        Drawdown measures the peak-to-trough decline:
            DD_t = (Peak_t - Value_t) / Peak_t
        
        Maximum drawdown is the largest drawdown over the period.
        
        Args:
            returns: Series of period returns.
        
        Returns:
            Series of drawdown values (negative, representing losses).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of period returns.
        
        Returns:
            Maximum drawdown as a negative decimal (e.g., -0.20 = 20% drawdown).
        """
        drawdown = PortfolioAnalytics.calculate_drawdown(returns)
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, 
        risk_free_rate: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate the Sharpe ratio.
        
        Sharpe = (Mean Return - Risk Free Rate) / Standard Deviation
        
        Args:
            returns: Series of period returns.
            risk_free_rate: Annualized risk-free rate.
            annualize: If True, annualize the ratio (assume 252 trading days).
        
        Returns:
            Sharpe ratio.
        """
        excess_return = returns.mean() - risk_free_rate / 252
        volatility = returns.std()
        
        if volatility == 0:
            return 0.0
        
        sharpe = excess_return / volatility
        
        if annualize:
            sharpe *= np.sqrt(252)
        
        return sharpe
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate portfolio volatility (standard deviation).
        
        Args:
            returns: Series of period returns.
            annualize: If True, annualize the volatility.
        
        Returns:
            Volatility (annualized if specified).
        """
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(252)
        
        return vol
    
    @staticmethod
    def generate_performance_summary(
        returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Generate a performance summary DataFrame for multiple strategies.
        
        Args:
            returns_dict: Dictionary mapping strategy names to return series.
        
        Returns:
            DataFrame with performance metrics for each strategy.
        """
        metrics = []
        
        for name, returns in returns_dict.items():
            metrics.append({
                'Strategy': name,
                'Annualized Volatility': f"{PortfolioAnalytics.calculate_volatility(returns):.2%}",
                'Sharpe Ratio': f"{PortfolioAnalytics.calculate_sharpe_ratio(returns):.2f}",
                'Max Drawdown': f"{PortfolioAnalytics.calculate_max_drawdown(returns):.2%}",
                'Cumulative Return': f"{PortfolioAnalytics.calculate_cumulative_returns(returns).iloc[-1]:.2%}"
            })
        
        return pd.DataFrame(metrics).set_index('Strategy')
