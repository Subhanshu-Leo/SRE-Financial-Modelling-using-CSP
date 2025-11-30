"""
Paper 1: Toeplitz Approximation to Empirical Correlation Matrix
High-quality implementation with error handling and validation
"""

import numpy as np
import pandas as pd
from scipy. fftpack import dct, idct
from scipy.optimize import minimize_scalar
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import warnings

class ToeplitzDCTAnalyzer:
        
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with returns data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (N assets x T time points)
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        
        if returns.isnull().any().any():
            warnings.warn("Returns contain NaN values. They will be dropped.")
            returns = returns.dropna()
        
        self.returns = returns
        self.N = returns.shape[1]  # Number of assets
        self. T = returns.shape[0]  # Number of time points
        self.asset_names = returns.columns.tolist()
        
        print(f"Initialized with {self.N} assets and {self.T} observations")
    
    def empirical_correlation(self, window: Optional[int] = None) -> np.ndarray:
        """
        Compute empirical correlation matrix
        
        Parameters:
        -----------
        window : int, optional
            Use last 'window' observations.  If None, use all data.
        
        Returns:
        --------
        R_emp : ndarray (N, N)
            Empirical correlation matrix
        """
        if window is not None:
            if window > self.T:
                raise ValueError(f"Window {window} exceeds data length {self.T}")
            data = self.returns.iloc[-window:]
        else:
            data = self.returns
        
        R_emp = data.corr(). values
        
        # Ensure it's positive definite
        eigvals = np.linalg. eigvalsh(R_emp)
        if np.min(eigvals) < -1e-10:
            warnings.warn(f"Correlation matrix not positive definite. Min eigenvalue: {np.min(eigvals)}")
        
        return R_emp
    
    def fit_toeplitz_global(self, R_emp: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Global Toeplitz approximation - find single rho
        
        Parameters:
        -----------
        R_emp : ndarray (N, N)
            Empirical correlation matrix
        
        Returns:
        --------
        rho_opt : float
            Optimal correlation parameter
        R_toep : ndarray (N, N)
            Toeplitz approximation
        error : float
            Frobenius norm squared error
        """
        N = R_emp.shape[0]
        
        if R_emp.shape[0] != R_emp.shape[1]:
            raise ValueError("Correlation matrix must be square")
        
        def objective(rho):
            """Frobenius norm squared between empirical and Toeplitz"""
            first_row = rho ** np.arange(N)
            R_toep = toeplitz(first_row)
            return np.sum((R_emp - R_toep) ** 2)
        
        # Optimize with bounds
        result = minimize_scalar(objective, bounds=(0, 0.999), method='bounded')
        
        if not result.success:
            warnings.warn("Optimization did not converge")
        
        rho_opt = result.x
        first_row = rho_opt ** np.arange(N)
        R_toep = toeplitz(first_row)
        
        return rho_opt, R_toep, result.fun
    
    def ar1_eigenvalues(self, rho: float, N: Optional[int] = None) -> np.ndarray:
        """
        Closed-form eigenvalues for AR(1) Toeplitz matrix
        
        Formula from Ray & Driver (1970):
        lambda_k = (1-rho^2) / (1 - 2*rho*cos(theta_k) + rho^2)
        
        Parameters:
        -----------
        rho : float
            AR(1) correlation parameter
        N : int, optional
            Matrix size.  If None, use self.N
        
        Returns:
        --------
        eigenvalues : ndarray (N,)
            Eigenvalues in descending order
        """
        if N is None:
            N = self.N
        
        if not 0 <= rho < 1:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        
        k = np.arange(1, N + 1)
        theta_k = k * np.pi / (N + 1)
        
        numerator = 1 - rho**2
        denominator = 1 - 2*rho*np.cos(theta_k) + rho**2
        
        eigenvalues = numerator / denominator
        
        # Sort descending
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return eigenvalues
    
    def ar1_eigenvectors(self, N: Optional[int] = None) -> np.ndarray:
        """
        Closed-form eigenvectors for AR(1) Toeplitz matrix
        
        Formula: v_k(n) = sqrt(2/(N+1)) * sin(n*k*pi/(N+1))
        
        Parameters:
        -----------
        N : int, optional
            Matrix size. If None, use self.N
        
        Returns:
        --------
        V : ndarray (N, N)
            Eigenvector matrix (columns are eigenvectors)
        """
        if N is None:
            N = self.N
        
        n = np.arange(1, N + 1). reshape(-1, 1)  # Row indices
        k = np.arange(1, N + 1). reshape(1, -1)  # Column indices
        
        V = np.sqrt(2 / (N + 1)) * np.sin(n * k * np.pi / (N + 1))
        
        # Sort to match eigenvalue ordering (descending)
        # The eigenvectors are already ordered correctly for AR(1)
        
        return V
    
    def dct2d(self, R: np.ndarray) -> np.ndarray:
        """
        Apply 2D DCT Type-II to matrix
        
        Parameters:
        -----------
        R : ndarray (N, N)
            Input matrix
        
        Returns:
        --------
        R_dct : ndarray (N, N)
            DCT coefficients
        """
        # Apply DCT to rows, then columns
        R_dct = dct(dct(R. T, norm='ortho').T, norm='ortho')
        return R_dct
    
    def idct2d(self, R_dct: np.ndarray) -> np.ndarray:
        """
        Apply 2D inverse DCT Type-II
        
        Parameters:
        -----------
        R_dct : ndarray (N, N)
            DCT coefficients
        
        Returns:
        --------
        R : ndarray (N, N)
            Reconstructed matrix
        """
        R = idct(idct(R_dct.T, norm='ortho').T, norm='ortho')
        return R
    
    def eigenfilter(self, R: np.ndarray, Q: int, 
                    method: str = 'empirical') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigenfilter correlation matrix - keep top Q factors
        
        Parameters:
        -----------
        R : ndarray (N, N)
            Correlation matrix
        Q : int
            Number of factors to keep
        method : str
            'empirical' (full eigendecomp) or 'toeplitz' (closed-form)
        
        Returns:
        --------
        R_filtered : ndarray (N, N)
            Filtered correlation matrix
        eigenvalues : ndarray (N,)
            All eigenvalues (sorted descending)
        eigenvectors : ndarray (N, N)
            All eigenvectors
        """
        N = R.shape[0]
        
        if Q > N:
            raise ValueError(f"Q ({Q}) cannot exceed N ({N})")
        if Q < 1:
            raise ValueError(f"Q must be at least 1, got {Q}")
        
        if method == 'empirical':
            # Full eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(R)
            
            # Sort descending
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
        elif method == 'toeplitz':
            # Closed-form for AR(1)
            rho, _, _ = self.fit_toeplitz_global(R)
            eigenvalues = self.ar1_eigenvalues(rho, N)
            eigenvectors = self.ar1_eigenvectors(N)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Keep top Q factors
        eigenvalues_filtered = eigenvalues. copy()
        eigenvalues_filtered[Q:] = 0
        
        # Reconstruct
        R_filtered = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
        
        # Energy-preserving diagonal correction
        # Set diagonal to 1 (correlation property)
        diagonal_correction = 1 - np.diag(R_filtered)
        R_filtered += np.diag(diagonal_correction)
        
        return R_filtered, eigenvalues, eigenvectors
    
    def compaction_efficiency(self, eigenvalues: np.ndarray) -> float:
        """
        Compute compaction efficiency: eta_c = GM / AM
        
        Parameters:
        -----------
        eigenvalues : ndarray
            Transform coefficient variances
        
        Returns:
        --------
        eta_c : float
            Compaction efficiency in [0, 1]
        """
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
        
        if len(eigenvalues) == 0:
            return 0.0
        
        gm = np.exp(np.mean(np.log(eigenvalues)))  # Geometric mean
        am = np. mean(eigenvalues)  # Arithmetic mean
        
        eta_c = gm / am
        
        return eta_c
    
    def portfolio_risk(self, R: np.ndarray, 
                      weights: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute portfolio risk via eigendecomposition
        
        Formula: sigma_p^2 = sum_k lambda_k * (v_k^T w)^2
        
        Parameters:
        -----------
        R : ndarray (N, N)
            Correlation matrix
        weights : ndarray (N,)
            Portfolio weights
        
        Returns:
        --------
        risk : float
            Portfolio standard deviation
        risk_contrib : ndarray (N,)
            Risk contributions by factor
        eigenvalues : ndarray (N,)
            Eigenvalues
        """
        if weights.shape[0] != R.shape[0]:
            raise ValueError(f"Weight dimension {weights.shape[0]} doesn't match R dimension {R.shape[0]}")
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Factor exposures
        exposures = eigenvectors.T @ weights
        
        # Risk contributions
        risk_contrib = eigenvalues * (exposures ** 2)
        
        # Total risk
        total_risk = np.sqrt(np.sum(risk_contrib))
        
        return total_risk, risk_contrib, eigenvalues
    
    def compare_methods(self, Q: int = 5, 
                       window: Optional[int] = None) -> Dict:
        """
        Compare KLT vs DCT vs Toeplitz approximation
        
        Parameters:
        -----------
        Q : int
            Number of factors for filtering
        window : int, optional
            Data window for correlation estimation
        
        Returns:
        --------
        results : dict
            Comprehensive comparison results
        """
        print("\n" + "="*60)
        print("COMPARING KLT, DCT, AND TOEPLITZ METHODS")
        print("="*60)
        
        # Empirical correlation
        R_emp = self.empirical_correlation(window=window)
        
        # Global Toeplitz approximation
        rho_opt, R_toep, error_toep = self.fit_toeplitz_global(R_emp)
        print(f"\nOptimal ρ: {rho_opt:.4f}")
        print(f"Toeplitz approximation error: {np.sqrt(error_toep / self.N**2):.4f} (RMSE per element)")
        
        # Eigenfilter with KLT
        R_filt_klt, eig_klt, vec_klt = self.eigenfilter(R_emp, Q, 'empirical')
        eta_c_klt = self.compaction_efficiency(eig_klt)
        
        # Eigenfilter with Toeplitz
        R_filt_toep, eig_toep, vec_toep = self.eigenfilter(R_emp, Q, 'toeplitz')
        eta_c_toep = self.compaction_efficiency(eig_toep)
        
        # DCT analysis
        R_dct = self.dct2d(R_emp)
        eig_dct = np.sort(np.diag(R_dct))[::-1]  # Approximate eigenvalues from diagonal
        eta_c_dct = self. compaction_efficiency(eig_dct)
        
        # Portfolio risk comparison (equal-weighted)
        weights = np.ones(self.N) / self.N
        
        risk_emp, rc_emp, _ = self.portfolio_risk(R_emp, weights)
        risk_klt, rc_klt, _ = self.portfolio_risk(R_filt_klt, weights)
        risk_toep, rc_toep, _ = self.portfolio_risk(R_filt_toep, weights)
        
        # Variance explained
        var_exp_klt = np.cumsum(eig_klt) / np.sum(eig_klt) * 100
        var_exp_toep = np.cumsum(eig_toep) / np.sum(eig_toep) * 100
        
        # Performance ratio
        performance_ratio = eta_c_toep / eta_c_klt
        
        print(f"\nCompaction Efficiency:")
        print(f"  KLT:     {eta_c_klt:.4f}")
        print(f"  Toeplitz: {eta_c_toep:.4f}")
        print(f"  DCT:     {eta_c_dct:.4f}")
        print(f"  Toeplitz/KLT: {performance_ratio:.2%}")
        
        print(f"\nVariance Explained by {Q} factors:")
        print(f"  KLT:     {var_exp_klt[Q-1]:.2f}%")
        print(f"  Toeplitz: {var_exp_toep[Q-1]:.2f}%")
        
        print(f"\nPortfolio Risk (equal-weighted):")
        print(f"  Empirical:  {risk_emp:.4f}")
        print(f"  KLT Filter: {risk_klt:.4f}")
        print(f"  Toep Filter: {risk_toep:.4f}")
        
        results = {
            'rho_opt': rho_opt,
            'R_emp': R_emp,
            'R_toep': R_toep,
            'R_filt_klt': R_filt_klt,
            'R_filt_toep': R_filt_toep,
            'error_toeplitz': error_toep,
            'eigenvalues_klt': eig_klt,
            'eigenvalues_toeplitz': eig_toep,
            'eigenvalues_dct': eig_dct,
            'eigenvectors_klt': vec_klt,
            'eigenvectors_toeplitz': vec_toep,
            'eta_c_klt': eta_c_klt,
            'eta_c_toeplitz': eta_c_toep,
            'eta_c_dct': eta_c_dct,
            'performance_ratio': performance_ratio,
            'risk_empirical': risk_emp,
            'risk_klt': risk_klt,
            'risk_toeplitz': risk_toep,
            'risk_contrib_emp': rc_emp,
            'risk_contrib_klt': rc_klt,
            'risk_contrib_toep': rc_toep,
            'var_explained_klt': var_exp_klt,
            'var_explained_toeplitz': var_exp_toep,
            'Q': Q
        }
        
        return results
    
    def visualize_results(self, results: Dict, 
                         save_path: Optional[str] = None):
        """
        Create comprehensive visualization of results
        
        Parameters:
        -----------
        results : dict
            Results from compare_methods()
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        Q = results['Q']
        
        # 1.  Empirical vs Toeplitz correlation heatmaps
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(results['R_emp'], cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax1.set_title('Empirical Correlation', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Asset Index')
        ax1.set_ylabel('Asset Index')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(results['R_toep'], cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax2.set_title(f'Toeplitz (ρ={results["rho_opt"]:.3f})', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Asset Index')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = results['R_emp'] - results['R_toep']
        im3 = ax3. imshow(diff, cmap='seismic', vmin=-0.3, vmax=0.3)
        ax3.set_title('Approximation Error', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Asset Index')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 2.  Eigenvalue spectrum
        ax4 = fig. add_subplot(gs[1, 0])
        x = np.arange(1, self.N + 1)
        ax4.bar(x - 0.2, results['eigenvalues_klt'], width=0.4, 
               alpha=0.7, label='KLT', color='blue')
        ax4.bar(x + 0.2, results['eigenvalues_toeplitz'], width=0.4, 
               alpha=0.7, label='Toeplitz', color='green')
        ax4.axvline(Q + 0.5, color='red', linestyle='--', linewidth=2, label=f'Q={Q}')
        ax4.set_xlabel('Factor Index')
        ax4.set_ylabel('Eigenvalue')
        ax4.set_title('Eigenvalue Spectrum', fontsize=10, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0.5, self.N + 0.5)
        
        # 3. Cumulative variance explained
        ax5 = fig. add_subplot(gs[1, 1])
        ax5. plot(x, results['var_explained_klt'], 'o-', label='KLT', markersize=4)
        ax5.plot(x, results['var_explained_toeplitz'], 's-', label='Toeplitz', markersize=4)
        ax5.axvline(Q, color='red', linestyle='--', linewidth=2)
        ax5.axhline(80, color='gray', linestyle=':', linewidth=2, label='80% threshold')
        ax5.set_xlabel('Number of Factors')
        ax5.set_ylabel('Variance Explained (%)')
        ax5.set_title('Cumulative Variance Explained', fontsize=10, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, self.N + 0.5)
        ax5.set_ylim(0, 105)
        
        # 4. Risk contributions
        ax6 = fig.add_subplot(gs[1, 2])
        factors_to_plot = min(10, self.N)
        x_rc = np.arange(1, factors_to_plot + 1)
        ax6.bar(x_rc - 0.2, results['risk_contrib_klt'][:factors_to_plot], 
               width=0.4, alpha=0.7, label='KLT', color='blue')
        ax6.bar(x_rc + 0.2, results['risk_contrib_toep'][:factors_to_plot], 
               width=0.4, alpha=0.7, label='Toeplitz', color='green')
        ax6.set_xlabel('Factor Index')
        ax6.set_ylabel('Risk Contribution')
        ax6.set_title('Factor Risk Contributions (Top 10)', fontsize=10, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 5. Portfolio risk comparison
        ax7 = fig.add_subplot(gs[2, 0])
        methods = ['Empirical', 'KLT\nFiltered', 'Toeplitz\nFiltered']
        risks = [results['risk_empirical'], results['risk_klt'], results['risk_toeplitz']]
        colors = ['blue', 'green', 'orange']
        bars = ax7.bar(methods, risks, color=colors, alpha=0.7)
        ax7.set_ylabel('Portfolio Risk')
        ax7.set_title('Portfolio Risk Comparison', fontsize=10, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars:
            height = bar. get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 6. Compaction efficiency comparison
        ax8 = fig. add_subplot(gs[2, 1])
        methods_eta = ['KLT', 'Toeplitz', 'DCT']
        etas = [results['eta_c_klt'], results['eta_c_toeplitz'], results['eta_c_dct']]
        colors_eta = ['blue', 'green', 'purple']
        bars_eta = ax8.bar(methods_eta, etas, color=colors_eta, alpha=0.7)
        ax8.set_ylabel('Compaction Efficiency')
        ax8.set_title('Compaction Efficiency (η_c)', fontsize=10, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_ylim(0, max(etas) * 1.2)
        
        # Add values on bars
        for bar in bars_eta:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 7. Summary statistics
        ax9 = fig. add_subplot(gs[2, 2])
        summary_text = f"""
SUMMARY STATISTICS
{'='*35}

Correlation Structure:
  Optimal ρ: {results['rho_opt']:.4f}
  Toeplitz RMSE: {np.sqrt(results['error_toeplitz']/self.N**2):.4f}

Performance (Q={Q} factors):
  Var.  Explained (KLT): {results['var_explained_klt'][Q-1]:.2f}%
  Var. Explained (Toep): {results['var_explained_toeplitz'][Q-1]:.2f}%
  
Compaction Efficiency:
  KLT:     {results['eta_c_klt']:.4f}
  Toeplitz: {results['eta_c_toeplitz']:.4f}
  DCT:     {results['eta_c_dct']:.4f}
  Toep/KLT: {results['performance_ratio']:.2%}

Portfolio Risk (equal-weighted):
  Empirical:  {results['risk_empirical']:.4f}
  KLT Filter: {results['risk_klt']:.4f}
  Toep Filter: {results['risk_toeplitz']:.4f}

Computational Advantage:
  KLT: O(N³) = O({self.N**3:,})
  DCT: O(N log N) = O({int(self.N * np.log2(self.N)):,})
  Speedup: ~{int(self.N**2 / np.log2(self.N)):,}×
        """
        
        ax9.text(0.05, 0.95, summary_text,
                transform=ax9.transAxes,
                fontsize=8.5, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax9.axis('off')
        
        plt.suptitle('Toeplitz/DCT Analysis: Paper 1 (Akansu & Torun, 2012)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt. savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        return fig


def demo_paper1():
    """Demonstration of Paper 1 implementation"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("\n" + "="*70)
    print(" PAPER 1: TOEPLITZ APPROXIMATION - DEMONSTRATION")
    print("="*70)
    
    # Download data
    print("\nDownloading data...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'JPM', 'BAC', 'GS', 'C', 'WFC',
               'XOM', 'CVX', 'COP', 'SLB', 'MPC']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False,auto_adjust=False)
    prices = data['Adj Close']
    
    # Compute returns
    returns = np.log(prices / prices.shift(1)). dropna()
    
    # Normalize
    returns_normalized = (returns - returns.mean()) / returns.std()
    
    print(f"Data: {len(returns)} days, {len(tickers)} assets")
    
    # Initialize analyzer
    analyzer = ToeplitzDCTAnalyzer(returns_normalized)
    
    # Run comparison
    results = analyzer.compare_methods(Q=5, window=252)  # 1 year window
    
    # Visualize
    fig = analyzer.visualize_results(results, save_path='paper1_results.png')
    plt.show()
    
    print("\n" + "="*70)
    print(" PAPER 1 DEMONSTRATION COMPLETE")
    print("="*70)
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = demo_paper1()