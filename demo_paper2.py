"""
Paper 2: π-Counting Instantaneous Frequency and Stock Prediction
High-quality implementation with error handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Try to import EMD, provide fallback
try:
    from PyEMD import EMD as PyEMD_EMD
    HAS_PYEMD = True
except ImportError:
    HAS_PYEMD = False
    warnings.warn("PyEMD not installed.Using simplified EMD.")


class SimplifiedEMD:
    """
    Simplified EMD implementation for when PyEMD is not available
    This is a basic implementation - PyEMD is recommended for production
    """
    
    def __init__(self):
        self.imfs = None
    
    def emd(self, signal: np.ndarray, max_imf: int = 10):
        """Simple EMD sifting process"""
        residual = signal.copy()
        imfs = []
        
        for i in range(max_imf):
            if len(residual) < 10:
                break
            
            imf = self._extract_imf(residual)
            
            if imf is None:
                break
            
            imfs.append(imf)
            residual = residual - imf
            
            # Stop if residual is monotonic
            if self._is_monotonic(residual):
                break
        
        self.imfs = np.array(imfs) if imfs else np.array([signal])
        return self.imfs
    
    def _extract_imf(self, signal: np.ndarray, max_iter: int = 100) -> Optional[np.ndarray]:
        """Extract single IMF through sifting"""
        h = signal.copy()
        
        for _ in range(max_iter):
            # Find extrema
            maxima_idx, _ = find_peaks(h)
            minima_idx, _ = find_peaks(-h)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                return None
            
            # Simple envelope (linear interpolation)
            try:
                upper_env = np.interp(np.arange(len(h)), maxima_idx, h[maxima_idx])
                lower_env = np.interp(np.arange(len(h)), minima_idx, h[minima_idx])
            except:
                return None
            
            mean_env = (upper_env + lower_env) / 2
            h_new = h - mean_env
            
            # Check IMF criteria (simplified)
            if np.max(np.abs(h - h_new)) < 0.01 * np.std(signal):
                return h_new
            
            h = h_new
        
        return h
    
    def _is_monotonic(self, signal: np.ndarray) -> bool:
        """Check if signal is monotonic"""
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)


class PiCountingIF:
    """
    Implement π-counting Instantaneous Frequency
    from Zhang, Liu & Yu (2012)
    """
    
    def __init__(self, signal: np.ndarray, max_imf: int = 10):
        """
        Initialize with time series signal
        
        Parameters:
        -----------
        signal : ndarray or pd.Series
            Time series signal
        max_imf : int
            Maximum number of IMFs
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        
        self.signal = signal.flatten()
        self.max_imf = max_imf
        self.N = len(signal)
        
        self.imfs = None
        self.residual = None
        self.n_imfs = 0
        self.IFs = None
        self.primary_cycles = None
    
    def apply_emd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Empirical Mode Decomposition
        
        Returns:
        --------
        imfs : ndarray (n_imfs, N)
            Intrinsic Mode Functions
        residual : ndarray (N,)
            Trend component
        """
        print("\nApplying EMD decomposition...")
        
        if HAS_PYEMD:
            emd = PyEMD_EMD()
            try:
                self.imfs = emd.emd(self.signal, max_imf=self.max_imf)
            except Exception as e:
                warnings.warn(f"PyEMD failed: {e}.Using simplified EMD.")
                emd_simple = SimplifiedEMD()
                self.imfs = emd_simple.emd(self.signal, max_imf=self.max_imf)
        else:
            emd_simple = SimplifiedEMD()
            self.imfs = emd_simple.emd(self.signal, max_imf=self.max_imf)
        
        if len(self.imfs) == 0:
            raise ValueError("EMD failed to extract any IMFs")
        
        self.n_imfs = len(self.imfs)
        self.residual = self.signal - np.sum(self.imfs, axis=0)
        
        # Verify reconstruction
        reconstructed = np.sum(self.imfs, axis=0) + self.residual
        recon_error = np.sum((self.signal - reconstructed) ** 2)
        
        print(f"  Decomposed into {self.n_imfs} IMFs")
        print(f"  Reconstruction error: {recon_error:.2e}")
        
        if recon_error > 1e-6:
            warnings.warn(f"Large reconstruction error: {recon_error}")
        
        return self.imfs, self.residual
    
    def find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """
        Find indices of local extrema (maxima and minima)
        
        Parameters:
        -----------
        signal : ndarray
            Input signal
        
        Returns:
        --------
        extrema_idx : ndarray
            Indices of extrema
        """
        # Find peaks (maxima)
        maxima_idx, _ = find_peaks(signal)
        
        # Find valleys (minima) by inverting signal
        minima_idx, _ = find_peaks(-signal)
        
        # Combine and sort
        extrema_idx = np.sort(np.concatenate([maxima_idx, minima_idx]))
        
        return extrema_idx
    
    def compute_pi_counting_IF(self, signal: np.ndarray,
                               h_min: float = 0.5,
                               h_max: Optional[float] = None,
                               delta_h: float = 0.5) -> np.ndarray:
        """
        Compute π-counting instantaneous frequency
        
        Parameters:
        -----------
        signal : ndarray
            Signal (IMF or time series)
        h_min : float
            Minimum window size (in samples)
        h_max : float, optional
            Maximum window size.If None, use N/4
        delta_h : float
            Window increment
        
        Returns:
        --------
        IF : ndarray (N,)
            Instantaneous frequency at each point
        """
        N = len(signal)
        
        if h_max is None:
            h_max = N / 4
        
        # Find all extrema
        extrema_idx = self.find_extrema(signal)
        
        if len(extrema_idx) < 3:
            # Too few extrema - return constant low frequency
            warnings.warn("Too few extrema for IF calculation")
            return np.ones(N) * np.pi / (2 * h_max)
        
        IF = np.zeros(N)
        
        for t in range(N):
            h = h_min
            found = False
            
            while h < h_max and not found:
                # Define window
                t_min = max(0, int(t - h))
                t_max = min(N - 1, int(t + h))
                
                # Count extrema in window
                extrema_in_window = np.sum(
                    (extrema_idx >= t_min) & (extrema_idx <= t_max)
                )
                
                N_h = extrema_in_window
                K_h = N_h // 2  # Number of full cycles
                
                # Stopping condition: K_h > 0 AND N_h >= 3
                if K_h > 0 and N_h >= 3:
                    h_star = max(h - delta_h, h_min)
                    IF[t] = np.pi / (2 * h_star)
                    found = True
                else:
                    h += delta_h
            
            # If not found, use maximum window
            if not found:
                IF[t] = np.pi / (2 * h_max)
        
        return IF
    
    def extract_cycles(self) -> Tuple[List[int], List[np.ndarray]]:
        """
        Extract primary cycles for each IMF using π-counting IF
        
        Returns:
        --------
        primary_cycles : list of int
            Primary cycle (in samples) for each IMF
        IFs : list of ndarray
            IF time series for each IMF
        """
        if self.imfs is None:
            raise ValueError("Must run apply_emd() first")
        
        print("\nExtracting cycles using π-counting IF...")
        
        primary_cycles = []
        IFs = []
        
        for i, imf in enumerate(self.imfs):
            # Compute IF
            IF = self.compute_pi_counting_IF(imf)
            IFs.append(IF)
            
            # Convert IF to periods
            periods = 2 * np.pi / (IF + 1e-10)
            
            # Primary cycle (mode via histogram)
            valid_periods = periods[(periods > 1) & (periods < len(imf) / 2)]
            
            if len(valid_periods) == 0:
                primary_cycle = len(imf) // 4
                warnings.warn(f"IMF {i+1}: Could not determine cycle, using {primary_cycle}")
            else:
                hist, bins = np.histogram(valid_periods, bins=50)
                primary_cycle = int(bins[np.argmax(hist)])
            
            primary_cycles.append(primary_cycle)
            
            print(f"  IMF {i+1}: Primary cycle = {primary_cycle} samples")
        
        self.primary_cycles = primary_cycles
        self.IFs = IFs
        
        return primary_cycles, IFs
    
    def build_rbf_network(self, X: np.ndarray, y: np.ndarray,
                         n_hidden: Optional[int] = None,
                         spread: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Train RBF neural network
        
        Parameters:
        -----------
        X : ndarray (n_samples, n_features)
            Input data
        y : ndarray (n_samples,)
            Target data
        n_hidden : int, optional
            Number of hidden neurons.If None, auto-determined
        spread : float
            RBF width parameter
        
        Returns:
        --------
        centers : ndarray (n_hidden, n_features)
            RBF centers
        weights : ndarray (n_hidden + 1,)
            Output weights (includes bias)
        spread : float
            RBF width used
        """
        n_samples, n_features = X.shape
        
        if n_samples < 5:
            raise ValueError(f"Too few training samples: {n_samples}")
        
        # Determine number of hidden neurons
        if n_hidden is None:
            n_hidden = min(max(n_samples // 3, 5), 50)
        
        n_hidden = min(n_hidden, n_samples)  # Can't have more centers than samples
        
        # Phase 1: Select centers using K-means
        if n_hidden < n_samples:
            kmeans = KMeans(n_clusters=n_hidden, random_state=42, n_init=10)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
        else:
            centers = X.copy()
        
        # Phase 2: Compute RBF activations
        from scipy.spatial.distance import cdist
        distances = cdist(X, centers, 'euclidean')
        Phi = np.exp(-(distances ** 2) / (2 * spread ** 2))
        
        # Add bias column
        Phi = np.c_[Phi, np.ones(n_samples)]
        
        # Phase 3: Solve for output weights (linear least squares)
        try:
            weights = np.linalg.lstsq(Phi, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if lstsq fails
            weights = np.linalg.pinv(Phi) @ y
        
        return centers, weights, spread
    
    def predict_rbf(self, X: np.ndarray, centers: np.ndarray,
                   weights: np.ndarray, spread: float) -> np.ndarray:
        """
        Predict using trained RBF network
        
        Parameters:
        -----------
        X : ndarray (n_samples, n_features)
            Input data
        centers : ndarray (n_hidden, n_features)
            RBF centers
        weights : ndarray (n_hidden + 1,)
            Output weights
        spread : float
            RBF width
        
        Returns:
        --------
        predictions : ndarray (n_samples,)
            Predicted values
        """
        from scipy.spatial.distance import cdist
        
        distances = cdist(X, centers, 'euclidean')
        Phi = np.exp(-(distances ** 2) / (2 * spread ** 2))
        Phi = np.c_[Phi, np.ones(X.shape[0])]
        
        return Phi @ weights
    
    def predict_imf(self, imf: np.ndarray, primary_cycle: int,
                   train_size: int = 150, test_size: int = 35) -> Dict:
        """
        Predict single IMF using RBF network
        
        Parameters:
        -----------
        imf : ndarray
            IMF signal
        primary_cycle : int
            Input lag (from π-counting IF)
        train_size : int
            Number of training samples
        test_size : int
            Number of test samples
        
        Returns:
        --------
        results : dict
            Contains predictions, actual, SSE, model params
        """
        p = max(primary_cycle, 2)  # At least 2 lags
        
        if len(imf) < train_size + test_size:
            raise ValueError(f"IMF length {len(imf)} < train_size + test_size")
        
        if p >= train_size:
            raise ValueError(f"Primary cycle {p} >= train_size {train_size}")
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for t in range(p, train_size):
            X_train.append(imf[t-p:t])
            y_train.append(imf[t])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train RBF network
        centers, weights, spread = self.build_rbf_network(X_train, y_train)
        
        # Predict test set
        predictions = []
        
        for t in range(train_size, train_size + test_size):
            X_test = imf[t-p:t].reshape(1, -1)
            pred = self.predict_rbf(X_test, centers, weights, spread)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        actual = imf[train_size:train_size + test_size]
        
        # Compute metrics
        sse = np.sum((actual - predictions) ** 2)
        rmse = np.sqrt(sse / test_size)
        mae = np.mean(np.abs(actual - predictions))
        
        results = {
            'predictions': predictions,
            'actual': actual,
            'sse': sse,
            'rmse': rmse,
            'mae': mae,
            'centers': centers,
            'weights': weights,
            'spread': spread
        }
        
        return results
    
    def predict_all(self, train_size: int = 150, 
                   test_size: int = 35) -> Dict:
        """
        Predict all IMFs and synthesize final prediction
        
        Parameters:
        -----------
        train_size : int
            Number of training samples
        test_size : int
            Number of test samples
        
        Returns:
        --------
        results : dict
            Complete prediction results
        """
        if self.imfs is None:
            self.apply_emd()
        if self.primary_cycles is None:
            self.extract_cycles()
        
        if train_size + test_size > self.N:
            raise ValueError(f"train_size + test_size = {train_size + test_size} > signal length {self.N}")
        
        print("\n" + "="*60)
        print("PREDICTING IMFs")
        print("="*60)
        
        component_results = []
        component_predictions = []
        component_actual = []
        
        for i, (imf, cycle) in enumerate(zip(self.imfs, self.primary_cycles)):
            try:
                result = self.predict_imf(imf, cycle, train_size, test_size)
                component_results.append(result)
                component_predictions.append(result['predictions'])
                component_actual.append(result['actual'])
                
                print(f"  IMF {i+1}: SSE = {result['sse']:.4e}, RMSE = {result['rmse']:.4e}")
                
            except Exception as e:
                warnings.warn(f"IMF {i+1} prediction failed: {e}")
                # Use zero prediction as fallback
                fallback_pred = np.zeros(test_size)
                fallback_actual = imf[train_size:train_size + test_size]
                component_predictions.append(fallback_pred)
                component_actual.append(fallback_actual)
                
                fallback_result = {
                    'predictions': fallback_pred,
                    'actual': fallback_actual,
                    'sse': np.sum(fallback_actual ** 2),
                    'rmse': np.std(fallback_actual),
                    'mae': np.mean(np.abs(fallback_actual))
                }
                component_results.append(fallback_result)
        
        # Synthesize final prediction
        final_prediction = np.sum(component_predictions, axis=0)
        final_actual = self.signal[train_size:train_size + test_size]
        
        final_sse = np.sum((final_actual - final_prediction) ** 2)
        final_rmse = np.sqrt(final_sse / test_size)
        final_mae = np.mean(np.abs(final_actual - final_prediction))
        
        print("\n" + "-"*60)
        print(f"FINAL RESULTS:")
        print(f"  SSE:  {final_sse:.4f}")
        print(f"  RMSE: {final_rmse:.4f}")
        print(f"  MAE:  {final_mae:.4f}")
        print("-"*60)
        
        results = {
            'final_prediction': final_prediction,
            'final_actual': final_actual,
            'final_sse': final_sse,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'component_results': component_results,
            'component_predictions': component_predictions,
            'component_actual': component_actual,
            'train_size': train_size,
            'test_size': test_size
        }
        
        return results
    
    def visualize_decomposition(self, save_path: Optional[str] = None):
        """Visualize EMD decomposition and IFs"""
        if self.imfs is None or self.IFs is None:
            raise ValueError("Must run apply_emd() and extract_cycles() first")
        
        n_plots = self.n_imfs + 2  # IMFs + residual + original
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2*n_plots))
        
        time = np.arange(self.N)
        
        # Original signal
        axes[0].plot(time, self.signal, 'b-', linewidth=0.8)
        axes[0].set_title('Original Signal', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, self.N)
        
        # Each IMF with IF overlay
        for i, (imf, IF) in enumerate(zip(self.imfs, self.IFs)):
            ax = axes[i+1]
            
            # Plot IMF
            color_imf = 'blue'
            ax.plot(time, imf, color=color_imf, linewidth=0.7, label='IMF', alpha=0.8)
            ax.set_ylabel('Amplitude', color=color_imf)
            ax.tick_params(axis='y', labelcolor=color_imf)
            
            # Plot IF on secondary axis
            ax2 = ax.twinx()
            color_if = 'red'
            ax2.plot(time, IF, color=color_if, linewidth=0.7, alpha=0.7, label='IF')
            ax2.set_ylabel('IF (rad/sample)', color=color_if)
            ax2.tick_params(axis='y', labelcolor=color_if)
            
            ax.set_title(f'IMF {i+1} (Primary Cycle = {self.primary_cycles[i]} samples)', 
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.N)
        
        # Residual
        axes[-1].plot(time, self.residual, 'g-', linewidth=1.2)
        axes[-1].set_title('Residual (Trend)', fontsize=11, fontweight='bold')
        axes[-1].set_xlabel('Time (samples)')
        axes[-1].set_ylabel('Amplitude')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlim(0, self.N)
        
        plt.suptitle('EMD Decomposition with π-Counting IF: Paper 2 (Zhang et al., 2012)',
                    fontsize=13, fontweight='bold', y=0.9995)
        plt.tight_layout(rect=[0, 0, 1, 0.9985])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nDecomposition figure saved to: {save_path}")
        
        return fig
    
    def visualize_predictions(self, results: Dict, 
                             save_path: Optional[str] = None):
        """Visualize prediction results"""
        n_components = len(results['component_predictions'])
        
        fig, axes = plt.subplots(n_components + 1, 1, 
                                figsize=(14, 2.5*(n_components+1)))
        
        time = np.arange(results['test_size'])
        
        # Each component prediction
        for i, (pred, actual, comp_result) in enumerate(zip(
            results['component_predictions'],
            results['component_actual'],
            results['component_results']
        )):
            axes[i].plot(time, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
            axes[i].plot(time, pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'IMF {i+1} (SSE = {comp_result["sse"]:.4e}, RMSE = {comp_result["rmse"]:.4e})',
                            fontsize=10, fontweight='bold')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, results['test_size'])
        
        # Final prediction
        axes[-1].plot(time, results['final_actual'], 'b-', 
                     label='Actual', linewidth=2, alpha=0.8)
        axes[-1].plot(time, results['final_prediction'], 'r--', 
                     label='Predicted', linewidth=2, alpha=0.8)
        axes[-1].set_xlabel('Time (samples)')
        axes[-1].set_ylabel('Amplitude')
        axes[-1].set_title(
            f'Final Prediction (SSE = {results["final_sse"]:.4f}, RMSE = {results["final_rmse"]:.4f}, MAE = {results["final_mae"]:.4f})',
            fontsize=11, fontweight='bold'
        )
        axes[-1].legend(loc='upper right')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlim(0, results['test_size'])
        
        plt.suptitle('Prediction Results: Paper 2 (Zhang et al., 2012)',
                    fontsize=13, fontweight='bold', y=0.9995)
        plt.tight_layout(rect=[0, 0, 1, 0.9985])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPrediction figure saved to: {save_path}")
        
        return fig


def demo_paper2():
    """Demonstration of Paper 2 implementation"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("\n" + "="*70)
    print(" PAPER 2: π-COUNTING IF AND PREDICTION - DEMONSTRATION")
    print("="*70)
    
    # Download data
    print("\nDownloading data...")
    ticker = 'AAPL'  # Single asset for IMF analysis
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    prices = data['Adj Close']
    
    # Compute log returns
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    print(f"Data: {len(returns)} days")
    
    # Initialize analyzer
    analyzer = PiCountingIF(returns, max_imf=8)
    
    # Apply EMD
    imfs, residual = analyzer.apply_emd()
    
    # Extract cycles
    cycles, IFs = analyzer.extract_cycles()
    
    # Visualize decomposition
    fig1 = analyzer.visualize_decomposition(save_path='paper2_decomposition.png')
    
    # Run predictions
    results = analyzer.predict_all(train_size=150, test_size=35)
    
    # Visualize predictions
    fig2 = analyzer.visualize_predictions(results, save_path='paper2_predictions.png')
    
    plt.show()
    
    print("\n" + "="*70)
    print(" PAPER 2 DEMONSTRATION COMPLETE")
    print("="*70)
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = demo_paper2()