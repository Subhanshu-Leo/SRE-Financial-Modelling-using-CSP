"""
Complete Integrated System - All Three Papers
Single file with all dependencies included
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from scipy.fftpack import dct, idct
from scipy.optimize import minimize_scalar
from scipy.linalg import toeplitz
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# PAPER 1: TOEPLITZ/DCT ANALYZER
# ============================================================================

class ToeplitzDCTAnalyzer:
    """Paper 1: Toeplitz Approximation and DCT"""
    
    def __init__(self, returns: pd.DataFrame):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        
        self.returns = returns.dropna()
        self.N = returns.shape[1]
        self.T = returns.shape[0]
        self.asset_names = returns.columns.tolist()
        
        print(f"Initialized with {self.N} assets and {self.T} observations")
    
    def empirical_correlation(self, window: Optional[int] = None) -> np.ndarray:
        if window is not None:
            data = self.returns.iloc[-window:]
        else:
            data = self.returns
        return data.corr().values
    
    def fit_toeplitz_global(self, R_emp: np.ndarray) -> Tuple[float, np.ndarray, float]:
        N = R_emp.shape[0]
        
        def objective(rho):
            first_row = rho ** np.arange(N)
            R_toep = toeplitz(first_row)
            return np.sum((R_emp - R_toep) ** 2)
        
        result = minimize_scalar(objective, bounds=(0, 0.999), method='bounded')
        rho_opt = result.x
        first_row = rho_opt ** np.arange(N)
        R_toep = toeplitz(first_row)
        
        return rho_opt, R_toep, result.fun
    
    def ar1_eigenvalues(self, rho: float, N: Optional[int] = None) -> np.ndarray:
        if N is None:
            N = self.N
        
        k = np.arange(1, N + 1)
        theta_k = k * np.pi / (N + 1)
        
        numerator = 1 - rho**2
        denominator = 1 - 2*rho*np.cos(theta_k) + rho**2
        eigenvalues = numerator / denominator
        
        return np.sort(eigenvalues)[::-1]
    
    def ar1_eigenvectors(self, N: Optional[int] = None) -> np.ndarray:
        if N is None:
            N = self.N
        
        n = np.arange(1, N + 1).reshape(-1, 1)
        k = np.arange(1, N + 1).reshape(1, -1)
        V = np.sqrt(2 / (N + 1)) * np.sin(n * k * np.pi / (N + 1))
        
        return V
    
    def eigenfilter(self, R: np.ndarray, Q: int, 
                    method: str = 'empirical') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = R.shape[0]
        
        if method == 'empirical':
            eigenvalues, eigenvectors = np.linalg.eigh(R)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        elif method == 'toeplitz':
            rho, _, _ = self.fit_toeplitz_global(R)
            eigenvalues = self.ar1_eigenvalues(rho, N)
            eigenvectors = self.ar1_eigenvectors(N)
        
        eigenvalues_filtered = eigenvalues.copy()
        eigenvalues_filtered[Q:] = 0
        
        R_filtered = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
        diagonal_correction = 1 - np.diag(R_filtered)
        R_filtered += np.diag(diagonal_correction)
        
        return R_filtered, eigenvalues, eigenvectors


# ============================================================================
# PAPER 2: EMD + π-COUNTING IF
# ============================================================================

class SimplifiedEMD:
    """Simplified EMD when PyEMD not available"""
    
    def __init__(self):
        self.imfs = None
    
    def emd(self, signal: np.ndarray, max_imf: int = 10):
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
            
            if self._is_monotonic(residual):
                break
        
        self.imfs = np.array(imfs) if imfs else np.array([signal])
        return self.imfs
    
    def _extract_imf(self, signal: np.ndarray, max_iter: int = 100) -> Optional[np.ndarray]:
        h = signal.copy()
        
        for _ in range(max_iter):
            maxima_idx, _ = find_peaks(h)
            minima_idx, _ = find_peaks(-h)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                return None
            
            try:
                upper_env = np.interp(np.arange(len(h)), maxima_idx, h[maxima_idx])
                lower_env = np.interp(np.arange(len(h)), minima_idx, h[minima_idx])
            except:
                return None
            
            mean_env = (upper_env + lower_env) / 2
            h_new = h - mean_env
            
            if np.max(np.abs(h - h_new)) < 0.01 * np.std(signal):
                return h_new
            
            h = h_new
        
        return h
    
    def _is_monotonic(self, signal: np.ndarray) -> bool:
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)


class PiCountingIF:
    """Paper 2: π-Counting Instantaneous Frequency"""
    
    def __init__(self, signal: np.ndarray, max_imf: int = 10):
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
        try:
            from PyEMD import EMD as PyEMD_EMD
            emd = PyEMD_EMD()
            self.imfs = emd.emd(self.signal, max_imf=self.max_imf)
        except:
            emd = SimplifiedEMD()
            self.imfs = emd.emd(self.signal, max_imf=self.max_imf)
        
        self.n_imfs = len(self.imfs)
        self.residual = self.signal - np.sum(self.imfs, axis=0)
        
        return self.imfs, self.residual
    
    def find_extrema(self, signal: np.ndarray) -> np.ndarray:
        maxima_idx, _ = find_peaks(signal)
        minima_idx, _ = find_peaks(-signal)
        extrema_idx = np.sort(np.concatenate([maxima_idx, minima_idx]))
        return extrema_idx
    
    def compute_pi_counting_IF(self, signal: np.ndarray,
                               h_min: float = 0.5,
                               h_max: Optional[float] = None,
                               delta_h: float = 0.5) -> np.ndarray:
        N = len(signal)
        if h_max is None:
            h_max = N / 4
        
        extrema_idx = self.find_extrema(signal)
        if len(extrema_idx) < 3:
            return np.ones(N) * np.pi / (2 * h_max)
        
        IF = np.zeros(N)
        
        for t in range(N):
            h = h_min
            found = False
            
            while h < h_max and not found:
                t_min = max(0, int(t - h))
                t_max = min(N - 1, int(t + h))
                
                extrema_in_window = np.sum(
                    (extrema_idx >= t_min) & (extrema_idx <= t_max)
                )
                
                N_h = extrema_in_window
                K_h = N_h // 2
                
                if K_h > 0 and N_h >= 3:
                    h_star = max(h - delta_h, h_min)
                    IF[t] = np.pi / (2 * h_star)
                    found = True
                else:
                    h += delta_h
            
            if not found:
                IF[t] = np.pi / (2 * h_max)
        
        return IF
    
    def extract_cycles(self) -> Tuple[List[int], List[np.ndarray]]:
        if self.imfs is None:
            raise ValueError("Must run apply_emd() first")
        
        primary_cycles = []
        IFs = []
        
        for i, imf in enumerate(self.imfs):
            IF = self.compute_pi_counting_IF(imf)
            IFs.append(IF)
            
            periods = 2 * np.pi / (IF + 1e-10)
            valid_periods = periods[(periods > 1) & (periods < len(imf) / 2)]
            
            if len(valid_periods) == 0:
                primary_cycle = len(imf) // 4
            else:
                hist, bins = np.histogram(valid_periods, bins=50)
                primary_cycle = int(bins[np.argmax(hist)])
            
            primary_cycles.append(primary_cycle)
        
        self.primary_cycles = primary_cycles
        self.IFs = IFs
        
        return primary_cycles, IFs


# ============================================================================
# PAPER 3: LANGEVIN PARTICLE FILTER
# ============================================================================

class LangevinParticleFilter:
    """Paper 3: Langevin Dynamics with Particle Filter"""
    
    def __init__(self, prices: np.ndarray, params: Optional[Dict] = None):
        if isinstance(prices, pd.Series):
            self.price_index = prices.index
            prices = prices.values
        else:
            self.price_index = None
        
        self.prices = prices.flatten()
        self.N_samples = len(self.prices)
        
        price_scale = np.mean(self.prices)
        
        if params is None:
            params = {}
        
        self.alpha = params.get('alpha', 0.1)
        self.sigma_theta = params.get('sigma_theta', 0.01 * price_scale)
        self.lambda_jump = params.get('lambda', 5.0)
        self.mu_J = params.get('mu_J', 0.0)
        self.sigma_J = params.get('sigma_J', 0.05 * price_scale)
        self.sigma_y = params.get('sigma_y', 0.001 * price_scale)
        
        self.A = np.array([[0, 1], [0, -self.alpha]])
        self.B = np.array([[0], [self.sigma_theta]])
        self.H = np.array([[1, 0]])
        
        self.state_estimates = None
        self.trend_estimates = None
    
    def matrix_exponential(self, dt: float) -> np.ndarray:
        alpha = self.alpha
        if alpha < 1e-10:
            return np.array([[1, dt], [0, 1]])
        else:
            exp_neg_alpha_t = np.exp(-alpha * dt)
            return np.array([[1, (1 - exp_neg_alpha_t) / alpha],
                           [0, exp_neg_alpha_t]])
    
    def transition_covariance(self, dt: float) -> np.ndarray:
        alpha = self.alpha
        sigma_theta = self.sigma_theta
        
        if alpha < 1e-10:
            Sigma_22 = sigma_theta**2 * dt
            Sigma_12 = sigma_theta**2 * dt**2 / 2
            Sigma_11 = sigma_theta**2 * dt**3 / 3
        else:
            exp_neg_alpha_t = np.exp(-alpha * dt)
            exp_neg_2alpha_t = np.exp(-2 * alpha * dt)
            
            Sigma_22 = (sigma_theta**2 / (2 * alpha)) * (1 - exp_neg_2alpha_t)
            Sigma_12 = (sigma_theta**2 / (2 * alpha**2)) * (1 - exp_neg_alpha_t)**2
            Sigma_11 = (sigma_theta**2 / (2 * alpha**3)) * \
                       (2*alpha*dt - 3 + 4*exp_neg_alpha_t - exp_neg_2alpha_t)
        
        return np.array([[Sigma_11, Sigma_12], [Sigma_12, Sigma_22]])
    
    def transition_with_jumps(self, x_n: np.ndarray, dt: float,
                             jump_times: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        if len(jump_times) == 0:
            exp_Adt = self.matrix_exponential(dt)
            mu = exp_Adt @ x_n
            Sigma = self.transition_covariance(dt)
        else:
            mu = x_n.copy()
            Sigma = np.zeros((2, 2))
            t_prev = 0.0
            
            for tau in sorted(jump_times):
                if tau <= 0 or tau >= dt:
                    continue
                
                dt_pre = tau - t_prev
                if dt_pre > 0:
                    exp_Adt_pre = self.matrix_exponential(dt_pre)
                    Q_pre = self.transition_covariance(dt_pre)
                    mu = exp_Adt_pre @ mu
                    Sigma = exp_Adt_pre @ Sigma @ exp_Adt_pre.T + Q_pre
                
                mu = mu + np.array([0, self.mu_J])
                Sigma = Sigma + np.array([[0, 0], [0, self.sigma_J**2]])
                t_prev = tau
            
            dt_post = dt - t_prev
            if dt_post > 0:
                exp_Adt_post = self.matrix_exponential(dt_post)
                Q_post = self.transition_covariance(dt_post)
                mu = exp_Adt_post @ mu
                Sigma = exp_Adt_post @ Sigma @ exp_Adt_post.T + Q_post
        
        return mu, Sigma
    
    def kalman_predict(self, m_n: np.ndarray, Sigma_n: np.ndarray,
                      dt: float, jump_times: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        return self.transition_with_jumps(m_n, dt, jump_times)
    
    def kalman_update(self, m_pred: np.ndarray, Sigma_pred: np.ndarray,
                     y_n: float) -> Tuple[np.ndarray, np.ndarray, float]:
        y_pred = (self.H @ m_pred)[0]
        S = (self.H @ Sigma_pred @ self.H.T)[0, 0] + self.sigma_y**2
        S = max(S, 1e-10)
        
        K = (Sigma_pred @ self.H.T) / S
        K = K.reshape(-1)
        
        innovation = y_n - y_pred
        m_updated = m_pred + K * innovation
        Sigma_updated = Sigma_pred - np.outer(K, K) * S
        
        Sigma_updated = (Sigma_updated + Sigma_updated.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma_updated)
        if np.min(eigvals) < 0:
            Sigma_updated += np.eye(2) * (1e-6 - np.min(eigvals))
        
        likelihood = norm.pdf(innovation, 0, np.sqrt(S))
        
        return m_updated, Sigma_updated, likelihood
    
    def propose_jump_times(self, dt: float) -> List[float]:
        jump_times = []
        t = 0.0
        
        while True:
            if self.lambda_jump > 0:
                inter_arrival = np.random.exponential(1.0 / self.lambda_jump)
            else:
                break
            
            t_jump = t + inter_arrival
            if t_jump < dt:
                jump_times.append(t_jump)
                t = t_jump
            else:
                break
        
        return jump_times
    
    def particle_filter(self, N_particles: int = 100,
                       dt: float = 1.0,
                       show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        
        particles = []
        for i in range(N_particles):
            particle = {
                'jump_times_full': [],
                'm': np.array([self.prices[0], 0.0]),
                'Sigma': np.eye(2) * 100.0,
                'weight': 1.0 / N_particles
            }
            particles.append(particle)
        
        state_estimates = np.zeros((self.N_samples, 2))
        trend_estimates = np.zeros(self.N_samples)
        
        state_estimates[0] = np.array([self.prices[0], 0.0])
        trend_estimates[0] = 0.0
        
        iterator = range(1, self.N_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering")
        
        for n in iterator:
            observation = self.prices[n]
            new_particles = []
            
            for particle in particles:
                ideal_offspring = N_particles * particle['weight']
                n_offspring = int(np.floor(ideal_offspring))
                
                if np.random.rand() < (ideal_offspring - n_offspring):
                    n_offspring += 1
                
                if n_offspring == 0:
                    continue
                
                for _ in range(n_offspring):
                    new_jumps = self.propose_jump_times(dt)
                    
                    offspring = {
                        'jump_times_full': particle['jump_times_full'] + new_jumps,
                        'jump_times_interval': new_jumps,
                        'm': particle['m'].copy(),
                        'Sigma': particle['Sigma'].copy(),
                        'weight': particle['weight'] / n_offspring,
                        'jumped': len(new_jumps) > 0
                    }
                    
                    new_particles.append(offspring)
            
            if len(new_particles) == 0:
                new_particles = []
                for i in range(N_particles):
                    particle = {
                        'jump_times_full': [],
                        'jump_times_interval': [],
                        'm': state_estimates[n-1].copy(),
                        'Sigma': np.eye(2) * 100.0,
                        'weight': 1.0 / N_particles,
                        'jumped': False
                    }
                    new_particles.append(particle)
            
            jumping = [p for p in new_particles if p['jumped']]
            non_jumping = [p for p in new_particles if not p['jumped']]
            
            if len(non_jumping) > 0:
                merged = {
                    'jump_times_full': non_jumping[0]['jump_times_full'],
                    'jump_times_interval': [],
                    'm': non_jumping[0]['m'].copy(),
                    'Sigma': non_jumping[0]['Sigma'].copy(),
                    'weight': sum(p['weight'] for p in non_jumping),
                    'jumped': False
                }
                particles = jumping + [merged]
            else:
                particles = jumping
            
            for particle in particles:
                try:
                    m_pred, Sigma_pred = self.kalman_predict(
                        particle['m'],
                        particle['Sigma'],
                        dt,
                        particle['jump_times_interval']
                    )
                    
                    m_updated, Sigma_updated, likelihood = self.kalman_update(
                        m_pred,
                        Sigma_pred,
                        observation
                    )
                    
                    particle['m'] = m_updated
                    particle['Sigma'] = Sigma_updated
                    particle['weight'] *= max(likelihood, 1e-300)
                    
                except:
                    particle['weight'] = 1e-300
            
            total_weight = sum(p['weight'] for p in particles)
            
            if total_weight < 1e-300:
                for particle in particles:
                    particle['weight'] = 1.0 / len(particles)
            else:
                for particle in particles:
                    particle['weight'] /= total_weight
            
            state_mean = np.zeros(2)
            for particle in particles:
                state_mean += particle['weight'] * particle['m']
            
            state_estimates[n] = state_mean
            trend_estimates[n] = state_mean[1]
        
        self.state_estimates = state_estimates
        self.trend_estimates = trend_estimates
        
        return state_estimates, trend_estimates
    
    def generate_trading_signals(self, trend_estimates: Optional[np.ndarray] = None,
                                smoothing_window: int = 4) -> np.ndarray:
        if trend_estimates is None:
            trend_estimates = self.trend_estimates
        
        trend_diff = np.diff(trend_estimates, prepend=trend_estimates[0])
        raw_signals = np.sign(trend_diff)
        
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed_signals = np.convolve(raw_signals, kernel, mode='same')
        else:
            smoothed_signals = raw_signals
        
        signal_std = np.std(smoothed_signals)
        if signal_std > 1e-10:
            transformed_signals = np.tanh(smoothed_signals / signal_std)
        else:
            transformed_signals = smoothed_signals
        
        returns = np.diff(self.prices) / self.prices[:-1]
        returns = np.concatenate([[0], returns])
        
        vol_window = min(20, len(returns) // 5)
        volatility = pd.Series(returns).rolling(
            window=vol_window,
            min_periods=1
        ).std().values
        
        volatility = np.clip(volatility, 0.001, None)
        scaled_signals = transformed_signals / volatility
        
        max_abs = np.percentile(np.abs(scaled_signals), 95)
        if max_abs > 0:
            scaled_signals = np.clip(scaled_signals / max_abs, -1, 1)
        
        return scaled_signals


# ============================================================================
# INTEGRATED TRADING SYSTEM
# ============================================================================

class IntegratedTradingSystem:
    """Unified system combining all three papers"""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.N = len(tickers)
        self.start_date = start_date
        self.end_date = end_date
        
        self.data = None
        self.toeplitz_analyzer = None
        self.emd_analyzers = {}
        self.langevin_filters = {}
        
        self.signals = None
        self.weights = None
        self.backtest_results = None
        
        print(f"Initialized integrated system with {self.N} assets")
        print(f"Period: {start_date} to {end_date}")
    
    def load_data(self):
        print("\n" + "="*70)
        print("STEP 1: DATA LOADING")
        print("="*70)
        
        print(f"\nDownloading data for {self.N} assets...")
        
        try:
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=False
            )
            
            prices = data['Adj Close']
            
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
                prices.columns = self.tickers
            
        except Exception as e:
            raise ValueError(f"Failed to download data: {e}")
        
        prices = prices.ffill().bfill().dropna()
        
        returns = np.log(prices / prices.shift(1)).dropna()
        returns_normalized = (returns - returns.mean()) / returns.std()
        
        self.data = {
            'prices': prices,
            'returns': returns,
            'returns_normalized': returns_normalized
        }
        
        print(f" Loaded {len(prices)} days of data")
        print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"  Assets: {self.N}")
        
        return self.data
    
    def apply_emd_to_all(self, max_imf: int = 8):
        print("\n" + "="*70)
        print("STEP 2: MULTI-SCALE DECOMPOSITION (Paper 2)")
        print("="*70)
        
        decompositions = {}
        
        for ticker in self.tickers:
            print(f"\n  Processing {ticker}...")
            
            returns = self.data['returns'][ticker].values
            
            try:
                analyzer = PiCountingIF(returns, max_imf=max_imf)
                imfs, residual = analyzer.apply_emd()
                cycles, IFs = analyzer.extract_cycles()
                
                decompositions[ticker] = {
                    'analyzer': analyzer,
                    'imfs': imfs,
                    'residual': residual,
                    'cycles': cycles,
                    'IFs': IFs
                }
                
                self.emd_analyzers[ticker] = analyzer
                print(f"     Decomposed into {len(imfs)} IMFs")
                
            except Exception as e:
                warnings.warn(f"EMD failed for {ticker}: {e}")
                decompositions[ticker] = None
        
        self.decompositions = decompositions
        return decompositions
    
    def track_all_assets(self, N_particles: int = 50):
        print("\n" + "="*70)
        print("STEP 3: REAL-TIME TRACKING (Paper 3)")
        print("="*70)
        
        tracking_results = {}
        
        for ticker in self.tickers:
            print(f"\n  Tracking {ticker}...")
            
            prices = self.data['prices'][ticker].values
            price_scale = np.mean(prices)
            
            params = {
                'alpha': 0.1,
                'sigma_theta': 0.01 * price_scale,
                'lambda': 5.0,
                'mu_J': 0.0,
                'sigma_J': 0.05 * price_scale,
                'sigma_y': 0.001 * price_scale
            }
            
            try:
                langevin = LangevinParticleFilter(prices, params)
                
                state_estimates, trend_estimates = langevin.particle_filter(
                    N_particles=N_particles,
                    dt=1.0,
                    show_progress=False
                )
                
                tracking_results[ticker] = {
                    'filter': langevin,
                    'state_estimates': state_estimates,
                    'trend_estimates': trend_estimates
                }
                
                self.langevin_filters[ticker] = langevin
                print(f"     Tracking complete")
                
            except Exception as e:
                warnings.warn(f"Tracking failed for {ticker}: {e}")
                tracking_results[ticker] = None
        
        self.tracking_results = tracking_results
        return tracking_results
    
    def generate_signals(self):
        print("\n" + "="*70)
        print("STEP 4: SIGNAL GENERATION")
        print("="*70)
        
        signals_dict = {}
        
        for ticker in self.tickers:
            print(f"  Generating signal for {ticker}...")
            
            if ticker not in self.tracking_results or self.tracking_results[ticker] is None:
                signals_dict[ticker] = np.zeros(len(self.data['prices']))
                continue
            
            try:
                langevin = self.langevin_filters[ticker]
                signal = langevin.generate_trading_signals()
                signals_dict[ticker] = signal
                
            except Exception as e:
                signals_dict[ticker] = np.zeros(len(self.data['prices']))
        
        signals = pd.DataFrame(signals_dict, index=self.data['prices'].index)
        
        self.signals = signals
        print(f"\n Generated signals for {self.N} assets")
        
        return signals
    
    def optimize_portfolio(self, Q: int = 5):
        print("\n" + "="*70)
        print("STEP 5: PORTFOLIO OPTIMIZATION (Paper 1)")
        print("="*70)
        
        if self.signals is None:
            raise ValueError("Must generate signals first")
        
        self.toeplitz_analyzer = ToeplitzDCTAnalyzer(
            self.data['returns_normalized']
        )
        
        R_emp = self.toeplitz_analyzer.empirical_correlation()
        rho_opt, R_toep, error = self.toeplitz_analyzer.fit_toeplitz_global(R_emp)
        
        print(f"\n  Optimal ρ: {rho_opt:.4f}")
        
        R_filtered, eigenvalues, eigenvectors = self.toeplitz_analyzer.eigenfilter(
            R_emp, Q, method='toeplitz'
        )
        
        var_explained = np.sum(eigenvalues[:Q]) / np.sum(eigenvalues) * 100
        print(f"  Keeping {Q} factors ({var_explained:.1f}% variance explained)")
        
        weights_list = []
        
        for t in range(len(self.signals)):
            s_t = self.signals.iloc[t].values
            
            lambda_risk = 1.0
            
            try:
                R_reg = R_filtered + 1e-6 * np.eye(self.N)
                w_opt = np.linalg.solve(R_reg, s_t) / lambda_risk
                
                w_sum = np.sum(np.abs(w_opt))
                if w_sum > 0:
                    w_opt = w_opt / w_sum
                else:
                    w_opt = np.ones(self.N) / self.N
                
                w_max = 0.2
                w_opt = np.clip(w_opt, -w_max, w_max)
                
                w_sum = np.sum(np.abs(w_opt))
                if w_sum > 0:
                    w_opt = w_opt / w_sum
                
            except:
                w_opt = np.ones(self.N) / self.N
            
            weights_list.append(w_opt)
        
        weights = pd.DataFrame(
            weights_list,
            index=self.signals.index,
            columns=self.signals.columns
        )
        
        self.weights = weights
        
        print(f" Portfolio optimization complete")
        
        return weights
    
    def backtest_portfolio(self, transaction_cost: float = 0.001):
        print("\n" + "="*70)
        print("STEP 6: PORTFOLIO BACKTEST")
        print("="*70)
        
        if self.weights is None:
            raise ValueError("Must optimize portfolio first")
        
        returns = self.data['returns']
        
        portfolio_returns = (self.weights.shift(1) * returns).sum(axis=1)
        portfolio_returns = portfolio_returns.dropna()
        
        weight_changes = self.weights.diff().abs().sum(axis=1)
        costs = weight_changes * transaction_cost
        costs = costs.loc[portfolio_returns.index]
        
        portfolio_returns_net = portfolio_returns - costs
        
        cumulative_returns = (1 + portfolio_returns_net).cumprod()
        
        mean_return = portfolio_returns_net.mean()
        volatility = portfolio_returns_net.std()
        
        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        n_days = len(portfolio_returns_net)
        n_positive = (portfolio_returns_net > 0).sum()
        win_rate = n_positive / n_days if n_days > 0 else 0
        
        results = {
            'portfolio_returns': portfolio_returns_net,
            'cumulative_returns': cumulative_returns,
            'mean_return': mean_return,
            'volatility': volatility,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': cumulative_returns.iloc[-1] - 1,
            'win_rate': win_rate,
            'drawdown': drawdown
        }
        
        self.backtest_results = results
        
        print(f"\n{'='*60}")
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Return:       {results['total_return']*100:>8.2f}%")
        print(f"Annual Return:      {results['annual_return']*100:>8.2f}%")
        print(f"Annual Volatility:  {results['annual_volatility']*100:>8.2f}%")
        print(f"Sharpe Ratio:       {results['sharpe_ratio']:>8.2f}")
        print(f"Max Drawdown:       {results['max_drawdown']*100:>8.2f}%")
        print(f"Win Rate:           {results['win_rate']*100:>8.1f}%")
        print(f"{'='*60}")
        
        return results
    
    def run_full_pipeline(self, max_imf: int = 8, N_particles: int = 50, Q: int = 5):
        print("\n" + "="*70)
        print(" INTEGRATED TRADING SYSTEM - FULL PIPELINE")
        print("="*70)
        
        self.load_data()
        decompositions = self.apply_emd_to_all(max_imf=max_imf)
        tracking_results = self.track_all_assets(N_particles=N_particles)
        signals = self.generate_signals()
        weights = self.optimize_portfolio(Q=Q)
        backtest_results = self.backtest_portfolio()
        
        results = {
            'data': self.data,
            'decompositions': decompositions,
            'tracking_results': tracking_results,
            'signals': signals,
            'weights': weights,
            'backtest_results': backtest_results
        }
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE!")
        print("="*70)
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def demo_integrated_system():
    """Run complete integrated system demo"""
    print("\n" + "="*70)
    print(" INTEGRATED SYSTEM DEMONSTRATION")
    print(" Combining Papers 1, 2, and 3")
    print("="*70)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'JPM', 'BAC', 'GS', 'XOM', 'CVX']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    system = IntegratedTradingSystem(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    results = system.run_full_pipeline(
        max_imf=6,
        N_particles=30,
        Q=5
    )
    
    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE!")
    print("="*70)
    
    return system, results


# Add this at the end of complete_system.py

# Import plotting module
from plotting_module import TradingSystemPlotter, create_all_plots

def demo_with_plots():
    """Run system and create all plots"""
    
    # Run the integrated system
    system, results = demo_integrated_system()
    
    # Create all plots
    print("\n" + "="*70)
    print(" CREATING VISUALIZATIONS")
    print("="*70)
    
    create_all_plots(system)
    
    print("\n All visualizations complete!")
    print("  Check the './plots' directory for all figures")
    
    return system, results

if __name__ == "__main__":
    system, results = demo_with_plots()