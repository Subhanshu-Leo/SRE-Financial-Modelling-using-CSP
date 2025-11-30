"""
Paper 3: Langevin Dynamics for High-Frequency Futures Trading
High-quality implementation with error handling and optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.linalg import expm
from scipy.stats import norm
import warnings
from tqdm import tqdm


class LangevinParticleFilter:
    """
    Implement Langevin jump-diffusion model with Variable Rate Particle Filter
    from Christensen, Murphy & Godsill (2012)
    """
    
    def __init__(self, prices: np.ndarray, params: Optional[Dict] = None):
        """
        Initialize Langevin filter
        
        Parameters:
        -----------
        prices : ndarray or pd.Series
            Price time series
        params : dict, optional
            Model parameters. If None, use defaults scaled to data
        """
        if isinstance(prices, pd.Series):
            self.price_index = prices.index
            prices = prices.values
        else:
            self.price_index = None
        self.prices = prices.to_numpy().flatten()

        self.N_samples = len(self.prices)
        
        # Price scale for parameter defaults
        price_scale = np.mean(self.prices)
        
        # Default parameters (scaled to data)
        if params is None:
            params = {}
        
        self.alpha = params.get('alpha', 0.1)  # Mean reversion rate
        self.sigma_theta = params.get('sigma_theta', 0.01 * price_scale)  # Trend noise
        self.lambda_jump = params.get('lambda', 5.0)  # Jump rate (jumps per unit time)
        self.mu_J = params.get('mu_J', 0.0)  # Mean jump size
        self.sigma_J = params.get('sigma_J', 0.05 * price_scale)  # Jump size std
        self.sigma_y = params.get('sigma_y', 0.001 * price_scale)  # Observation noise
        
        # State transition matrices
        self.A = np.array([[0, 1],
                          [0, -self.alpha]])
        self.B = np.array([[0],
                          [self.sigma_theta]])
        self.H = np.array([[1, 0]])  # Observation matrix
        
        # Storage for results
        self.state_estimates = None
        self.trend_estimates = None
        self.particles_history = None
        
        print(f"Initialized Langevin filter with {self.N_samples} observations")
        print(f"Parameters: α={self.alpha:.3f}, σ_θ={self.sigma_theta:.4f}, "
              f"λ={self.lambda_jump:.1f}, σ_J={self.sigma_J:.4f}, σ_y={self.sigma_y:.4f}")
    
    def matrix_exponential(self, dt: float) -> np.ndarray:
        """
        Compute matrix exponential e^{At} analytically
        
        For A = [[0, 1], [0, -α]], we have:
        e^{At} = [[1, (1-e^{-αt})/α], [0, e^{-αt}]]
        
        Parameters:
        -----------
        dt : float
            Time interval
        
        Returns:
        --------
        exp_At : ndarray (2, 2)
            Matrix exponential
        """
        alpha = self.alpha
        
        if alpha < 1e-10:  # Handle α ≈ 0 case
            exp_At = np.array([[1, dt],
                              [0, 1]])
        else:
            exp_neg_alpha_t = np.exp(-alpha * dt)
            exp_At = np.array([[1, (1 - exp_neg_alpha_t) / alpha],
                              [0, exp_neg_alpha_t]])
        
        return exp_At
    
    def transition_covariance(self, dt: float) -> np.ndarray:
        """
        Compute transition covariance matrix Q(dt) analytically
        
        Uses closed-form formulas from the paper
        
        Parameters:
        -----------
        dt : float
            Time interval
        
        Returns:
        --------
        Q : ndarray (2, 2)
            Transition covariance
        """
        alpha = self.alpha
        sigma_theta = self.sigma_theta
        
        if alpha < 1e-10:  # Handle α ≈ 0 case
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
        
        Q = np.array([[Sigma_11, Sigma_12],
                     [Sigma_12, Sigma_22]])
        
        return Q
    
    def transition_with_jumps(self, x_n: np.ndarray, dt: float,
                             jump_times: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state transition with jumps
        
        Parameters:
        -----------
        x_n : ndarray (2,)
            State at time t_n: [value, trend]
        dt : float
            Time interval to t_{n+1}
        jump_times : list of float
            Jump times in interval (0, dt), relative to t_n
        
        Returns:
        --------
        mu : ndarray (2,)
            Mean of transition distribution
        Sigma : ndarray (2, 2)
            Covariance of transition distribution
        """
        if len(jump_times) == 0:
            # No jumps - simple diffusion
            exp_Adt = self.matrix_exponential(dt)
            mu = exp_Adt @ x_n
            Sigma = self.transition_covariance(dt)
        else:
            # With jumps - chain multiple transitions
            mu = x_n.copy()
            Sigma = np.zeros((2, 2))
            
            t_prev = 0.0
            
            for tau in sorted(jump_times):
                if tau <= 0 or tau >= dt:
                    continue  # Skip invalid jump times
                
                # Pre-jump diffusion: t_prev to tau
                dt_pre = tau - t_prev
                if dt_pre > 0:
                    exp_Adt_pre = self.matrix_exponential(dt_pre)
                    Q_pre = self.transition_covariance(dt_pre)
                    
                    mu = exp_Adt_pre @ mu
                    Sigma = exp_Adt_pre @ Sigma @ exp_Adt_pre.T + Q_pre
                
                # Jump at tau
                jump_mean = np.array([0, self.mu_J])
                jump_cov = np.array([[0, 0],
                                    [0, self.sigma_J**2]])
                
                mu = mu + jump_mean
                Sigma = Sigma + jump_cov
                
                t_prev = tau
            
            # Post-jump diffusion: t_prev to dt
            dt_post = dt - t_prev
            if dt_post > 0:
                exp_Adt_post = self.matrix_exponential(dt_post)
                Q_post = self.transition_covariance(dt_post)
                
                mu = exp_Adt_post @ mu
                Sigma = exp_Adt_post @ Sigma @ exp_Adt_post.T + Q_post
        
        return mu, Sigma
    
    def kalman_predict(self, m_n: np.ndarray, Sigma_n: np.ndarray,
                      dt: float, jump_times: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman filter prediction step
        
        Parameters:
        -----------
        m_n : ndarray (2,)
            Posterior mean at t_n
        Sigma_n : ndarray (2, 2)
            Posterior covariance at t_n
        dt : float
            Time to next observation
        jump_times : list
            Jump times in (0, dt)
        
        Returns:
        --------
        m_pred : ndarray (2,)
            Predicted mean at t_{n+1}
        Sigma_pred : ndarray (2, 2)
            Predicted covariance at t_{n+1}
        """
        # Get transition statistics
        F_mu, F_Sigma = self.transition_with_jumps(
            np.zeros(2), dt, jump_times
        )  # Get transition for zero state
        
        # Now apply to actual state
        mu_trans, Sigma_trans = self.transition_with_jumps(m_n, dt, jump_times)
        
        return mu_trans, Sigma_trans
    
    def kalman_update(self, m_pred: np.ndarray, Sigma_pred: np.ndarray,
                     y_n: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kalman filter update step
        
        Parameters:
        -----------
        m_pred : ndarray (2,)
            Predicted mean
        Sigma_pred : ndarray (2, 2)
            Predicted covariance
        y_n : float
            Observation
        
        Returns:
        --------
        m_updated : ndarray (2,)
            Updated mean
        Sigma_updated : ndarray (2, 2)
            Updated covariance
        likelihood : float
            Observation likelihood p(y_n | y_{1:n-1})
        """
        # Predicted observation
        y_pred = (self.H @ m_pred)[0]
        
        # Innovation covariance
        S = (self.H @ Sigma_pred @ self.H.T)[0, 0] + self.sigma_y**2
        
        # Ensure S is positive
        S = max(S, 1e-10)
        
        # Kalman gain
        K = (Sigma_pred @ self.H.T) / S
        K = K.reshape(-1)  # Ensure 1D
        
        # Innovation
        innovation = y_n - y_pred
        
        # Update
        m_updated = m_pred + K * innovation
        Sigma_updated = Sigma_pred - np.outer(K, K) * S
        
        # Ensure Sigma remains positive definite
        Sigma_updated = (Sigma_updated + Sigma_updated.T) / 2  # Symmetrize
        eigvals = np.linalg.eigvalsh(Sigma_updated)
        if np.min(eigvals) < 0:
            Sigma_updated += np.eye(2) * (1e-6 - np.min(eigvals))
        
        # Observation likelihood (Gaussian PDF)
        likelihood = norm.pdf(innovation, 0, np.sqrt(S))
        
        return m_updated, Sigma_updated, likelihood
    
    def propose_jump_times(self, dt: float) -> List[float]:
        """
        Propose jump times in interval (0, dt) using Poisson process
        
        Parameters:
        -----------
        dt : float
            Time interval
        
        Returns:
        --------
        jump_times : list of float
            Proposed jump times (relative to start of interval)
        """
        jump_times = []
        t = 0.0
        
        # Sample jumps until beyond dt
        while True:
            # Inter-arrival time (exponential distribution)
            if self.lambda_jump > 0:
                inter_arrival = np.random.exponential(1.0 / self.lambda_jump)
            else:
                break  # No jumps if lambda = 0
            
            t_jump = t + inter_arrival
            
            if t_jump < dt:
                jump_times.append(t_jump)
                t = t_jump
            else:
                break
        
        return jump_times
    
    def particle_filter(self, N_particles: int = 100,
                       dt: float = 1.0,
                       show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Variable Rate Particle Filter with Rao-Blackwellization
        
        Parameters:
        -----------
        N_particles : int
            Number of particles
        dt : float
            Time step between observations (e.g., 1.0 for daily)
        show_progress : bool
            Show progress bar
        
        Returns:
        --------
        state_estimates : ndarray (N_samples, 2)
            Estimated states [value, trend] at each time
        trend_estimates : ndarray (N_samples,)
            Estimated trend at each time
        """
        print("\n" + "="*60)
        print("RUNNING VARIABLE RATE PARTICLE FILTER")
        print("="*60)
        print(f"Particles: {N_particles}, Observations: {self.N_samples}")
        
        # Initialize particles
        particles = []
        for i in range(N_particles):
            particle = {
                'jump_times_full': [],  # All jump times from start
                'm': np.array([self.prices[0], 0.0]),  # [value, trend]
                'Sigma': np.eye(2) * 100.0,
                'weight': 1.0 / N_particles
            }
            particles.append(particle)
        
        # Storage for estimates
        state_estimates = np.zeros((self.N_samples, 2))
        trend_estimates = np.zeros(self.N_samples)
        
        # Initial estimate
        state_estimates[0] = np.array([self.prices[0], 0.0])
        trend_estimates[0] = 0.0
        
        # Progress bar
        iterator = range(1, self.N_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering")
        
        for n in iterator:
            observation = self.prices[n]
            new_particles = []
            
            # Step 1: RESAMPLE and PROPOSE for each particle
            for particle in particles:
                # Determine number of offspring (resampling)
                ideal_offspring = N_particles * particle['weight']
                n_offspring = int(np.floor(ideal_offspring))
                
                # Stochastic rounding
                if np.random.rand() < (ideal_offspring - n_offspring):
                    n_offspring += 1
                
                if n_offspring == 0:
                    continue
                
                # Create offspring
                for _ in range(n_offspring):
                    # Propose jump times in this interval
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
                warnings.warn(f"All particles died at step {n}.Reinitializing.")
                # Reinitialize
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
            
            # Step 2: COLLAPSE non-jumping particles
            jumping = [p for p in new_particles if p['jumped']]
            non_jumping = [p for p in new_particles if not p['jumped']]
            
            if len(non_jumping) > 0:
                # Merge into single particle
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
            
            # Step 3: UPDATE with Kalman filter and reweight
            for particle in particles:
                try:
                    # Kalman predict
                    m_pred, Sigma_pred = self.kalman_predict(
                        particle['m'],
                        particle['Sigma'],
                        dt,
                        particle['jump_times_interval']
                    )
                    
                    # Kalman update
                    m_updated, Sigma_updated, likelihood = self.kalman_update(
                        m_pred,
                        Sigma_pred,
                        observation
                    )
                    
                    # Update particle
                    particle['m'] = m_updated
                    particle['Sigma'] = Sigma_updated
                    particle['weight'] *= max(likelihood, 1e-300)  # Avoid underflow
                    
                except Exception as e:
                    warnings.warn(f"Particle update failed: {e}")
                    particle['weight'] = 1e-300
            
            # Step 4: NORMALIZE weights
            total_weight = sum(p['weight'] for p in particles)
            
            if total_weight < 1e-300:
                # All weights became zero - reset uniformly
                for particle in particles:
                    particle['weight'] = 1.0 / len(particles)
            else:
                for particle in particles:
                    particle['weight'] /= total_weight
            
            # Compute weighted mean estimate
            state_mean = np.zeros(2)
            for particle in particles:
                state_mean += particle['weight'] * particle['m']
            
            state_estimates[n] = state_mean
            trend_estimates[n] = state_mean[1]
        
        print("\nParticle filter complete.")
        print(f"Final state estimate: value={state_estimates[-1, 0]:.2f}, "
              f"trend={state_estimates[-1, 1]:.4f}")
        
        self.state_estimates = state_estimates
        self.trend_estimates = trend_estimates
        
        return state_estimates, trend_estimates
    
    def generate_trading_signals(self, trend_estimates: Optional[np.ndarray] = None,
                                smoothing_window: int = 4,
                                nonlinear_scale: float = 1.0) -> np.ndarray:
        """
        Generate trading signals from trend estimates
        
        Parameters:
        -----------
        trend_estimates : ndarray, optional
            Trend estimates. If None, use self.trend_estimates
        smoothing_window : int
            FIR smoothing window size
        nonlinear_scale : float
            Scale for tanh transformation
        
        Returns:
        --------
        signals : ndarray (N_samples,)
            Trading signals in [-1, 1]
        """
        if trend_estimates is None:
            if self.trend_estimates is None:
                raise ValueError("Must run particle_filter() first")
            trend_estimates = self.trend_estimates
        
        # Step 1: Sign of trend change
        trend_diff = np.diff(trend_estimates, prepend=trend_estimates[0])
        raw_signals = np.sign(trend_diff)
        
        # Step 2: FIR smoothing
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed_signals = np.convolve(raw_signals, kernel, mode='same')
        else:
            smoothed_signals = raw_signals
        
        # Step 3: Nonlinear transformation (tanh)
        signal_std = np.std(smoothed_signals)
        if signal_std > 1e-10:
            transformed_signals = np.tanh(smoothed_signals / (signal_std * nonlinear_scale))
        else:
            transformed_signals = smoothed_signals
        
        # Step 4: Volatility scaling (optional - using rolling std of returns)
        returns = np.diff(self.prices) / self.prices[:-1]
        returns = np.concatenate([[0], returns])
        
        # Simple rolling volatility
        vol_window = min(20, len(returns) // 5)
        volatility = pd.Series(returns).rolling(
            window=vol_window,
            min_periods=1
        ).std().values
        
        volatility = np.clip(volatility, 1e-6, None)  # Avoid division by zero
        scaled_signals = transformed_signals / volatility
        scaled_signals = np.nan_to_num(scaled_signals, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to [-1, 1]
        max_abs = np.percentile(np.abs(scaled_signals), 95)  # Use 95th percentile for robustness
        if max_abs > 0:
            scaled_signals = np.clip(scaled_signals / max_abs, -1, 1)
        
        return scaled_signals
    
    def backtest(self, signals: Optional[np.ndarray] = None,
                transaction_cost: float = 0.001,
                initial_capital: float = 1.0) -> Dict:
        """
        Backtest trading signals
        
        Parameters:
        -----------
        signals : ndarray, optional
            Trading signals [-1, 1]. If None, generate from trends
        transaction_cost : float
            Round-trip cost as fraction (e.g., 0.001 = 0.1%)
        initial_capital : float
            Initial capital (normalized)
        
        Returns:
        --------
        results : dict
            Backtest results including returns, Sharpe, drawdown, etc.
        """
        if signals is None:
            signals = self.generate_trading_signals()
        
        print("\n" + "="*60)
        print("BACKTESTING STRATEGY")
        print("="*60)
        
        # Market returns
        returns = np.diff(self.prices) / self.prices[:-1]
        returns = np.concatenate([[0], returns])
        
        # Strategy returns (signal[t-1] * return[t])
        strategy_returns = signals[:-1] * returns[1:]
        
        # Transaction costs
        position_changes = np.abs(np.diff(signals))
        costs = position_changes * transaction_cost
        
        # Net returns
        strategy_returns_net = strategy_returns - costs
        
        # Cumulative returns
        cumulative_returns = initial_capital * np.cumprod(1 + strategy_returns_net)
        
        # Performance metrics
        mean_return = np.mean(strategy_returns_net)
        volatility = np.std(strategy_returns_net)
        
        # Annualized metrics (assuming daily data)
        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        n_trades = np.sum(position_changes > 0.01)
        winning_trades = np.sum(strategy_returns_net > 0)
        win_rate = winning_trades / len(strategy_returns_net) if len(strategy_returns_net) > 0 else 0
        
        results = {
            'signals': signals,
            'returns': returns,
            'strategy_returns': strategy_returns_net,
            'cumulative_returns': cumulative_returns,
            'mean_return': mean_return,
            'volatility': volatility,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': cumulative_returns[-1] - initial_capital,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'initial_capital': initial_capital
        }
        
        print(f"\nBacktest Results:")
        print(f"  Total Return: {results['total_return']*100:.2f}%")
        print(f"  Annual Return: {results['annual_return']*100:.2f}%")
        print(f"  Annual Volatility: {results['annual_volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"  Number of Trades: {results['n_trades']}")
        print(f"  Win Rate: {results['win_rate']*100:.1f}%")
        
        return results
    
    def visualize_tracking(self, save_path: Optional[str] = None):
        """Visualize tracking results"""
        if self.state_estimates is None:
            raise ValueError("Must run particle_filter() first")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        if self.price_index is not None:
            time = self.price_index
        else:
            time = np.arange(self.N_samples)
        
        # 1.Price and value estimate
        axes[0].plot(time, self.prices, 'b-', label='Observed Price',
                    alpha=0.7, linewidth=1.5)
        axes[0].plot(time, self.state_estimates[:, 0], 'r--',
                    label='Estimated Value', linewidth=2, alpha=0.8)
        axes[0].set_ylabel('Price', fontsize=11)
        axes[0].set_title('Price Tracking', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 2.Trend estimate
        axes[1].plot(time, self.trend_estimates, 'g-',
                    label='Estimated Trend', linewidth=2)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].fill_between(time, 0, self.trend_estimates,
                            where=(self.trend_estimates > 0),
                            alpha=0.3, color='green', label='Positive Trend')
        axes[1].fill_between(time, 0, self.trend_estimates,
                            where=(self.trend_estimates < 0),
                            alpha=0.3, color='red', label='Negative Trend')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].set_title('Trend Estimation', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # 3.Trading signal
        signals = self.generate_trading_signals()
        print("Signals stats:")
        print(f"  Min: {np.min(signals):.4f}")
        print(f"  Max: {np.max(signals):.4f}")
        print(f"  Mean: {np.mean(signals):.4f}")
        print(f"  Std: {np.std(signals):.4f}")
        axes[2].plot(time, signals, 'purple', linewidth=1.5, label='Trading Signal')
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        axes[2].fill_between(time, 0, signals,
                            where=(signals > 0),
                            alpha=0.3, color='green', label='Long')
        axes[2].fill_between(time, 0, signals,
                            where=(signals < 0),
                            alpha=0.3, color='red', label='Short')
        axes[2].set_xlabel('Time', fontsize=11)
        axes[2].set_ylabel('Signal', fontsize=11)
        axes[2].set_title('Trading Signal', fontsize=12, fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-1.2, 1.2)
        
        plt.suptitle('Langevin Dynamics Tracking: Paper 3 (Christensen et al., 2012)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nTracking figure saved to: {save_path}")
        
        return fig
    
    def visualize_backtest(self, results: Dict, save_path: Optional[str] = None):
        """Visualize backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time = np.arange(len(results['strategy_returns']))
        if self.price_index is not None and len(self.price_index) > len(time):
            time_index = self.price_index[1:len(time)+1]
        else:
            time_index = time
        
        # 1. Cumulative returns
        axes[0, 0].plot(time_index, results['cumulative_returns'],
                       'b-', linewidth=2)
        axes[0, 0].axhline(results['initial_capital'], color='k',
                          linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].set_title('Strategy Performance', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2.Daily returns distribution
        axes[0, 1].hist(results['strategy_returns'], bins=50,
                       edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2,
                          label='Zero Return')
        axes[0, 1].axvline(np.mean(results['strategy_returns']),
                          color='g', linestyle='-', linewidth=2,
                          label='Mean Return')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Return Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3.Drawdown
        running_max = np.maximum.accumulate(results['cumulative_returns'])
        drawdown = (results['cumulative_returns'] - running_max) / running_max
        
        axes[1, 0].fill_between(time_index, 0, drawdown * 100,
                               alpha=0.5, color='red')
        axes[1, 0].plot(time_index, drawdown * 100, 'r-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].set_title('Drawdown Over Time', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4.Performance summary
        summary_text = f"""
PERFORMANCE SUMMARY
{'='*40}

Returns:
  Total Return:      {results['total_return']*100:>8.2f}%
  Annual Return:     {results['annual_return']*100:>8.2f}%
  Annual Volatility: {results['annual_volatility']*100:>8.2f}%

Risk-Adjusted:
  Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}
  Max Drawdown:      {results['max_drawdown']*100:>8.2f}%

Trading:
  Number of Trades:  {results['n_trades']:>8.0f}
  Win Rate:          {results['win_rate']*100:>8.1f}%

Model Parameters:
  α (mean reversion):  {self.alpha:.3f}
  λ (jump rate):       {self.lambda_jump:.1f}
  σ_θ (trend noise):   {self.sigma_theta:.4f}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text,
                       transform=axes[1, 1].transAxes,
                       fontsize=9.5, verticalalignment='top',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        axes[1, 1].axis('off')
        
        plt.suptitle('Backtest Results: Paper 3 (Christensen et al., 2012)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nBacktest figure saved to: {save_path}")
        
        return fig


def demo_paper3():
    """Demonstration of Paper 3 implementation"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("\n" + "="*70)
    print(" PAPER 3: LANGEVIN DYNAMICS - DEMONSTRATION")
    print("="*70)
    
    # Download data
    print("\nDownloading data...")
    ticker = 'SPY'  # S&P 500 ETF
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False,auto_adjust=False)
    prices = data['Adj Close']
    
    print(f"Data: {len(prices)} days")
    
    # Initialize with optimal parameters
    price_scale = np.mean(prices.values)
    params = {
        'alpha': 0.1,
        'sigma_theta': 0.01 * price_scale,
        'lambda': 5.0,
        'mu_J': 0.0,
        'sigma_J': 0.05 * price_scale,
        'sigma_y': 0.001 * price_scale
    }
    
    # Initialize filter
    langevin = LangevinParticleFilter(prices, params)
    
    # Run particle filter
    state_estimates, trend_estimates = langevin.particle_filter(
        N_particles=50,
        dt=1.0,
        show_progress=True
    )
    
    # Visualize tracking
    fig1 = langevin.visualize_tracking(save_path='paper3_tracking.png')
    
    # Backtest
    results = langevin.backtest(transaction_cost=0.001)
    
    # Visualize backtest
    fig2 = langevin.visualize_backtest(results, save_path='paper3_backtest.png')
    
    plt.show()
    
    print("\n" + "="*70)
    print(" PAPER 3 DEMONSTRATION COMPLETE")
    print("="*70)
    
    return langevin, results


if __name__ == "__main__":
    langevin, results = demo_paper3()