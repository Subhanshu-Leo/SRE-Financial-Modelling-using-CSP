"""
Utilities for Financial Signal Processing
Helper functions, data loading, validation, and analysis tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from scipy import stats


class DataLoader:
    """
    Robust data loader with validation and preprocessing
    """
    
    @staticmethod
    def download_data(tickers: Union[str, List[str]],
                     start_date: str,
                     end_date: str,
                     interval: str = '1d') -> pd.DataFrame:
        """
        Download data with error handling
        
        Parameters:
        -----------
        tickers : str or list of str
            Single ticker or list of tickers
        start_date, end_date : str
            Date range 'YYYY-MM-DD'
        interval : str
            Data frequency ('1d', '1h', '15m', etc.)
        
        Returns:
        --------
        prices : pd.DataFrame
            Adjusted close prices
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
        
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if 'Close' in data.columns:
                prices = data['Close']
            elif len(data.columns.levels) > 1:
                prices = data['Close']
            else:
                prices = data[['Close']]
                prices.columns = tickers
            
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
                prices.columns = tickers
            
        except Exception as e:
            raise ValueError(f"Download failed: {e}")
        
        # Validation
        if prices.empty:
            raise ValueError("No data downloaded")
        
        if prices.isnull().all().any():
            bad_tickers = prices.columns[prices.isnull().all()].tolist()
            warnings.warn(f"No data for: {bad_tickers}")
            prices = prices.dropna(axis=1, how='all')
        
        # Handle missing values
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        prices = prices.dropna()
        
        print(f" Downloaded {len(prices)} observations for {len(prices.columns)} assets")
        
        return prices
    
    @staticmethod
    def compute_returns(prices: pd.DataFrame,
                       method: str = 'log') -> pd.DataFrame:
        """
        Compute returns with validation
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        method : str
            'log' for log returns, 'simple' for simple returns
        
        Returns:
        --------
        returns : pd.DataFrame
            Returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        elif method == 'simple':
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        returns = returns.dropna()
        
        # Check for infinite values
        if np.isinf(returns.values).any():
            warnings.warn("Infinite values detected in returns")
            returns = returns.replace([np.inf, -np.inf], np.nan)
            returns = returns.fillna(method='ffill').fillna(0)
        
        return returns
    
    @staticmethod
    def normalize_returns(returns: pd.DataFrame,
                         method: str = 'zscore',
                         window: Optional[int] = None) -> pd.DataFrame:
        """
        Normalize returns
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Raw returns
        method : str
            'zscore' for standardization, 'minmax' for [0,1] scaling
        window : int, optional
            Rolling window for normalization.If None, use full sample
        
        Returns:
        --------
        normalized : pd.DataFrame
            Normalized returns
        """
        if method == 'zscore':
            if window is None:
                mean = returns.mean()
                std = returns.std()
            else:
                mean = returns.rolling(window=window, min_periods=1).mean()
                std = returns.rolling(window=window, min_periods=1).std()
            
            normalized = (returns - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            if window is None:
                min_val = returns.min()
                max_val = returns.max()
            else:
                min_val = returns.rolling(window=window, min_periods=1).min()
                max_val = returns.rolling(window=window, min_periods=1).max()
            
            normalized = (returns - min_val) / (max_val - min_val + 1e-8)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return normalized
    
    @staticmethod
    def prepare_data(tickers: Union[str, List[str]],
                    start_date: str,
                    end_date: str,
                    interval: str = '1d') -> Dict:
        """
        Complete data preparation pipeline
        
        Returns:
        --------
        data : dict
            Contains 'prices', 'returns', 'returns_normalized'
        """
        # Download
        prices = DataLoader.download_data(tickers, start_date, end_date, interval)
        
        # Compute returns
        returns = DataLoader.compute_returns(prices, method='log')
        
        # Normalize
        returns_normalized = DataLoader.normalize_returns(returns, method='zscore')
        
        data = {
            'prices': prices,
            'returns': returns,
            'returns_normalized': returns_normalized
        }
        
        # Summary statistics
        print("\nData Summary:")
        print(f"  Period: {prices.index[0]} to {prices.index[-1]}")
        print(f"  Observations: {len(prices)}")
        print(f"  Assets: {len(prices.columns)}")
        print(f"  Mean return: {returns.mean().mean():.4f}")
        print(f"  Mean volatility: {returns.std().mean():.4f}")
        
        return data


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis tools
    """
    
    @staticmethod
    def compute_metrics(returns: Union[pd.Series, np.ndarray],
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252) -> Dict:
        """
        Compute comprehensive performance metrics
        
        Parameters:
        -----------
        returns : pd.Series or ndarray
            Strategy returns
        risk_free_rate : float
            Annualized risk-free rate
        periods_per_year : int
            Trading periods per year (252 for daily, 12 for monthly)
        
        Returns:
        --------
        metrics : dict
            All performance metrics
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {'error': 'No valid returns'}
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Annualized
        annual_return = mean_return * periods_per_year
        annual_volatility = std_return * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        annual_downside_std = downside_std * np.sqrt(periods_per_year)
        sortino_ratio = excess_return / annual_downside_std if annual_downside_std > 0 else 0
        
        # Cumulative returns and drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        n_positive = np.sum(returns > 0)
        win_rate = n_positive / len(returns)
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        metrics = {
            'total_return': cumulative[-1] - 1,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'n_observations': len(returns)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Print metrics in formatted table"""
        print("\n" + "="*60)
        print(" PERFORMANCE METRICS")
        print("="*60)
        
        print("\nReturns:")
        print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"  Annualized Return:   {metrics['annual_return']*100:>8.2f}%")
        print(f"  Annualized Volatility: {metrics['annual_volatility']*100:>8.2f}%")
        
        print("\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>8.2f}")
        
        print("\nRisk:")
        print(f"  Maximum Drawdown:    {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  VaR (95%):           {metrics['var_95']*100:>8.2f}%")
        print(f"  CVaR (95%):          {metrics['cvar_95']*100:>8.2f}%")
        
        print("\nDistribution:")
        print(f"  Skewness:            {metrics['skewness']:>8.2f}")
        print(f"  Kurtosis:            {metrics['kurtosis']:>8.2f}")
        print(f"  Win Rate:            {metrics['win_rate']*100:>8.1f}%")
        
        print(f"\nObservations: {metrics['n_observations']}")
        print("="*60)
    
    @staticmethod
    def compare_strategies(strategies: Dict[str, Union[pd.Series, np.ndarray]],
                          save_path: Optional[str] = None):
        """
        Compare multiple strategies
        
        Parameters:
        -----------
        strategies : dict
            Dictionary of {strategy_name: returns}
        save_path : str, optional
            Path to save comparison figure
        """
        metrics_dict = {}
        
        for name, returns in strategies.items():
            metrics = PerformanceAnalyzer.compute_metrics(returns)
            metrics_dict[name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics_dict).T
        
        # Select key metrics
        key_metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio',
                      'max_drawdown', 'sortino_ratio', 'win_rate']
        comparison_df = comparison_df[key_metrics]
        
        print("\n" + "="*80)
        print(" STRATEGY COMPARISON")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80)
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1.Returns comparison
        for name, returns in strategies.items():
            if isinstance(returns, pd.Series):
                returns = returns.values
            cumulative = np.cumprod(1 + returns)
            axes[0, 0].plot(cumulative, label=name, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2.Sharpe ratios
        names = list(strategies.keys())
        sharpes = [metrics_dict[name]['sharpe_ratio'] for name in names]
        axes[0, 1].bar(names, sharpes, color='steelblue', alpha=0.7)
        axes[0, 1].set_title('Sharpe Ratios', fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3.Max drawdowns
        drawdowns = [metrics_dict[name]['max_drawdown']*100 for name in names]
        axes[0, 2].bar(names, drawdowns, color='red', alpha=0.7)
        axes[0, 2].set_title('Maximum Drawdowns', fontweight='bold')
        axes[0, 2].set_ylabel('Max Drawdown (%)')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4.Return distributions
        for name, returns in strategies.items():
            if isinstance(returns, pd.Series):
                returns = returns.values
            axes[1, 0].hist(returns*100, bins=50, alpha=0.5, label=name)
        axes[1, 0].set_title('Return Distributions', fontweight='bold')
        axes[1, 0].set_xlabel('Return (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5.Risk-return scatter
        ann_returns = [metrics_dict[name]['annual_return']*100 for name in names]
        ann_vols = [metrics_dict[name]['annual_volatility']*100 for name in names]
        axes[1, 1].scatter(ann_vols, ann_returns, s=200, alpha=0.6)
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (ann_vols[i], ann_returns[i]),
                              fontsize=9, ha='center')
        axes[1, 1].set_xlabel('Annualized Volatility (%)')
        axes[1, 1].set_ylabel('Annualized Return (%)')
        axes[1, 1].set_title('Risk-Return Profile', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6.Summary metrics
        summary_text = "SUMMARY\n" + "="*30 + "\n\n"
        summary_text += "Best Sharpe: " + max(names, key=lambda x: metrics_dict[x]['sharpe_ratio']) + "\n"
        summary_text += "Best Return: " + max(names, key=lambda x: metrics_dict[x]['annual_return']) + "\n"
        summary_text += "Lowest Risk: " + min(names, key=lambda x: metrics_dict[x]['annual_volatility']) + "\n"
        summary_text += "Lowest Drawdown: " + max(names, key=lambda x: metrics_dict[x]['max_drawdown']) + "\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text,
                       transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].axis('off')
        
        plt.suptitle('Strategy Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison figure saved to: {save_path}")
        
        return comparison_df, fig


class ValidationTools:
    """
    Tools for validating signal processing results
    """
    
    @staticmethod
    def check_correlation_matrix(R: np.ndarray, tol: float = 1e-6) -> Dict:
        """
        Validate correlation matrix properties
        
        Parameters:
        -----------
        R : ndarray
            Correlation matrix
        tol : float
            Tolerance for checks
        
        Returns:
        --------
        results : dict
            Validation results
        """
        n = R.shape[0]
        
        results = {
            'is_square': R.shape[0] == R.shape[1],
            'is_symmetric': np.allclose(R, R.T, atol=tol),
            'unit_diagonal': np.allclose(np.diag(R), 1, atol=tol),
            'is_positive_definite': False,
            'min_eigenvalue': None,
            'condition_number': None,
            'bounds_ok': np.all((R >= -1-tol) & (R <= 1+tol))
        }
        
        # Check positive definiteness
        try:
            eigvals = np.linalg.eigvalsh(R)
            results['min_eigenvalue'] = np.min(eigvals)
            results['is_positive_definite'] = results['min_eigenvalue'] >= -tol
            results['condition_number'] = np.max(eigvals) / max(np.min(eigvals), 1e-10)
        except:
            pass
        
        # Overall validity
        results['is_valid'] = all([
            results['is_square'],
            results['is_symmetric'],
            results['unit_diagonal'],
            results['bounds_ok']
        ])
        
        return results
    
    @staticmethod
    def print_validation(results: Dict):
        """Print validation results"""
        print("\nCorrelation Matrix Validation:")
        print(f"  Square: {results['is_square']}")
        print(f"  Symmetric: {results['is_symmetric']}")
        print(f"  Unit diagonal: {results['unit_diagonal']}")
        print(f"  Positive definite: {results['is_positive_definite']}")
        print(f"  Bounds [-1, 1]: {results['bounds_ok']}")
        
        if results['min_eigenvalue'] is not None:
            print(f"  Min eigenvalue: {results['min_eigenvalue']:.6f}")
            print(f"  Condition number: {results['condition_number']:.2f}")
        
        print(f"\n  Overall valid: {results['is_valid']}")
    
    @staticmethod
    def check_imf_properties(imf: np.ndarray) -> Dict:
        """
        Validate IMF properties
        
        Parameters:
        -----------
        imf : ndarray
            Intrinsic Mode Function
        
        Returns:
        --------
        results : dict
            IMF validation results
        """
        from scipy.signal import find_peaks
        
        # Find extrema
        maxima_idx, _ = find_peaks(imf)
        minima_idx, _ = find_peaks(-imf)
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(imf)))[0]
        
        n_extrema = len(maxima_idx) + len(minima_idx)
        n_zeros = len(zero_crossings)
        
        # Check IMF criterion: |n_extrema - n_zeros| <= 1
        extrema_zero_diff = abs(n_extrema - n_zeros)
        
        # Mean of envelopes should be ~0
        if len(maxima_idx) >= 2 and len(minima_idx) >= 2:
            upper_env = np.interp(np.arange(len(imf)), maxima_idx, imf[maxima_idx])
            lower_env = np.interp(np.arange(len(imf)), minima_idx, imf[minima_idx])
            mean_env = (upper_env + lower_env) / 2
            mean_env_max = np.max(np.abs(mean_env))
        else:
            mean_env_max = np.nan
        
        results = {
            'n_extrema': n_extrema,
            'n_zero_crossings': n_zeros,
            'extrema_zero_diff': extrema_zero_diff,
            'criterion_satisfied': extrema_zero_diff <= 1,
            'mean_envelope_max': mean_env_max,
            'envelope_near_zero': mean_env_max < 0.1 * np.std(imf) if not np.isnan(mean_env_max) else False
        }
        
        return results


def create_example_report(save_path: str = 'example_analysis_report.txt'):
    """
    Create a comprehensive example analysis report
    """
    print("\n" + "="*70)
    print(" CREATING EXAMPLE ANALYSIS REPORT")
    print("="*70)
    
    # Load sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    tickers = ['SPY', 'QQQ', 'IWM']
    
    data = DataLoader.prepare_data(
        tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Create report
    report = []
    report.append("="*70)
    report.append(" FINANCIAL SIGNAL PROCESSING - ANALYSIS REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nAssets: {', '.join(tickers)}")
    report.append(f"Period: {data['prices'].index[0]} to {data['prices'].index[-1]}")
    report.append(f"Observations: {len(data['prices'])}")
    
    report.append("\n\n" + "="*70)
    report.append(" SUMMARY STATISTICS")
    report.append("="*70)
    
    for ticker in tickers:
        returns = data['returns'][ticker]
        metrics = PerformanceAnalyzer.compute_metrics(returns.values)
        
        report.append(f"\n{ticker}:")
        report.append(f"  Annual Return:    {metrics['annual_return']*100:>8.2f}%")
        report.append(f"  Annual Volatility: {metrics['annual_volatility']*100:>8.2f}%")
        report.append(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.2f}")
        report.append(f"  Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
    
    report.append("\n\n" + "="*70)
    report.append(" CORRELATION ANALYSIS (Paper 1)")
    report.append("="*70)
    
    analyzer = ToeplitzDCTAnalyzer(data['returns_normalized'])
    R_emp = analyzer.empirical_correlation()
    rho_opt, R_toep, error = analyzer.fit_toeplitz_global(R_emp)
    
    report.append(f"\nOptimal ρ: {rho_opt:.4f}")
    report.append(f"Approximation error: {np.sqrt(error/analyzer.N**2):.4f}")
    
    # Eigenvalues
    results = analyzer.compare_methods(Q=3)
    report.append(f"\nEigenvalue spectrum:")
    for i, eig in enumerate(results['eigenvalues_klt'][:5]):
        report.append(f"  λ_{i+1} = {eig:.4f}")
    
    report.append(f"\nVariance explained by 3 factors: {results['var_explained_klt'][2]:.1f}%")
    
    report.append("\n\n" + "="*70)
    report.append(" CONCLUSION")
    report.append("="*70)
    report.append("\nThis report demonstrates the application of signal processing")
    report.append("techniques to financial data, integrating methods from three")
    report.append("seminal papers:")
    report.append("  1.Toeplitz/DCT approximation (Akansu & Torun, 2012)")
    report.append("  2.π-Counting IF and EMD (Zhang et al., 2012)")
    report.append("  3.Langevin dynamics (Christensen et al., 2012)")
    
    # Write to file
    report_text = '\n'.join(report)
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n Report saved to: {save_path}")
    print("\nReport preview:")
    print(report_text[:500] + "...\n")
    
    return report_text


# Main execution example
def run_complete_example():
    """
    Run complete example workflow using all three papers
    """
    print("\n" + "="*70)
    print(" COMPLETE EXAMPLE: ALL THREE PAPERS")
    print("="*70)
    
    # 1.Download and prepare data
    print("\n1.DATA PREPARATION")
    print("-" * 70)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)  # ~2 years of trading days
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    
    data = DataLoader.prepare_data(
        tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # 2.Paper 1: Correlation analysis
    print("\n2.PAPER 1: TOEPLITZ/DCT ANALYSIS")
    print("-" * 70)
    
    analyzer1 = ToeplitzDCTAnalyzer(data['returns_normalized'])
    results1 = analyzer1.compare_methods(Q=3)
    
    # 3.Paper 2: EMD analysis (single asset)
    print("\n3.PAPER 2: EMD + π-IF ANALYSIS")
    print("-" * 70)
    
    analyzer2 = PiCountingIF(data['returns']['AAPL'].values, max_imf=6)
    imfs, residual = analyzer2.apply_emd()
    cycles, IFs = analyzer2.extract_cycles()
    pred_results = analyzer2.predict_all(train_size=int(len(data['returns'])*0.7), 
                                        test_size=50)
    
    # 4.Paper 3: Langevin tracking
    print("\n4.PAPER 3: LANGEVIN DYNAMICS")
    print("-" * 70)
    
    langevin = LangevinParticleFilter(
        data['prices']['AAPL'].values,
        params={
            'alpha': 0.1,
            'sigma_theta': 0.01 * np.mean(data['prices']['AAPL'].values),
            'lambda': 5.0,
            'mu_J': 0.0,
            'sigma_J': 0.05 * np.mean(data['prices']['AAPL'].values),
            'sigma_y': 0.001 * np.mean(data['prices']['AAPL'].values)
        }
    )
    state_est, trend_est = langevin.particle_filter(N_particles=30, show_progress=False)
    backtest_results = langevin.backtest()
    
    # 5.Create visualizations
    print("\n5.CREATING VISUALIZATIONS")
    print("-" * 70)
    
    # Paper 1 viz
    fig1 = analyzer1.visualize_results(results1, save_path='example_paper1.png')
    
    # Paper 2 viz
    fig2 = analyzer2.visualize_decomposition(save_path='example_paper2_decomp.png')
    fig3 = analyzer2.visualize_predictions(pred_results, save_path='example_paper2_pred.png')
    
    # Paper 3 viz
    fig4 = langevin.visualize_tracking(save_path='example_paper3_tracking.png')
    fig5 = langevin.visualize_backtest(backtest_results, save_path='example_paper3_backtest.png')
    
    # 6.Performance comparison
    print("\n6.PERFORMANCE COMPARISON")
    print("-" * 70)
    
    # Buy-and-hold benchmark
    bh_returns = data['returns']['AAPL'].values
    
    # Strategy returns
    strategy_returns = backtest_results['strategy_returns']
    
    strategies = {
        'Buy & Hold': bh_returns[-len(strategy_returns):],
        'Langevin Strategy': strategy_returns
    }
    
    comparison_df, fig6 = PerformanceAnalyzer.compare_strategies(
        strategies,
        save_path='example_comparison.png'
    )
    
    plt.show()
    
    # 7.Generate report
    print("\n7.GENERATING REPORT")
    print("-" * 70)
    
    report = create_example_report('example_analysis_report.txt')
    
    print("\n" + "="*70)
    print(" COMPLETE EXAMPLE FINISHED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - example_paper1.png")
    print("  - example_paper2_decomp.png")
    print("  - example_paper2_pred.png")
    print("  - example_paper3_tracking.png")
    print("  - example_paper3_backtest.png")
    print("  - example_comparison.png")
    print("  - example_analysis_report.txt")
    
    return {
        'data': data,
        'paper1_results': results1,
        'paper2_results': pred_results,
        'paper3_results': backtest_results,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    # Run complete example
    results = run_complete_example()