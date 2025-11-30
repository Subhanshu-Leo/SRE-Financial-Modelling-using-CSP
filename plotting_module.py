"""
plotting_module.py
Standalone plotting functions for the integrated trading system
Add this file and import it to create all visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TradingSystemPlotter:
    """Comprehensive plotting for the integrated trading system"""
    
    @staticmethod
    def plot_paper1_results(analyzer, results: Dict, save_path: Optional[str] = None):
        """
        Visualize Paper 1: Toeplitz/DCT Analysis
        
        Parameters:
        -----------
        analyzer : ToeplitzDCTAnalyzer
            The analyzer object
        results : dict
            Results from compare_methods()
        save_path : str, optional
            Path to save figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        Q = results['Q']
        N = analyzer.N
        
        # 1.Empirical vs Toeplitz correlation heatmaps
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(results['R_emp'], cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax1.set_title('Empirical Correlation', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Asset Index')
        ax1.set_ylabel('Asset Index')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(results['R_toep'], cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax2.set_title(f'Toeplitz (ρ={results["rho_opt"]:.3f})', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Asset Index')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = results['R_emp'] - results['R_toep']
        im3 = ax3.imshow(diff, cmap='seismic', vmin=-0.3, vmax=0.3)
        ax3.set_title('Approximation Error', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Asset Index')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 2.Eigenvalue spectrum
        ax4 = fig.add_subplot(gs[1, 0])
        x = np.arange(1, N + 1)
        width = 0.35
        ax4.bar(x - width/2, results['eigenvalues_klt'], width, 
               alpha=0.7, label='KLT', color='blue')
        ax4.bar(x + width/2, results['eigenvalues_toeplitz'], width, 
               alpha=0.7, label='Toeplitz', color='green')
        ax4.axvline(Q + 0.5, color='red', linestyle='--', linewidth=2, label=f'Q={Q}')
        ax4.set_xlabel('Factor Index', fontsize=10)
        ax4.set_ylabel('Eigenvalue', fontsize=10)
        ax4.set_title('Eigenvalue Spectrum', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0.5, N + 0.5)
        
        # 3.Cumulative variance explained
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(x, results['var_explained_klt'], 'o-', label='KLT', markersize=5, linewidth=2)
        ax5.plot(x, results['var_explained_toeplitz'], 's-', label='Toeplitz', markersize=5, linewidth=2)
        ax5.axvline(Q, color='red', linestyle='--', linewidth=2)
        ax5.axhline(80, color='gray', linestyle=':', linewidth=2, label='80% threshold')
        ax5.set_xlabel('Number of Factors', fontsize=10)
        ax5.set_ylabel('Variance Explained (%)', fontsize=10)
        ax5.set_title('Cumulative Variance Explained', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, N + 0.5)
        ax5.set_ylim(0, 105)
        
        # 4.Risk contributions
        ax6 = fig.add_subplot(gs[1, 2])
        factors_to_plot = min(10, N)
        x_rc = np.arange(1, factors_to_plot + 1)
        width = 0.35
        ax6.bar(x_rc - width/2, results['risk_contrib_klt'][:factors_to_plot], 
               width, alpha=0.7, label='KLT', color='blue')
        ax6.bar(x_rc + width/2, results['risk_contrib_toep'][:factors_to_plot], 
               width, alpha=0.7, label='Toeplitz', color='green')
        ax6.set_xlabel('Factor Index', fontsize=10)
        ax6.set_ylabel('Risk Contribution', fontsize=10)
        ax6.set_title('Factor Risk Contributions (Top 10)', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 5.Portfolio risk comparison
        ax7 = fig.add_subplot(gs[2, 0])
        methods = ['Empirical', 'KLT\nFiltered', 'Toeplitz\nFiltered']
        risks = [results['risk_empirical'], results['risk_klt'], results['risk_toeplitz']]
        colors = ['blue', 'green', 'orange']
        bars = ax7.bar(methods, risks, color=colors, alpha=0.7)
        ax7.set_ylabel('Portfolio Risk', fontsize=10)
        ax7.set_title('Portfolio Risk Comparison', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 6.Compaction efficiency comparison
        ax8 = fig.add_subplot(gs[2, 1])
        methods_eta = ['KLT', 'Toeplitz']
        etas = [results['eta_c_klt'], results['eta_c_toeplitz']]
        colors_eta = ['blue', 'green']
        bars_eta = ax8.bar(methods_eta, etas, color=colors_eta, alpha=0.7)
        ax8.set_ylabel('Compaction Efficiency', fontsize=10)
        ax8.set_title('Compaction Efficiency (η_c)', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_ylim(0, max(etas) * 1.2)
        
        for bar in bars_eta:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 7.Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        summary_text = f"""
PAPER 1 SUMMARY
{'='*35}

Correlation Structure:
  Optimal ρ: {results['rho_opt']:.4f}
  Toeplitz RMSE: {np.sqrt(results['error_toeplitz']/N**2):.4f}

Performance (Q={Q}):
  Var Explained (KLT): {results['var_explained_klt'][Q-1]:.2f}%
  Var Explained (Toep): {results['var_explained_toeplitz'][Q-1]:.2f}%
  
Compaction Efficiency:
  KLT:     {results['eta_c_klt']:.4f}
  Toeplitz: {results['eta_c_toeplitz']:.4f}
  Ratio:   {results['performance_ratio']:.2%}

Portfolio Risk:
  Empirical:  {results['risk_empirical']:.4f}
  KLT Filter: {results['risk_klt']:.4f}
  Toep Filter: {results['risk_toeplitz']:.4f}

Computational Speedup:
  ~{int(N**2 / np.log2(N)):,}× faster
        """
        
        ax9.text(0.05, 0.95, summary_text,
                transform=ax9.transAxes,
                fontsize=8.5, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax9.axis('off')
        
        plt.suptitle('Paper 1: Toeplitz/DCT Analysis (Akansu & Torun, 2012)', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_emd_decomposition(analyzer, save_path: Optional[str] = None):
        """
        Visualize Paper 2: EMD Decomposition with IFs
        
        Parameters:
        -----------
        analyzer : PiCountingIF
            The EMD analyzer
        save_path : str, optional
            Path to save figure
        """
        if analyzer.imfs is None or analyzer.IFs is None:
            raise ValueError("Must run apply_emd() and extract_cycles() first")
        
        n_plots = analyzer.n_imfs + 2
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 2.5*n_plots))
        
        time = np.arange(analyzer.N)
        
        # Original signal
        axes[0].plot(time, analyzer.signal, 'b-', linewidth=1, alpha=0.8)
        axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, analyzer.N)
        
        # Each IMF with IF overlay
        for i, (imf, IF) in enumerate(zip(analyzer.imfs, analyzer.IFs)):
            ax = axes[i+1]
            
            color_imf = f'C{i}'
            ax.plot(time, imf, color=color_imf, linewidth=0.8, alpha=0.8)
            ax.set_ylabel('IMF Amp', color=color_imf, fontsize=9)
            ax.tick_params(axis='y', labelcolor=color_imf)
            
            # IF on secondary axis
            ax2 = ax.twinx()
            color_if = 'red'
            ax2.plot(time, IF, color=color_if, linewidth=0.8, alpha=0.6)
            ax2.set_ylabel('IF', color=color_if, fontsize=9)
            ax2.tick_params(axis='y', labelcolor=color_if)
            
            ax.set_title(f'IMF {i+1} (Cycle = {analyzer.primary_cycles[i]} samples)', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, analyzer.N)
        
        # Residual
        axes[-1].plot(time, analyzer.residual, 'g-', linewidth=1.2)
        axes[-1].set_title('Residual (Trend)', fontsize=12, fontweight='bold')
        axes[-1].set_xlabel('Time (samples)', fontsize=10)
        axes[-1].set_ylabel('Amplitude', fontsize=10)
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlim(0, analyzer.N)
        
        plt.suptitle('Paper 2: EMD with π-Counting IF (Zhang et al., 2012)',
                    fontsize=14, fontweight='bold', y=0.9995)
        plt.tight_layout(rect=[0, 0, 1, 0.998])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_emd_predictions(results: Dict, save_path: Optional[str] = None):
        """
        Visualize Paper 2: Prediction Results
        
        Parameters:
        -----------
        results : dict
            Results from predict_all()
        save_path : str, optional
            Path to save figure
        """
        n_components = len(results['component_predictions'])
        
        fig, axes = plt.subplots(n_components + 1, 1, 
                                figsize=(15, 2.5*(n_components+1)))
        
        time = np.arange(results['test_size'])
        
        # Each component prediction
        for i, (pred, actual, comp_result) in enumerate(zip(
            results['component_predictions'],
            results['component_actual'],
            results['component_results']
        )):
            axes[i].plot(time, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
            axes[i].plot(time, pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
            axes[i].set_ylabel('Amplitude', fontsize=10)
            axes[i].set_title(f'IMF {i+1} (SSE={comp_result["sse"]:.2e}, RMSE={comp_result["rmse"]:.2e})',
                            fontsize=11, fontweight='bold')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, results['test_size'])
        
        # Final prediction
        axes[-1].plot(time, results['final_actual'], 'b-', 
                     label='Actual', linewidth=2, alpha=0.8)
        axes[-1].plot(time, results['final_prediction'], 'r--', 
                     label='Predicted', linewidth=2, alpha=0.8)
        axes[-1].set_xlabel('Time (samples)', fontsize=10)
        axes[-1].set_ylabel('Amplitude', fontsize=10)
        axes[-1].set_title(
            f'Final (SSE={results["final_sse"]:.4f}, RMSE={results["final_rmse"]:.4f})',
            fontsize=11, fontweight='bold'
        )
        axes[-1].legend(loc='upper right')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_xlim(0, results['test_size'])
        
        plt.suptitle('Paper 2: Prediction Results (Zhang et al., 2012)',
                    fontsize=14, fontweight='bold', y=0.9995)
        plt.tight_layout(rect=[0, 0, 1, 0.998])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_langevin_tracking(langevin_filter, save_path: Optional[str] = None):
        """
        Visualize Paper 3: Langevin Tracking Results
        
        Parameters:
        -----------
        langevin_filter : LangevinParticleFilter
            The filter object with results
        save_path : str, optional
            Path to save figure
        """
        if langevin_filter.state_estimates is None:
            raise ValueError("Must run particle_filter() first")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        if langevin_filter.price_index is not None:
            time = langevin_filter.price_index
        else:
            time = np.arange(langevin_filter.N_samples)
        
        # 1.Price and value estimate
        axes[0].plot(time, langevin_filter.prices, 'b-', label='Observed Price',
                    alpha=0.7, linewidth=1.5)
        axes[0].plot(time, langevin_filter.state_estimates[:, 0], 'r--',
                    label='Estimated Value', linewidth=2, alpha=0.8)
        axes[0].set_ylabel('Price', fontsize=11)
        axes[0].set_title('Price Tracking', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 2.Trend estimate
        axes[1].plot(time, langevin_filter.trend_estimates, 'g-',
                    label='Estimated Trend', linewidth=2)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].fill_between(time, 0, langevin_filter.trend_estimates,
                            where=(langevin_filter.trend_estimates > 0),
                            alpha=0.3, color='green', label='Positive Trend')
        axes[1].fill_between(time, 0, langevin_filter.trend_estimates,
                            where=(langevin_filter.trend_estimates < 0),
                            alpha=0.3, color='red', label='Negative Trend')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].set_title('Trend Estimation', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # 3.Trading signal
        signals = langevin_filter.generate_trading_signals()
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
        
        plt.suptitle('Paper 3: Langevin Dynamics (Christensen et al., 2012)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_backtest_results(results: Dict, save_path: Optional[str] = None):
        """
        Visualize Paper 3: Backtest Results
        
        Parameters:
        -----------
        results : dict
            Backtest results
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time = np.arange(len(results['strategy_returns']))
        
        # 1.Cumulative returns
        axes[0, 0].plot(time, results['cumulative_returns'],
                       'b-', linewidth=2.5)
        axes[0, 0].axhline(1, color='k', linestyle='--', alpha=0.5, label='Initial')
        axes[0, 0].set_xlabel('Time', fontsize=10)
        axes[0, 0].set_ylabel('Cumulative Return', fontsize=10)
        axes[0, 0].set_title('Strategy Performance', fontsize=11, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2.Return distribution
        axes[0, 1].hist(results['strategy_returns']*100, bins=50,
                       edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
        axes[0, 1].axvline(np.mean(results['strategy_returns'])*100,
                          color='g', linestyle='-', linewidth=2, label='Mean')
        axes[0, 1].set_xlabel('Return (%)', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].set_title('Return Distribution', fontsize=11, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3.Drawdown
        running_max = np.maximum.accumulate(results['cumulative_returns'])
        drawdown = (results['cumulative_returns'] - running_max) / running_max
        
        axes[1, 0].fill_between(time, 0, drawdown * 100,
                               alpha=0.5, color='red')
        axes[1, 0].plot(time, drawdown * 100, 'r-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time', fontsize=10)
        axes[1, 0].set_ylabel('Drawdown (%)', fontsize=10)
        axes[1, 0].set_title('Drawdown Over Time', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4.Performance summary
        summary_text = f"""
BACKTEST SUMMARY
{'='*40}

Returns:
  Total Return:      {results['total_return']*100:>8.2f}%
  Annual Return:     {results['annual_return']*100:>8.2f}%
  Annual Volatility: {results['annual_volatility']*100:>8.2f}%

Risk-Adjusted:
  Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}
  Max Drawdown:      {results['max_drawdown']*100:>8.2f}%

Trading:
  Win Rate:          {results['win_rate']*100:>8.1f}%
        """
        
        axes[1, 1].text(0.05, 0.5, summary_text,
                       transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        axes[1, 1].axis('off')
        
        plt.suptitle('Paper 3: Backtest Results (Christensen et al., 2012)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_integrated_system(system, save_path: Optional[str] = None):
        """
        Visualize Integrated System Results
        
        Parameters:
        -----------
        system : IntegratedTradingSystem
            The integrated system with results
        save_path : str, optional
            Path to save figure
        """
        if system.backtest_results is None:
            raise ValueError("Must run pipeline first")
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1.Portfolio cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        cumret = system.backtest_results['cumulative_returns']
        ax1.plot(cumret.index, cumret.values, 'b-', linewidth=2.5)
        ax1.axhline(1, color='k', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_ylabel('Cumulative Return', fontsize=11)
        ax1.set_title('Portfolio Performance', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2.Portfolio weights heatmap
        ax2 = fig.add_subplot(gs[1, :])
        weights_plot = system.weights.iloc[::max(1, len(system.weights)//100)]
        im = ax2.imshow(weights_plot.T, aspect='auto', cmap='RdYlGn',
                       vmin=-0.2, vmax=0.2, interpolation='nearest')
        ax2.set_yticks(range(len(system.tickers)))
        ax2.set_yticklabels(system.tickers, fontsize=9)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_title('Portfolio Weights Over Time', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Weight', fraction=0.046)
        
        # 3.Individual signals (sample 3 assets)
        sample_tickers = system.tickers[:min(3, len(system.tickers))]
        for idx, ticker in enumerate(sample_tickers):
            ax = fig.add_subplot(gs[2, idx])
            signal = system.signals[ticker]
            ax.plot(signal.index, signal.values, linewidth=1.5, color='purple')
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.fill_between(signal.index, 0, signal.values,
                           where=(signal.values > 0), alpha=0.3, color='green')
            ax.fill_between(signal.index, 0, signal.values,
                           where=(signal.values < 0), alpha=0.3, color='red')
            ax.set_title(f'{ticker} Signal', fontsize=10, fontweight='bold')
            ax.set_ylim(-1.2, 1.2)
            ax.grid(True, alpha=0.3)
        
        # 4.Drawdown
        ax_dd = fig.add_subplot(gs[3, 0])
        drawdown = system.backtest_results['drawdown']
        ax_dd.fill_between(drawdown.index, 0, drawdown.values * 100,
                          alpha=0.6, color='red')
        ax_dd.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1.5)
        ax_dd.set_xlabel('Time', fontsize=10)
        ax_dd.set_ylabel('Drawdown (%)', fontsize=10)
        ax_dd.set_title('Drawdown', fontsize=11, fontweight='bold')
        ax_dd.grid(True, alpha=0.3)
        
        # 5.Return distribution
        ax_dist = fig.add_subplot(gs[3, 1])
        returns = system.backtest_results['portfolio_returns']
        ax_dist.hist(returns * 100, bins=50, edgecolor='black', 
                    alpha=0.7, color='steelblue')
        ax_dist.axvline(0, color='r', linestyle='--', linewidth=2)
        ax_dist.axvline(returns.mean() * 100, color='g', linestyle='-', linewidth=2)
        ax_dist.set_xlabel('Return (%)', fontsize=10)
        ax_dist.set_ylabel('Frequency', fontsize=10)
        ax_dist.set_title('Return Distribution', fontsize=11, fontweight='bold')
        ax_dist.grid(True, alpha=0.3, axis='y')
        
        # 6.Performance summary
        ax_summary = fig.add_subplot(gs[3, 2])
        summary_text = f"""
INTEGRATED SYSTEM
{'='*35}

Portfolio Metrics:
  Total Return:  {system.backtest_results['total_return']*100:>7.2f}%
  Annual Return: {system.backtest_results['annual_return']*100:>7.2f}%
  Ann Volatility:{system.backtest_results['annual_volatility']*100:>7.2f}%
  Sharpe Ratio:  {system.backtest_results['sharpe_ratio']:>7.2f}
  Max Drawdown:  {system.backtest_results['max_drawdown']*100:>7.2f}%
  Win Rate:      {system.backtest_results['win_rate']*100:>7.1f}%

Configuration:
  Assets:        {system.N:>7}
  Trading Days:  {len(system.data['prices']):>7}
  
Papers:
   Paper 1: Toeplitz/DCT
   Paper 2: EMD + π-IF
   Paper 3: Langevin
        """
        
        ax_summary.text(0.05, 0.95, summary_text,
                       transform=ax_summary.transAxes,
                       fontsize=9, verticalalignment='top',
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        ax_summary.axis('off')
        
        plt.suptitle('Integrated Trading System: All Three Papers',
                    fontsize=15, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_all_results(system, save_dir: str = './plots'):
        """
        Generate all plots for the system
        
        Parameters:
        -----------
        system : IntegratedTradingSystem
            The system with all results
        save_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print(" GENERATING ALL VISUALIZATIONS")
        print("="*70)
        
        # 1.Paper 1 plots
        if system.toeplitz_analyzer is not None:
            print("\n1.Generating Paper 1 plots...")
            try:
                results1 = system.toeplitz_analyzer.compare_methods(Q=5)
                TradingSystemPlotter.plot_paper1_results(
                    system.toeplitz_analyzer, 
                    results1, 
                    f"{save_dir}/paper1_toeplitz_dct.png"
                )
            except Exception as e:
                print(f"    Paper 1 plots failed: {e}")
        
        # 2.Paper 2 plots (for first asset)
        if len(system.emd_analyzers) > 0:
            print("\n2.Generating Paper 2 plots...")
            first_ticker = list(system.emd_analyzers.keys())[0]
            analyzer2 = system.emd_analyzers[first_ticker]
            
            try:
                TradingSystemPlotter.plot_emd_decomposition(
                    analyzer2,
                    f"{save_dir}/paper2_emd_decomposition_{first_ticker}.png"
                )
            except Exception as e:
                print(f"    Paper 2 decomposition plots failed: {e}")
        
        # 3.Paper 3 plots (for first asset)
        if len(system.langevin_filters) > 0:
            print("\n3.Generating Paper 3 plots...")
            first_ticker = list(system.langevin_filters.keys())[0]
            langevin3 = system.langevin_filters[first_ticker]
            
            try:
                TradingSystemPlotter.plot_langevin_tracking(
                    langevin3,
                    f"{save_dir}/paper3_langevin_tracking_{first_ticker}.png"
                )
            except Exception as e:
                print(f"    Paper 3 tracking plots failed: {e}")
        
        # 4.Backtest results
        if system.backtest_results is not None:
            print("\n4.Generating backtest plots...")
            try:
                TradingSystemPlotter.plot_backtest_results(
                    system.backtest_results,
                    f"{save_dir}/backtest_results.png"
                )
            except Exception as e:
                print(f"    Backtest plots failed: {e}")
        
        # 5.Integrated system overview
        print("\n5.Generating integrated system plot...")
        try:
            TradingSystemPlotter.plot_integrated_system(
                system,
                f"{save_dir}/integrated_system_overview.png"
            )
        except Exception as e:
            print(f"    Integrated system plots failed: {e}")
        
        print("\n" + "="*70)
        print(f" ALL PLOTS SAVED TO: {save_dir}/")
        print("="*70)


# Example usage function
def create_all_plots(system):
    """
    Convenience function to create all plots
    
    Usage:
    ------
    from plotting_module import create_all_plots
    system, results = demo_integrated_system()
    create_all_plots(system)
    """
    plotter = TradingSystemPlotter()
    plotter.plot_all_results(system, save_dir='./plots')
    plt.show()


if __name__ == "__main__":
    print("This is a plotting module.Import it in your main script:")
    print("\nfrom plotting_module import TradingSystemPlotter, create_all_plots")
    print("\n# After running your system:")
    print("create_all_plots(system)")