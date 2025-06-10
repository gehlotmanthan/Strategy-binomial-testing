import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Trading Strategy Part 
class MovingAverageCrossover:
    def __init__(self, symbol, fast_ma=20, slow_ma=50, initial_capital=10000):
        self.symbol = symbol
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.portfolio = None
        
    def fetch_data(self, period='10y'):
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"✓ Fetched {len(self.data)} days of data for {self.symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_moving_averages(self):
        if self.data is None:
            print("No data available. Please fetch data first.")
            return False
            
        self.data[f'MA_{self.fast_ma}'] = self.data['Close'].rolling(window=self.fast_ma).mean()
        self.data[f'MA_{self.slow_ma}'] = self.data['Close'].rolling(window=self.slow_ma).mean()
        self.data = self.data.dropna()
        
        print(f"✓ Calculated MA{self.fast_ma} and MA{self.slow_ma}")
        return True
    
    def generate_signals(self):
        if self.data is None:
            print("No data available. Please calculate moving averages first.")
            return False

        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Price'] = self.data['Close']
        self.signals[f'MA_{self.fast_ma}'] = self.data[f'MA_{self.fast_ma}']
        self.signals[f'MA_{self.slow_ma}'] = self.data[f'MA_{self.slow_ma}']
        
        self.signals['Signal'] = np.where(
            self.signals[f'MA_{self.fast_ma}'] > self.signals[f'MA_{self.slow_ma}'], 1, 0
        )
        
        self.signals['Position'] = self.signals['Signal'].diff()
        
        buy_signals = len(self.signals[self.signals['Position'] == 1])
        sell_signals = len(self.signals[self.signals['Position'] == -1])
        
        print(f"✓ Generated {buy_signals} buy signals and {sell_signals} sell signals")
        return True
    
    def backtest_strategy(self):
        if self.signals is None:
            print("No signals available. Please generate signals first.")
            return False
        
        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio['Price'] = self.signals['Price']
        self.portfolio['Signal'] = self.signals['Signal']
        
        self.portfolio['Daily_Return'] = self.portfolio['Price'].pct_change()
        self.portfolio['Strategy_Return'] = self.portfolio['Daily_Return'] * self.portfolio['Signal'].shift(1)
        self.portfolio['Cumulative_Market_Return'] = (1 + self.portfolio['Daily_Return']).cumprod()
        self.portfolio['Cumulative_Strategy_Return'] = (1 + self.portfolio['Strategy_Return']).cumprod()
        
        self.portfolio['Portfolio_Value'] = self.initial_capital * self.portfolio['Cumulative_Strategy_Return']
        self.portfolio['Market_Value'] = self.initial_capital * self.portfolio['Cumulative_Market_Return']
        
        print("✓ Backtesting completed")
        return True
    
    def calculate_performance_metrics(self):
        if self.portfolio is None:
            print("No portfolio data available. Please run backtest first.")
            return None
            
        returns = self.portfolio['Strategy_Return'].dropna()
        market_returns = self.portfolio['Daily_Return'].dropna()
        
        total_strategy_return = self.portfolio['Cumulative_Strategy_Return'].iloc[-1] - 1
        total_market_return = self.portfolio['Cumulative_Market_Return'].iloc[-1] - 1
        
        annual_strategy_return = (1 + total_strategy_return) ** (252 / len(returns)) - 1
        annual_market_return = (1 + total_market_return) ** (252 / len(market_returns)) - 1
        
        strategy_volatility = returns.std() * np.sqrt(252)
        market_volatility = market_returns.std() * np.sqrt(252)
        
        strategy_sharpe = (annual_strategy_return - 0.02) / strategy_volatility if strategy_volatility != 0 else 0
        market_sharpe = (annual_market_return - 0.02) / market_volatility if market_volatility != 0 else 0
        
        rolling_max = self.portfolio['Portfolio_Value'].expanding().max()
        drawdown = (self.portfolio['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Strategy Return': f"{total_strategy_return:.2%}",
            'Total Market Return': f"{total_market_return:.2%}",
            'Annual Strategy Return': f"{annual_strategy_return:.2%}",
            'Annual Market Return': f"{annual_market_return:.2%}",
            'Strategy Volatility': f"{strategy_volatility:.2%}",
            'Market Volatility': f"{market_volatility:.2%}",
            'Strategy Sharpe Ratio': f"{strategy_sharpe:.2f}",
            'Market Sharpe Ratio': f"{market_sharpe:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
        }
        
        return metrics
    
    def plot_strategy(self, figsize=(15, 10)):
        if self.signals is None or self.portfolio is None:
            print("No data to plot. Please run the complete strategy first.")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Price and Moving Averages with signals
        ax1.plot(self.signals.index, self.signals['Price'], label='Price', linewidth=1)
        ax1.plot(self.signals.index, self.signals[f'MA_{self.fast_ma}'], 
                label=f'MA{self.fast_ma}', alpha=0.7)
        ax1.plot(self.signals.index, self.signals[f'MA_{self.slow_ma}'], 
                label=f'MA{self.slow_ma}', alpha=0.7)
        
        # Mark buy and sell signals
        buy_signals = self.signals[self.signals['Position'] == 1]
        sell_signals = self.signals[self.signals['Position'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Price'], 
                   color='green', marker='^', s=100, label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['Price'], 
                   color='red', marker='v', s=100, label='Sell Signal')
        
        ax1.set_title(f'{self.symbol} - Moving Average Crossover Strategy')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value vs Buy & Hold
        ax2.plot(self.portfolio.index, self.portfolio['Portfolio_Value'], 
                label='Strategy', linewidth=2)
        ax2.plot(self.portfolio.index, self.portfolio['Market_Value'], 
                label='Buy & Hold', linewidth=2)
        ax2.set_title('Portfolio Value Comparison')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Returns
        ax3.plot(self.portfolio.index, 
                (self.portfolio['Cumulative_Strategy_Return'] - 1) * 100, 
                label='Strategy Return', linewidth=2)
        ax3.plot(self.portfolio.index, 
                (self.portfolio['Cumulative_Market_Return'] - 1) * 100, 
                label='Market Return', linewidth=2)
        ax3.set_title('Cumulative Returns Comparison')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_strategy(self, period='5y', plot=True):

        print(f"=== Running MA Crossover Strategy for {self.symbol} ===")
        print(f"Fast MA: {self.fast_ma}, Slow MA: {self.slow_ma}")
        print(f"Initial Capital: ${self.initial_capital:,}")
        print("-" * 50)
        
        if not self.fetch_data(period):
            return False
        
        if not self.calculate_moving_averages():
            return False
        
        if not self.generate_signals():
            return False
        
        if not self.backtest_strategy():
            return False
            
        metrics = self.calculate_performance_metrics()
        if metrics:
            print("\n=== PERFORMANCE METRICS ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        if plot:
            self.plot_strategy()
        
        return True
        
# Binomial Testing Part
class StrategyBinomialTest:
    def __init__(self, strategy):
        self.strategy = strategy
        self.trades = None
        self.winning_trades = None
        self.total_trades = None
        self.win_rate = None
        self.test_results = {}
        
    def extract_trades(self):
        if self.strategy.signals is None or self.strategy.portfolio is None:
            print("Strategy must be run first to extract trades")
            return False
            
        buy_signals = self.strategy.signals[self.strategy.signals['Position'] == 1].index
        sell_signals = self.strategy.signals[self.strategy.signals['Position'] == -1].index
        
        trades = []
        
        for buy_date in buy_signals:

            next_sell = sell_signals[sell_signals > buy_date]
            if len(next_sell) > 0:
                sell_date = next_sell[0]
                buy_price = self.strategy.signals.loc[buy_date, 'Price']
                sell_price = self.strategy.signals.loc[sell_date, 'Price']
                
                trade_return = (sell_price - buy_price) / buy_price
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': trade_return,
                    'profit_loss': sell_price - buy_price,
                    'is_winning': trade_return > 0
                })
        
        if len(buy_signals) > len(sell_signals):
            last_buy = buy_signals[-1]
            if last_buy not in [trade['buy_date'] for trade in trades]:
                buy_price = self.strategy.signals.loc[last_buy, 'Price']
                sell_price = self.strategy.signals['Price'].iloc[-1]  # Use last available price
                
                trade_return = (sell_price - buy_price) / buy_price
                trades.append({
                    'buy_date': last_buy,
                    'sell_date': self.strategy.signals.index[-1],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': trade_return,
                    'profit_loss': sell_price - buy_price,
                    'is_winning': trade_return > 0
                })
        
        self.trades = pd.DataFrame(trades)
        
        if len(self.trades) == 0:
            print("No complete trades found")
            return False
            
        self.winning_trades = len(self.trades[self.trades['is_winning']])
        self.total_trades = len(self.trades)
        self.win_rate = self.winning_trades / self.total_trades
        
        print(f"✓ Extracted {self.total_trades} trades")
        print(f"✓ Winning trades: {self.winning_trades}")
        print(f"✓ Win rate: {self.win_rate:.2%}")
        
        return True
    
    def perform_binomial_test(self, null_hypothesis=0.55, alpha=0.05):
        """
        Parameters:
        - null_hypothesis: Expected win rate under null hypothesis (default 0.55 for random)
        - alpha: Significance level (default 0.05)
        """
        if self.trades is None:
            print("Please extract trades first")
            return False
            
        try:
            result_two_sided = stats.binomtest(self.winning_trades, self.total_trades, null_hypothesis, alternative='two-sided')
            result_greater = stats.binomtest(self.winning_trades, self.total_trades, null_hypothesis, alternative='greater')
            result_less = stats.binomtest(self.winning_trades, self.total_trades, null_hypothesis, alternative='less')
            
            p_value = result_two_sided.pvalue
            p_value_greater = result_greater.pvalue
            p_value_less = result_less.pvalue
            
        except AttributeError:
        
            # Two-tailed test
            if self.winning_trades <= self.total_trades * null_hypothesis:
                p_value = 2 * stats.binom.cdf(self.winning_trades, self.total_trades, null_hypothesis)
            else:
                p_value = 2 * (1 - stats.binom.cdf(self.winning_trades - 1, self.total_trades, null_hypothesis))
            
            # One-tailed tests
            p_value_greater = 1 - stats.binom.cdf(self.winning_trades - 1, self.total_trades, null_hypothesis)
            p_value_less = stats.binom.cdf(self.winning_trades, self.total_trades, null_hypothesis)
        
        confidence_interval = stats.binom.interval(1-alpha, self.total_trades, self.win_rate)
        ci_lower = confidence_interval[0] / self.total_trades
        ci_upper = confidence_interval[1] / self.total_trades
        
        self.test_results = {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'null_hypothesis': null_hypothesis,
            'p_value_two_tailed': p_value,
            'p_value_greater': p_value_greater,
            'p_value_less': p_value_less,
            'alpha': alpha,
            'is_significant': p_value < alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'z_score': (self.win_rate - null_hypothesis) / np.sqrt(null_hypothesis * (1 - null_hypothesis) / self.total_trades)
        }
        
        return True
    
    def print_test_results(self):
        if not self.test_results:
            print("Please perform binomial test first")
            return
        
        print("\n" + "="*60)
        print("BINOMIAL TEST RESULTS")
        print("="*60)
        
        print(f"Total Trades: {self.test_results['total_trades']}")
        print(f"Winning Trades: {self.test_results['winning_trades']}")
        print(f"Observed Win Rate: {self.test_results['win_rate']:.4f} ({self.test_results['win_rate']:.2%})")
        print(f"Null Hypothesis (H0): Win Rate = {self.test_results['null_hypothesis']:.2%}")
        
        print(f"\nHypothesis Tests:")
        print(f"Two-tailed p-value: {self.test_results['p_value_two_tailed']:.6f}")
        print(f"One-tailed p-value (greater): {self.test_results['p_value_greater']:.6f}")
        print(f"One-tailed p-value (less): {self.test_results['p_value_less']:.6f}")
        
        print(f"\nStatistical Significance (α = {self.test_results['alpha']}):")
        if self.test_results['is_significant']:
            print("✓ SIGNIFICANT: The strategy's win rate is significantly different from random chance")
        else:
            print("✗ NOT SIGNIFICANT: Cannot reject null hypothesis - win rate not significantly different from random")
        
        print(f"\nConfidence Interval ({(1-self.test_results['alpha'])*100:.0f}%):")
        print(f"[{self.test_results['confidence_interval'][0]:.4f}, {self.test_results['confidence_interval'][1]:.4f}]")
        print(f"[{self.test_results['confidence_interval'][0]:.2%}, {self.test_results['confidence_interval'][1]:.2%}]")
        
        print(f"\nZ-Score: {self.test_results['z_score']:.4f}")
        
        print(f"\nInterpretation:")
        if self.test_results['win_rate'] > self.test_results['null_hypothesis']:
            if self.test_results['p_value_greater'] < self.test_results['alpha']:
                print("✓ Strategy appears to be PROFITABLE (significantly better than random)")
            else:
                print("Strategy shows positive bias but not statistically significant")
        else:
            if self.test_results['p_value_less'] < self.test_results['alpha']:
                print("✗ Strategy appears to be UNPROFITABLE (significantly worse than random)")
            else:
                print("Strategy shows negative bias but not statistically significant")
    
    def plot_binomial_analysis(self, figsize=(15, 12)):
        if not self.test_results or self.trades is None:
            print("Please extract trades and perform binomial test first")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        n = self.test_results['total_trades']
        p = self.test_results['null_hypothesis']
        
        x = np.arange(0, n+1)
        pmf = binom.pmf(x, n, p)
        
        ax1.bar(x, pmf, alpha=0.7, color='lightblue', edgecolor='blue')
        ax1.axvline(self.test_results['winning_trades'], color='red', linestyle='--', 
                   linewidth=2, label=f'Observed: {self.test_results["winning_trades"]}')
        ax1.axvline(n*p, color='green', linestyle='--', 
                   linewidth=2, label=f'Expected: {n*p:.1f}')
        
        ax1.set_title('Binomial Distribution (Null Hypothesis)')
        ax1.set_xlabel('Number of Winning Trades')
        ax1.set_ylabel('Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(self.trades['return'], bins=20, alpha=0.7, color='lightgreen', edgecolor='green')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.axvline(self.trades['return'].mean(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.trades["return"].mean():.3f}')
        
        ax2.set_title('Distribution of Trade Returns')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        cumulative_wins = self.trades['is_winning'].cumsum()
        cumulative_trades = np.arange(1, len(self.trades) + 1)
        cumulative_win_rate = cumulative_wins / cumulative_trades
        
        ax3.plot(cumulative_trades, cumulative_win_rate, linewidth=2, color='blue')
        ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='50% (Random)')
        ax3.axhline(self.test_results['win_rate'], color='green', linestyle='--', 
                   linewidth=2, label=f'Final: {self.test_results["win_rate"]:.2%}')
        
        ci_lower, ci_upper = self.test_results['confidence_interval']
        ax3.axhline(ci_lower, color='orange', linestyle=':', alpha=0.7, label='95% CI')
        ax3.axhline(ci_upper, color='orange', linestyle=':', alpha=0.7)
        ax3.fill_between(cumulative_trades, ci_lower, ci_upper, alpha=0.2, color='orange')
        
        ax3.set_title('Cumulative Win Rate Over Time')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        p_values = ['Two-tailed', 'Greater than 50%', 'Less than 50%']
        p_vals = [self.test_results['p_value_two_tailed'], 
                  self.test_results['p_value_greater'], 
                  self.test_results['p_value_less']]
        
        colors = ['blue', 'green', 'red']
        bars = ax4.bar(p_values, p_vals, color=colors, alpha=0.7)
        
        ax4.axhline(self.test_results['alpha'], color='red', linestyle='--', 
                   linewidth=2, label=f'α = {self.test_results["alpha"]}')
        
        for bar, p_val in zip(bars, p_vals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{p_val:.4f}', ha='center', va='bottom')
        
        ax4.set_title('P-Values for Different Tests')
        ax4.set_ylabel('P-Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(max(p_vals) * 1.2, self.test_results['alpha'] * 2))
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, null_hypothesis=0.5, alpha=0.05, plot=True):
        """Run complete binomial analysis"""
        print("\n=== RUNNING BINOMIAL TEST ANALYSIS ===")
        
        if not self.extract_trades():
            return False
        
        if not self.perform_binomial_test(null_hypothesis, alpha):
            return False
        
        self.print_test_results()
        
        if plot:
            self.plot_binomial_analysis()
        
        return True

# Usage
if __name__ == "__main__":
    # First run the strategy
    strategy = MovingAverageCrossover('META', fast_ma=20, slow_ma=50, initial_capital=10000)
    if strategy.run_complete_strategy(period='10y', plot=True):
        # Then perform binomial test
        binomial_test = StrategyBinomialTest(strategy)
        binomial_test.run_complete_analysis(null_hypothesis=0.55, alpha=0.05, plot=True)