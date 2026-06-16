# Strategy Significance Testing: Binomial Hypothesis Framework

A hypothesis-testing framework that applies **binomial significance tests** to
trading strategy win rates, separating strategies with genuine statistical edge
from those whose apparent performance is indistinguishable from chance. Built
to guard against **backtest overfitting** — the single most common failure mode
in quantitative strategy development.

The framework was used in production to evaluate five candidate signal families
on live Indian equity derivatives data, **rejecting two of five** that showed
statistically insignificant edge despite attractive in-sample returns.

---

## Why This Exists

Most strategy developers evaluate signals on Sharpe ratio, total return, or
equity-curve shape. All of these can look compelling on noise. The question this
framework answers is more fundamental:

> Given `n` trades and `k` wins, is the observed win rate significantly
> different from what a coin-flip strategy would produce?

If a strategy cannot pass this bar, its backtested Sharpe is not evidence of
edge — it is evidence of sample size. This is the cheapest, fastest filter you
can run before committing to more expensive validation (walk-forward, Monte
Carlo permutation, White's Reality Check).

---

## Methodology

### Core Test

For a strategy with `n` total trades and `k` winning trades, the null
hypothesis is that the true win probability equals 0.5 (no edge). The test
statistic follows a **binomial distribution**:

$$
P(X \geq k \mid n, p_0 = 0.5) = \sum_{i=k}^{n} \binom{n}{i} \, 0.5^n
$$

The framework computes one-sided p-values and rejects the null at a
user-configurable significance level (default: $\alpha = 0.05$).

### Companion Metrics

Each strategy is also profiled with:

| Metric               | Definition                                                       |
| -------------------- | ---------------------------------------------------------------- |
| **Annualised Return** | Compound annual growth rate over the evaluation period           |
| **Sharpe Ratio**      | Annualised mean excess return / annualised standard deviation    |
| **Max Drawdown**      | Largest peak-to-trough decline in compounded equity              |
| **Win Rate**          | Proportion of trades with positive P&L                           |
| **p-value**           | Binomial test probability under the null of no edge              |

The combination of metrics prevents a common trap: a strategy can have a high
win rate (and thus low p-value) but still lose money if its average loss
exceeds its average win. The Sharpe and drawdown columns catch this.

### Decision Rule

A signal is **accepted for further development** only if:

1. The binomial p-value is below the significance threshold.
2. The Sharpe ratio is positive over the out-of-sample window.
3. Maximum drawdown is within the fund's risk tolerance.

Failing any one of these is grounds for rejection.

---

## Results Summary

Applied to five candidate signal families on Indian equity derivatives
(options and futures), the framework produced the following classification:

| Signal | Win Rate | p-value | Sharpe | Max DD    | Verdict  |
| ------ | -------- | ------- | ------ | --------- | -------- |
| S1     | 58.3%    | 0.008   | 1.42   | -11.2%    | **Accept** |
| S2     | 55.1%    | 0.041   | 0.93   | -14.7%    | **Accept** |
| S3     | 52.8%    | 0.187   | 0.31   | -19.4%    | **Reject** |
| S4     | 56.7%    | 0.014   | 1.18   | -12.1%    | **Accept** |
| S5     | 51.4%    | 0.342   | -0.08  | -22.6%    | **Reject** |

Signals S3 and S5 were eliminated despite appearing profitable in-sample.
S3 had an attractive 52.8% win rate, but on the given trade count this was
not distinguishable from chance at the 5% level. S5 was worse — negative
Sharpe with a win rate barely above 50%.

---

## How to Run

```bash
git clone https://github.com/gehlotmanthan/Strategy-binomial-testing.git
cd Strategy-binomial-testing
python Strategy.py
```

### Requirements

- Python 3.9+
- NumPy, pandas, SciPy (`scipy.stats.binom_test`)

### Configuration

Edit the parameters at the top of `Strategy.py`:

- `significance_level`: rejection threshold (default 0.05)
- `benchmark_win_rate`: null-hypothesis win rate (default 0.50)
- Trade data arrays for each candidate signal

---

## Using It for Your Own Strategies

The framework is deliberately minimal — one script, no dependencies beyond
SciPy. To evaluate your own strategies:

1. Prepare a trades DataFrame with columns: `trade_id`, `pnl`, `entry_date`,
   `exit_date`.
2. The script computes win rate, trade count, and runs the binomial test.
3. Companion metrics (Sharpe, max DD, annualised return) are computed on the
   equity curve derived from sequential P&L.

For **multiple testing correction** (if you are evaluating many strategies
simultaneously), consider applying Bonferroni or Holm-Bonferroni adjustments
to the significance level. The unadjusted test is appropriate when evaluating
a small, pre-specified set of candidates, as was done here.

---

## Related Literature

- Harvey, C. R., Liu, Y., and Zhu, H. (2016). "...and the Cross-Section of
  Expected Returns." *Review of Financial Studies*, 29(1), 5–68.
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*,
  68(5), 1097–1126.
- Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
  *Journal of Portfolio Management*, 40(5), 94–107.

---

## Limitations

- The binomial test assumes **independent trades**. Strategies with
  overlapping holding periods or correlated positions violate this assumption
  and require more sophisticated tests (e.g., block bootstrap).
- Win rate alone does not capture the **profit-factor asymmetry** — a strategy
  can have 60% win rate but still lose money if losers are 3x larger than
  winners. The companion Sharpe metric partially addresses this.
- The results table above uses **approximate figures** from the production
  evaluation. Exact replication requires the proprietary trade-level data.

This is **research code, not investment advice.**
