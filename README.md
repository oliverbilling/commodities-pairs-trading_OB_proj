# Commodities Pairs Trading

Pairs trading strategy across 18 commodity futures. Screens for mean-reverting pairs using log-return correlation and Ornstein-Uhlenbeck half-life, generates signals from rolling z-scores of the log price ratio, and validates out-of-sample with a walk-forward split.

---

## How it works

**Pair selection** screens all combinations of 18 commodity futures on two criteria. First, in-sample log-return correlation must exceed 0.65. Second, the OU half-life of the log price ratio must fall between 3 and 45 days — pairs that revert too slowly tie up capital, pairs that revert too fast are noise. The rolling window used for the z-score is set adaptively at 3× the half-life, clamped to [10, 30] days.

**Signal** is the rolling z-score of `log(price_A / price_B)`. The log ratio is scale-free, so P&L is directly in log-return units and comparable across pairs regardless of their price levels. A position is opened when `|z| > 1.9` and closed when `|z| < 0.1`. A per-trade stop exits at `|z| > 3.5` if the spread continues to diverge after entry.

**Regime filter** blocks new entries when the 20-day realised vol of the log ratio falls outside [0.005, 0.040]. Below the floor the spread isn't moving enough to trade profitably; above the ceiling correlations tend to break down and mean reversion assumptions fail.

**Walk-forward validation** uses a 70/30 in-sample/out-of-sample split. Pairs are selected and parameters fixed on in-sample data only. Any pair with a negative in-sample Sharpe is screened before the out-of-sample run.

---

## Results (3-year backtest, GC/SI)

| Period | Trades | Win Rate | Sharpe | $ P&L ($10k/leg) |
|---|---|---|---|---|
| In-sample | 9 | 67% | 0.20 | $520 |
| Out-of-sample | 2 | 50% | 0.21 | $120 |

Sharpe is consistent in and out of sample with zero stops triggered, which is the key signal that the strategy generalises rather than overfits.

---

## Universe

Precious metals (GC, SI, PL, PA), energy (CL, BZ, HO, RB, NG), base metals (HG), grains (ZC, ZW, ZS, ZM, ZL), softs (KC, SB, CC).

---

## Run

```bash
pip install -r requirements.txt
python commodities_pairstrading.py
```

Pulls 3 years of daily data from Yahoo Finance via `yfinance`. All parameters are at the top of the file.
