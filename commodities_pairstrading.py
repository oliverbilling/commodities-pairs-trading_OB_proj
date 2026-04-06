#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Pairs Trading — Commodity Futures
# Correlation + half-life screening, log-ratio z-score signals, walk-forward validation

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

TICKERS = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "PL=F",  # Platinum
    "PA=F",  # Palladium
    "CL=F",  # WTI Crude
    "BZ=F",  # Brent Crude
    "HO=F",  # Heating Oil
    "RB=F",  # RBOB Gasoline
    "NG=F",  # Natural Gas
    "HG=F",  # Copper
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F",  # Soybeans
    "ZM=F",  # Soybean Meal
    "ZL=F",  # Soybean Oil
    "KC=F",  # Coffee
    "SB=F",  # Sugar
    "CC=F",  # Cocoa
]

PERIOD         = "3y"
FREQ           = "1d"
IN_SAMPLE_FRAC = 0.70

MIN_CORR       = 0.65
MIN_HALF_LIFE  = 3
MAX_HALF_LIFE  = 45

ROLLING_WINDOW = 30
ENTRY_Z        = 1.9
EXIT_Z         = 0.1

MAX_DD_HALT    = 0.40
STOP_LOSS_Z    = 3.5

VOL_WINDOW     = 20
VOL_LOW        = 0.005
VOL_HIGH       = 0.040

NOTIONAL       = 10_000

PLOT = True

# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------

raw      = yf.download(TICKERS, period=PERIOD, interval=FREQ, progress=False)["Close"]
prices   = raw.dropna(how="all").ffill().dropna()
log_rets = np.log(prices).diff().dropna()

n     = len(prices)
split = int(n * IN_SAMPLE_FRAC)

p_in  = prices.iloc[:split]
p_out = prices.iloc[split:]
r_in  = log_rets.iloc[:split]

print(f"Universe:   {len(prices.columns)} contracts | {n} days")
print(f"In-sample:  {p_in.index[0].date()} -> {p_in.index[-1].date()}  ({split} days)")
print(f"Out-sample: {p_out.index[0].date()} -> {p_out.index[-1].date()}  ({n - split} days)\n")

# -------------------------------------------------------------------------
# Correlation heatmap (in-sample)
# -------------------------------------------------------------------------

corr    = r_in.corr()
dist    = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
linkage = hierarchy.linkage(hierarchy.distance.squareform(dist), method="average")
dendro  = hierarchy.dendrogram(linkage, labels=corr.columns, no_plot=True)
ordered = dendro["ivl"]

if PLOT:
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr.loc[ordered, ordered], annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("In-Sample Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Half-life (Ornstein-Uhlenbeck)
# -------------------------------------------------------------------------

def half_life(series):
    s     = series.dropna()
    ds    = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    ds, s_lag = ds.align(s_lag, join="inner")
    lam   = OLS(ds.values, add_constant(s_lag.values)).fit().params[1]
    return -np.log(2) / lam if lam < 0 else np.inf

# -------------------------------------------------------------------------
# Pair selection
# -------------------------------------------------------------------------

print(f"{'Pair':<14} {'corr':>6} {'half-life':>10}  result")
print("-" * 46)

selected = []

for a, b in combinations(prices.columns, 2):
    c = r_in[a].corr(r_in[b])
    if c < MIN_CORR:
        continue

    log_ratio = np.log(p_in[a] / p_in[b])
    hl        = half_life(log_ratio)

    if MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE:
        status = "pass"
        # rolling window = 3x half-life, clamped to [10, 30]
        window = int(np.clip(3 * hl, 10, 30))
        selected.append((a, b, hl, window))
    elif hl < MIN_HALF_LIFE:
        status = "fail -- too fast"
    else:
        status = "fail -- too slow"

    print(f"{a}/{b:<8} {c:>6.3f} {hl:>9.1f}d  {status}" + (f"  (window={window}d)" if status == "pass" else ""))

print(f"\n{len(selected)} pair(s) selected.\n")

# -------------------------------------------------------------------------
# Regime filter
# -------------------------------------------------------------------------

def regime_filter(price_slice, a, b):
    log_ratio = np.log(price_slice[a] / price_slice[b])
    daily_vol = log_ratio.diff().rolling(VOL_WINDOW).std()
    return (daily_vol >= VOL_LOW) & (daily_vol <= VOL_HIGH)

# -------------------------------------------------------------------------
# Backtest
# -------------------------------------------------------------------------

def backtest(price_slice, a, b, regime_mask, window):
    log_ratio = np.log(price_slice[a] / price_slice[b])
    r_mean    = log_ratio.rolling(window).mean()
    r_std     = log_ratio.rolling(window).std()
    z         = (log_ratio - r_mean) / r_std

    trades, equity, eq_curve = [], 0.0, []
    position, peak, halted   = None, 0.0, False

    for t in range(len(z)):
        zt = z.iloc[t]
        rt = log_ratio.iloc[t]

        if np.isnan(zt):
            eq_curve.append(np.nan)
            continue

        if equity > 0:
            peak = max(peak, equity)
        if peak > 0 and (peak - equity) / peak > MAX_DD_HALT:
            halted = True
        if halted:
            eq_curve.append(equity)
            continue

        if position is None:
            if not (t < len(regime_mask) and regime_mask.iloc[t]):
                eq_curve.append(equity)
                continue
            if zt > ENTRY_Z:
                position = {"side": -1, "entry": t, "entry_ratio": rt, "entry_date": z.index[t]}
            elif zt < -ENTRY_Z:
                position = {"side":  1, "entry": t, "entry_ratio": rt, "entry_date": z.index[t]}
        else:
            normal_exit = abs(zt) < EXIT_Z
            stop_hit    = (position["side"] ==  1 and zt < -STOP_LOSS_Z) or \
                          (position["side"] == -1 and zt >  STOP_LOSS_Z)

            if normal_exit or stop_hit:
                pnl = position["side"] * (rt - position["entry_ratio"])
                trades.append({
                    "pair":       f"{a}/{b}",
                    "side":       "long" if position["side"] == 1 else "short",
                    "entry_date": position["entry_date"],
                    "exit_date":  z.index[t],
                    "hold_days":  t - position["entry"],
                    "pnl":        round(pnl, 5),
                    "exit_type":  "stop" if stop_hit else "target",
                })
                equity  += pnl
                position = None

        unrealised = position["side"] * (rt - position["entry_ratio"]) if position else 0
        eq_curve.append(equity + unrealised)

    if position is not None:
        pnl = position["side"] * (log_ratio.iloc[-1] - position["entry_ratio"])
        trades.append({
            "pair":       f"{a}/{b}",
            "side":       "long" if position["side"] == 1 else "short",
            "entry_date": position["entry_date"],
            "exit_date":  z.index[-1],
            "hold_days":  len(z) - 1 - position["entry"],
            "pnl":        round(pnl, 5),
            "exit_type":  "target",
        })

    eq_s = pd.Series(eq_curve, index=z.index[-len(eq_curve):])
    return pd.DataFrame(trades), eq_s, z, halted


def stats(df, eq_curve):
    if df.empty:
        return {"trades": 0}
    p        = df["pnl"]
    roll_max = eq_curve.dropna().cummax()
    dd       = eq_curve.dropna() - roll_max
    stop_rate = f"{(df['exit_type'] == 'stop').mean():.0%}" if "exit_type" in df.columns else "n/a"
    return {
        "trades":    len(p),
        "total_pnl": f"{p.sum():.4f}",
        "win_rate":  f"{(p > 0).mean():.0%}",
        "sharpe":    f"{p.mean() / p.std():.3f}" if p.std() > 0 else "n/a",
        "max_dd":    f"{dd.min():.4f}",
        "avg_hold":  f"{df['hold_days'].mean():.1f}d",
        "stop_rate": stop_rate,
    }

# -------------------------------------------------------------------------
# Walk-forward
# -------------------------------------------------------------------------

all_summary = []

for a, b, hl, window in selected:

    regime_is = regime_filter(p_in, a, b)
    trades_is, eq_is, z_is, halted_is = backtest(p_in, a, b, regime_is, window)
    s_is = stats(trades_is, eq_is)
    s_is.update({"pair": f"{a}/{b}", "period": "In-sample", "halted": halted_is})

    sharpe_is = float(s_is["sharpe"]) if s_is.get("sharpe", "n/a") != "n/a" else -99
    if sharpe_is <= 0 or s_is.get("trades", 0) < 2:
        print(f"{a}/{b}  [screened -- Sharpe {sharpe_is:.3f}]\n")
        continue

    all_summary.append(s_is)

    flag = "  [HALTED]" if halted_is else ""
    print(f"{a}/{b}  [In-sample]{flag}")
    for k, v in s_is.items():
        if k not in ("pair", "period", "halted"):
            print(f"  {k:<12}: {v}")
    print()

    if PLOT and not trades_is.empty:
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        axes[0].plot(z_is, color="steelblue", linewidth=0.8)
        axes[0].axhline( ENTRY_Z, color="red",   linestyle="--", lw=0.8, label=f"+{ENTRY_Z}")
        axes[0].axhline(-ENTRY_Z, color="green", linestyle="--", lw=0.8, label=f"-{ENTRY_Z}")
        axes[0].axhline( EXIT_Z,  color="gray",  linestyle=":",  lw=0.8, label=f"+/-{EXIT_Z}")
        axes[0].set_title(f"{a}/{b}  [In-sample]")
        axes[0].legend(fontsize=8)
        axes[1].plot(eq_is, color="darkorange", linewidth=0.9)
        axes[1].fill_between(eq_is.index, eq_is, 0, where=(eq_is < 0), color="red", alpha=0.15)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Cumulative P&L")
        plt.tight_layout()
        plt.show()

    regime_os = regime_filter(p_out, a, b)
    trades_os, eq_os, z_os, halted_os = backtest(p_out, a, b, regime_os, window)
    s_os = stats(trades_os, eq_os)
    s_os.update({"pair": f"{a}/{b}", "period": "Out-of-sample", "halted": halted_os})
    all_summary.append(s_os)

    flag = "  [HALTED]" if halted_os else ""
    print(f"{a}/{b}  [Out-of-sample]{flag}")
    for k, v in s_os.items():
        if k not in ("pair", "period", "halted"):
            print(f"  {k:<12}: {v}")
    print()

    if PLOT and not trades_os.empty:
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        axes[0].plot(z_os, color="steelblue", linewidth=0.8)
        axes[0].axhline( ENTRY_Z, color="red",   linestyle="--", lw=0.8, label=f"+{ENTRY_Z}")
        axes[0].axhline(-ENTRY_Z, color="green", linestyle="--", lw=0.8, label=f"-{ENTRY_Z}")
        axes[0].axhline( EXIT_Z,  color="gray",  linestyle=":",  lw=0.8, label=f"+/-{EXIT_Z}")
        axes[0].set_title(f"{a}/{b}  [Out-of-sample]")
        axes[0].legend(fontsize=8)
        axes[1].plot(eq_os, color="darkorange", linewidth=0.9)
        axes[1].fill_between(eq_os.index, eq_os, 0, where=(eq_os < 0), color="red", alpha=0.15)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Cumulative P&L")
        plt.tight_layout()
        plt.show()

# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------

print("=" * 70)
print("Walk-Forward Summary")
print("=" * 70)
df_sum = pd.DataFrame(all_summary)
if not df_sum.empty:
    cols = ["pair", "period", "trades", "total_pnl", "win_rate", "sharpe", "max_dd", "avg_hold", "stop_rate"]
    print(df_sum[cols].to_string(index=False))
    print()

print("=" * 70)
print(f"P&L  (${NOTIONAL:,} notional per leg)")
print("=" * 70)
if not df_sum.empty:
    print(f"\n{'Pair':<14} {'Period':<16} {'Log P&L':>10} {'% Return':>10} {'$ P&L':>10}")
    print("-" * 64)
    for _, row in df_sum.iterrows():
        try:
            log_pnl = float(row["total_pnl"])
            pct_ret = (np.exp(log_pnl) - 1) * 100
            dollar  = pct_ret / 100 * NOTIONAL
            print(f"{row['pair']:<14} {row['period']:<16} {log_pnl:>10.4f} {pct_ret:>9.2f}% {dollar:>9.2f}")
        except (ValueError, TypeError):
            print(f"{row['pair']:<14} {row['period']:<16} {'n/a':>10}")


# In[ ]:




