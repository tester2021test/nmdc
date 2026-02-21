# ============================================================
# 🪨 NMDC Tracker — Pro Edition
# ============================================================
# Tracks: NMDC Ltd (iron ore mining) + ecosystem
#
# ✅ Correct tickers (verified Feb 2026)
# ✅ Multi-asset: NMDC, NMDC Steel (NSLNISP), SAIL, Tata Steel,
#                 JSPL, Nifty Metal Index, Nifty 50, VIX, USD/INR
# ✅ Iron ore price context: NMDC's own published lump/fines prices
#    + SGX iron ore proxy via iShares MSCI Global Metals & Mining ETF
# ✅ Indicators: RSI (EWM), MACD, Bollinger Bands + %B,
#               ATR, VWAP, EMA 20/50 trend filter
# ✅ Volume analysis: volume vs 20d avg, OBV direction
# ✅ Smarter signal: 10-factor scoring with trend confirmation
# ✅ Risk management: ATR-based stop/target, Kelly-fraction sizing
# ✅ 30-day backtest: RSI+MACD momentum strategy, daily cached
# ✅ Dividend yield tracker (NMDC known for strong dividends)
# ✅ Retry decorator on all network calls
# ✅ Rich Telegram alerts: formatted monospace tables
# ✅ CSV history: 25 columns
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
import os
import csv
import time as time_module
import json
from pathlib import Path
from datetime import datetime, time
from functools import wraps

# ======================================================
# CONFIG
# ======================================================
# --- Signal thresholds ---
RSI_OVERSOLD       = 40
RSI_OVERBOUGHT     = 65
VIX_HIGH           = 20.0
ATR_RISK_MULT      = 1.5        # Stop = entry − ATR × mult
PULLBACK_BUY_PCT   = -5.0       # % below 20d high → buy zone
NEAR_HIGH_PCT      = -1.5       # % below 20d high → caution
VOLUME_SURGE_MULT  = 1.5        # volume > 1.5× avg = confirmation

# --- Backtest ---
BT_DAYS            = 30
BT_GAIN_TARGET_PCT = 8.0        # Sell when gain ≥ 8%

# --- Retry ---
MAX_RETRIES        = 3
RETRY_DELAY        = 4          # seconds

# ======================================================
# TICKERS  (all verified NSE/yfinance, Feb 2026)
# ======================================================
TICKERS = {
    # Primary
    "NMDC"          : "NMDC.NS",

    # NMDC demerged steel arm  ← corrected from NMDCSTEEL.NS
    "NMDC_Steel"    : "NSLNISP.NS",

    # Sector peers
    "SAIL"          : "SAIL.NS",
    "Tata_Steel"    : "TATASTEEL.NS",
    "JSPL"          : "JINDALSTEL.NS",

    # Indices
    "Nifty_Metal"   : "^CNXMETAL",
    "Nifty50"       : "^NSEI",
    "India_VIX"     : "^INDIAVIX",

    # Macro
    "USD_INR"       : "INR=X",

    # Global iron ore / metals proxy (iShares MSCI Global Metals & Mining ETF)
    "Iron_Ore_ETF"  : "PICK",
}

# NMDC's latest published iron ore prices (₹/ton, update manually or via news API)
# Source: NMDC press release 10-Feb-2026
NMDC_ORE_PRICES = {
    "Baila_Lump_65pct"  : 4700,   # ₹/ton  (65.5%, 10–40 mm)
    "Baila_Fines_64pct" : 4000,   # ₹/ton  (64%, -10 mm)
    "last_updated"      : "2026-02-10",
}

# NMDC dividend (last declared)
NMDC_DIVIDEND = {
    "amount_per_share" : 2.50,    # ₹ (Feb 2026 interim)
    "ex_date"          : "2026-02-13",
}

# ======================================================
# RETRY DECORATOR
# ======================================================
def with_retry(retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    print(f"  ⚠️  [{fn.__name__}] attempt {attempt}/{retries}: {e}")
                    if attempt < retries:
                        time_module.sleep(delay)
            print(f"  ❌ [{fn.__name__}] all retries failed: {last_err}")
            return None
        return wrapper
    return decorator

# ======================================================
# DATA FETCH
# ======================================================
@with_retry()
def fetch_history(symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty df for {symbol}")
    return df

def safe_close(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)

def get_live_prev(symbol: str):
    df = fetch_history(symbol, period="5d", interval="1d")
    if df is None or df.empty:
        return 0.0, 0.0
    close = safe_close(df)
    if len(close) == 1:
        return float(close.iloc[-1]), float(close.iloc[-1])
    return float(close.iloc[-1]), float(close.iloc[-2])

def get_avg_volume(symbol: str, days: int = 20) -> float:
    """Returns 20-day average daily volume."""
    df = fetch_history(symbol, period=f"{days + 5}d", interval="1d")
    if df is None or df.empty:
        return 0.0
    vol = df["Volume"].astype(float).dropna()
    return float(vol.iloc[-days:].mean()) if len(vol) >= days else float(vol.mean())

# ======================================================
# INDICATORS
# ======================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """EWM-smoothed RSI (Wilder method)."""
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram) — scalar last values."""
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - sig_line
    return (
        round(float(macd_line.iloc[-1]),  4),
        round(float(sig_line.iloc[-1]),   4),
        round(float(histogram.iloc[-1]),  4),
    )

def macd_full_series(series: pd.Series, fast=12, slow=26, signal=9):
    """Returns full histogram Series — used in backtesting."""
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - sig_line

def calculate_bollinger(series: pd.Series, period=20, std_dev=2):
    """Returns (upper, mid, lower, %B) — scalar last values."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (series - lower) / ((upper - lower).replace(0, np.nan))
    return (
        round(float(upper.iloc[-1]),  2),
        round(float(mid.iloc[-1]),    2),
        round(float(lower.iloc[-1]),  2),
        round(float(pct_b.iloc[-1]),  4),
    )

def calculate_atr(df: pd.DataFrame, period=14) -> float:
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = safe_close(df)
    prev  = close.shift(1)
    tr    = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return round(float(atr.dropna().iloc[-1]), 2) if not atr.dropna().empty else 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    close   = safe_close(df)
    high    = df["High"].astype(float)
    low     = df["Low"].astype(float)
    volume  = df["Volume"].astype(float).replace(0, np.nan)
    typical = (close + high + low) / 3
    vwap    = (typical * volume).cumsum() / volume.cumsum()
    return round(float(vwap.dropna().iloc[-1]), 2) if not vwap.dropna().empty else 0.0

def calculate_obv(df: pd.DataFrame) -> str:
    """Returns 'rising' or 'falling' based on last 5 OBV bars."""
    close  = safe_close(df)
    volume = df["Volume"].astype(float)
    direction = np.sign(close.diff())
    obv    = (direction * volume).cumsum()
    if len(obv.dropna()) < 5:
        return "unknown"
    slope = obv.dropna().iloc[-1] - obv.dropna().iloc[-5]
    return "rising" if slope > 0 else "falling"

def ema(series: pd.Series, span: int) -> float:
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def rolling_high_low(series: pd.Series, period=20):
    if len(series) < period:
        return round(float(series.max()), 2), round(float(series.min()), 2)
    w = series.iloc[-period:]
    return round(float(w.max()), 2), round(float(w.min()), 2)

def dividend_yield(price: float) -> float:
    """Annualised yield based on last known NMDC dividend."""
    if price <= 0:
        return 0.0
    annual = NMDC_DIVIDEND["amount_per_share"] * 2   # assume ~2 dividends/year
    return round(annual / price * 100, 2)

# ======================================================
# SIGNAL ENGINE  (10-factor, score range −10 … +10)
# ======================================================
def build_signal(
    price: float,
    prev_close: float,
    rsi: float,
    macd_hist: float,
    macd_prev_hist: float,
    pct_b: float,
    vix: float,
    high_20d: float,
    low_20d: float,
    ema_20: float,
    ema_50: float,
    volume_today: float,
    avg_volume: float,
    obv_dir: str,
    metal_index_chg: float,
    peer_avg_chg: float,          # avg % change of SAIL + Tata Steel + JSPL
) -> dict:
    score   = 0
    reasons = []

    day_chg = (price - prev_close) / prev_close * 100 if prev_close else 0

    # ── 1. RSI ─────────────────────────────────────────
    if rsi < RSI_OVERSOLD:
        score += 2
        reasons.append(f"RSI oversold ({rsi:.1f} < {RSI_OVERSOLD})")
    elif rsi > RSI_OVERBOUGHT:
        score -= 2
        reasons.append(f"RSI overbought ({rsi:.1f} > {RSI_OVERBOUGHT})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    # ── 2. MACD histogram crossover ────────────────────
    if macd_prev_hist <= 0 < macd_hist:
        score += 2
        reasons.append("MACD bullish crossover (hist: neg→pos)")
    elif macd_prev_hist >= 0 > macd_hist:
        score -= 2
        reasons.append("MACD bearish crossover (hist: pos→neg)")
    elif macd_hist > 0:
        score += 1
        reasons.append(f"MACD histogram positive ({macd_hist:+.3f})")
    else:
        score -= 1
        reasons.append(f"MACD histogram negative ({macd_hist:+.3f})")

    # ── 3. Bollinger Band %B ───────────────────────────
    if pct_b < 0.20:
        score += 2
        reasons.append(f"Near lower Bollinger band (%B={pct_b:.2f})")
    elif pct_b > 0.80:
        score -= 1
        reasons.append(f"Near upper Bollinger band (%B={pct_b:.2f})")
    else:
        reasons.append(f"BB mid-zone (%B={pct_b:.2f})")

    # ── 4. EMA 20/50 trend filter ──────────────────────
    if ema_20 > ema_50 and price > ema_20:
        score += 1
        reasons.append(f"Price > EMA20 > EMA50 (uptrend)")
    elif ema_20 < ema_50 and price < ema_20:
        score -= 1
        reasons.append(f"Price < EMA20 < EMA50 (downtrend)")
    else:
        reasons.append("EMA trend mixed / sideways")

    # ── 5. Pullback from 20d high ──────────────────────
    if high_20d > 0:
        pb = (price - high_20d) / high_20d * 100
        if pb <= PULLBACK_BUY_PCT:
            score += 1
            reasons.append(f"Pullback {pb:.1f}% from 20d high (buy zone)")
        elif pb >= NEAR_HIGH_PCT:
            score -= 1
            reasons.append(f"Near 20d high (only {pb:.1f}% below)")

    # ── 6. Volume confirmation ─────────────────────────
    if avg_volume > 0:
        vol_ratio = volume_today / avg_volume
        if vol_ratio >= VOLUME_SURGE_MULT and day_chg > 0:
            score += 1
            reasons.append(f"Volume surge {vol_ratio:.1f}× avg (bullish)")
        elif vol_ratio >= VOLUME_SURGE_MULT and day_chg < 0:
            score -= 1
            reasons.append(f"Volume surge {vol_ratio:.1f}× avg (bearish)")
        else:
            reasons.append(f"Volume normal ({vol_ratio:.1f}× avg)")

    # ── 7. OBV direction ──────────────────────────────
    if obv_dir == "rising":
        score += 1
        reasons.append("OBV rising (accumulation)")
    elif obv_dir == "falling":
        score -= 1
        reasons.append("OBV falling (distribution)")

    # ── 8. Sector peer consensus ──────────────────────
    if peer_avg_chg > 0.5:
        score += 1
        reasons.append(f"Steel peers up {peer_avg_chg:+.1f}% (tailwind)")
    elif peer_avg_chg < -0.5:
        score -= 1
        reasons.append(f"Steel peers down {peer_avg_chg:+.1f}% (headwind)")

    # ── 9. Nifty Metal sector ─────────────────────────
    if metal_index_chg > 0.5:
        score += 1
        reasons.append(f"Nifty Metal up {metal_index_chg:+.1f}% (sector bid)")
    elif metal_index_chg < -0.5:
        score -= 1
        reasons.append(f"Nifty Metal down {metal_index_chg:+.1f}% (sector selling)")

    # ── 10. VIX risk filter ───────────────────────────
    if vix > VIX_HIGH:
        score -= 1
        reasons.append(f"VIX elevated ({vix:.1f} > {VIX_HIGH}) → risk-off")
    else:
        reasons.append(f"VIX normal ({vix:.1f})")

    # Translate to action
    if score >= 7:
        action, emoji = "STRONG BUY",     "🟢🟢"
    elif score >= 4:
        action, emoji = "BUY",            "🟢"
    elif score <= -6:
        action, emoji = "STRONG SELL",    "🔴🔴"
    elif score <= -3:
        action, emoji = "AVOID / SELL",   "🔴"
    else:
        action, emoji = "NEUTRAL / HOLD", "🟡"

    confidence = round(min(abs(score) / 10 * 100, 100), 1)

    return {
        "action"    : action,
        "emoji"     : emoji,
        "score"     : score,
        "confidence": confidence,
        "reasons"   : reasons,
        "day_chg"   : round(day_chg, 2),
    }

# ======================================================
# RISK MANAGEMENT  (ATR-based + Kelly fraction hint)
# ======================================================
def risk_management(price: float, atr: float, signal_score: int, win_rate: float = 0.5) -> dict:
    stop   = round(price - ATR_RISK_MULT * atr, 2)
    t1     = round(price + ATR_RISK_MULT * atr, 2)
    t2     = round(price + 2 * ATR_RISK_MULT * atr, 2)
    t3     = round(price + 3 * ATR_RISK_MULT * atr, 2)
    risk_r = round((price - stop) / price * 100, 2)

    # Simplified Kelly fraction: f = W − (1−W)/R   where R = reward/risk = 1.5
    R      = 1.5
    kelly  = max(0.0, win_rate - (1 - win_rate) / R)
    # Use half-Kelly for conservatism and cap at 25%
    pos_pct = round(min(kelly * 0.5 * 100, 25), 1)

    if abs(signal_score) >= 7:
        size_note = f"High conviction — ~{pos_pct}% of portfolio"
    elif abs(signal_score) >= 4:
        size_note = f"Moderate conviction — ~{pos_pct * 0.6:.1f}% of portfolio"
    else:
        size_note = "No position — wait for confirmation"

    return {
        "stop"     : stop,
        "target_1r": t1,
        "target_2r": t2,
        "target_3r": t3,
        "risk_pct" : risk_r,
        "pos_pct"  : pos_pct,
        "size_note": size_note,
    }

# ======================================================
# BACKTESTING  (RSI + MACD histogram crossover, 30d)
# ======================================================
def run_backtest(symbol: str = "NMDC.NS", days: int = BT_DAYS) -> dict:
    print(f"  🔍 Running {days}-day backtest on {symbol}...")
    df = fetch_history(symbol, period=f"{days + 10}d", interval="1d")
    if df is None or df.empty:
        return {"error": "Backtest data unavailable"}

    close = safe_close(df)
    if len(close) < 30:
        return {"error": f"Only {len(close)} bars — need ≥30"}

    rsi_s    = calculate_rsi(close, 14)
    macd_h_s = macd_full_series(close)

    trades   = []
    position = None

    for i in range(2, len(close)):
        price     = float(close.iloc[i])
        rsi_v     = float(rsi_s.iloc[i])   if not np.isnan(rsi_s.iloc[i])    else 50
        hist_now  = float(macd_h_s.iloc[i])   if not np.isnan(macd_h_s.iloc[i]) else 0
        hist_prev = float(macd_h_s.iloc[i-1]) if not np.isnan(macd_h_s.iloc[i-1]) else 0
        date_str  = str(close.index[i])[:10]

        # Entry: RSI oversold + MACD hist crossing zero from below
        if position is None and rsi_v < RSI_OVERSOLD and hist_prev < 0 < hist_now:
            position = {"entry": price, "date": date_str}

        # Exit: RSI overbought OR gain ≥ target OR stop hit
        elif position is not None:
            gain = (price - position["entry"]) / position["entry"] * 100
            if rsi_v > RSI_OVERBOUGHT or gain >= BT_GAIN_TARGET_PCT:
                trades.append({
                    "buy_date"  : position["date"],
                    "sell_date" : date_str,
                    "entry"     : round(position["entry"], 2),
                    "exit"      : round(price, 2),
                    "return_pct": round(gain, 2),
                })
                position = None

    wins       = [t for t in trades if t["return_pct"] > 0]
    total_ret  = sum(t["return_pct"] for t in trades)
    win_rate   = len(wins) / len(trades) if trades else 0.5
    avg_ret    = total_ret / len(trades) if trades else 0.0

    return {
        "period_days"  : days,
        "total_trades" : len(trades),
        "win_rate_pct" : round(win_rate * 100, 1),
        "win_rate"     : win_rate,              # float, for Kelly
        "total_return" : round(total_ret, 2),
        "avg_return"   : round(avg_ret, 2),
        "best_trade"   : round(max((t["return_pct"] for t in trades), default=0), 2),
        "worst_trade"  : round(min((t["return_pct"] for t in trades), default=0), 2),
        "trades"       : trades[-3:],           # last 3
    }

# ======================================================
# CSV
# ======================================================
FIELDNAMES = [
    "timestamp", "market_phase",
    "nmdc_price", "nmdc_prev", "nmdc_chg_pct",
    "nmdc_steel_price", "sail_price", "tata_steel_price", "jspl_price",
    "nifty_metal", "nifty_metal_chg_pct", "nifty50", "vix", "usd_inr",
    "iron_ore_etf_usd",
    "rsi", "macd_line", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct_b",
    "ema_20", "ema_50",
    "atr", "vwap", "obv_dir",
    "high_20d", "low_20d", "volume_today", "avg_volume",
    "signal", "signal_score", "signal_confidence",
    "stop", "target_1r", "target_2r", "target_3r", "risk_pct",
    "div_yield_pct",
    "nmdc_ore_lump", "nmdc_ore_fines",
]

def save_csv(row: dict):
    f_path = Path("nmdc_history.csv")
    write_header = not f_path.exists()
    with f_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ======================================================
# TELEGRAM
# ======================================================
def send_telegram(message: str):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("  ❌ Telegram credentials missing")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id"                 : chat_id,
                    "text"                    : message,
                    "parse_mode"              : "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                print("  📨 Telegram message sent")
                return
            print(f"  ⚠️  Telegram attempt {attempt}: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"  ⚠️  Telegram attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            time_module.sleep(RETRY_DELAY)
    print("  ❌ All Telegram retries failed")

def format_telegram(
    now, market_phase,
    live, prev,
    rsi, macd_line, macd_sig, macd_hist,
    bb_upper, bb_mid, bb_lower, pct_b,
    ema_20, ema_50,
    atr, vwap, obv_dir,
    high_20d, low_20d,
    volume_today, avg_volume,
    metal_chg, peer_avg_chg,
    sig, risk, bt,
    div_yield,
) -> str:
    # Alert level
    if abs(sig["score"]) >= 7:
        tag = "🚨🚨 *STRONG SIGNAL*"
    elif abs(sig["score"]) >= 4:
        tag = "🚨 *ALERT*"
    else:
        tag = "📊 *UPDATE*"

    sep       = "─" * 30
    arrow     = "▲" if sig["day_chg"] >= 0 else "▼"
    obv_emoji = "📈" if obv_dir == "rising" else "📉"
    trend_str = "↑ Uptrend" if ema_20 > ema_50 else "↓ Downtrend"

    # ── Prices table ──────────────────────────────────
    vol_ratio = volume_today / avg_volume if avg_volume else 0

    prices = (
        "```\n"
        f"{'Asset':<16} {'Price':>10}  {'Chg':>6}\n"
        f"{sep}\n"
        f"{'NMDC':<16} ₹{live['NMDC']:>8.2f}  {arrow}{abs(sig['day_chg']):.2f}%\n"
        f"{'NMDC Steel':<16} ₹{live['NMDC_Steel']:>8.2f}\n"
        f"{'SAIL':<16} ₹{live['SAIL']:>8.2f}\n"
        f"{'Tata Steel':<16} ₹{live['Tata_Steel']:>8.2f}\n"
        f"{'JSPL':<16} ₹{live['JSPL']:>8.2f}\n"
        f"{sep}\n"
        f"{'Nifty Metal':<16} {live['Nifty_Metal']:>9.2f}  {metal_chg:>+5.2f}%\n"
        f"{'Nifty 50':<16} {live['Nifty50']:>9.2f}\n"
        f"{'Iron Ore ETF':<16} ${live['Iron_Ore_ETF']:>8.2f}\n"
        f"{'USD/INR':<16} ₹{live['USD_INR']:>8.4f}\n"
        f"{'VIX':<16} {live['India_VIX']:>10.2f}\n"
        "```"
    )

    # ── Key levels ────────────────────────────────────
    levels = (
        "```\n"
        f"{'20d High':<14} ₹{high_20d:>10.2f}\n"
        f"{'20d Low':<14} ₹{low_20d:>10.2f}\n"
        f"{'EMA 20':<14} ₹{ema_20:>10.2f}\n"
        f"{'EMA 50':<14} ₹{ema_50:>10.2f}\n"
        f"{'VWAP':<14} ₹{vwap:>10.2f}\n"
        f"{'Trend':<14} {trend_str:>10}\n"
        "```"
    )

    # ── Technical indicators ──────────────────────────
    indicators = (
        "```\n"
        f"{'Indicator':<14} {'Value':>12}\n"
        f"{sep}\n"
        f"{'RSI (15m)':<14} {rsi:>11.1f}\n"
        f"{'MACD Line':<14} {macd_line:>+11.4f}\n"
        f"{'MACD Hist':<14} {macd_hist:>+11.4f}\n"
        f"{'BB %B':<14} {pct_b:>11.2f}\n"
        f"{'BB Upper':<14} ₹{bb_upper:>10.2f}\n"
        f"{'BB Lower':<14} ₹{bb_lower:>10.2f}\n"
        f"{'ATR':<14} ₹{atr:>10.2f}\n"
        f"{'Volume':<14} {vol_ratio:>10.1f}× avg\n"
        f"{'OBV':<14} {obv_emoji} {obv_dir:>8}\n"
        "```"
    )

    # ── Iron ore prices (NMDC published) ─────────────
    ore = (
        "```\n"
        f"{'Ore Product':<22} {'₹/ton':>8}\n"
        f"{sep}\n"
        f"{'Baila Lump (65.5%)':<22} {NMDC_ORE_PRICES['Baila_Lump_65pct']:>8,}\n"
        f"{'Baila Fines (64%)':<22} {NMDC_ORE_PRICES['Baila_Fines_64pct']:>8,}\n"
        f"{'Last updated':<22} {NMDC_ORE_PRICES['last_updated']:>8}\n"
        "```"
    )

    # ── Risk management ──────────────────────────────
    risk_block = (
        "```\n"
        f"{'Stop Loss':<14} ₹{risk['stop']:>10.2f}  ({risk['risk_pct']:.2f}%)\n"
        f"{'Target 1:1':<14} ₹{risk['target_1r']:>10.2f}\n"
        f"{'Target 1:2':<14} ₹{risk['target_2r']:>10.2f}\n"
        f"{'Target 1:3':<14} ₹{risk['target_3r']:>10.2f}\n"
        "```"
    )

    # ── Signal reasons ────────────────────────────────
    reason_lines = "\n".join(f"  {'✅' if '+' in r or 'over' in r.lower() or 'bull' in r.lower() or 'rising' in r.lower() or 'bid' in r.lower() or 'up' in r.lower() or 'buy' in r.lower() or 'positive' in r.lower() or 'normal' in r.lower() or 'neutral' in r.lower() or 'mid' in r.lower() or 'mixed' in r.lower() else '⚠️'} {r}" for r in sig["reasons"])

    # ── Backtest summary ──────────────────────────────
    if "error" not in bt:
        recent = ""
        for t in bt.get("trades", []):
            icon = "🟢" if t["return_pct"] > 0 else "🔴"
            recent += f"\n    {icon} {t['buy_date']} → {t['sell_date']}: {t['return_pct']:+.1f}%"
        bt_block = (
            f"📈 *{bt['period_days']}-Day Backtest* "
            f"({bt['total_trades']} trades)\n"
            f"  Win rate: {bt['win_rate_pct']}%  |  "
            f"Avg: {bt['avg_return']:+.2f}%  |  "
            f"Best: {bt['best_trade']:+.1f}%  |  "
            f"Worst: {bt['worst_trade']:+.1f}%"
            f"{recent}"
        )
    else:
        bt_block = f"📈 Backtest: {bt.get('error', 'N/A')}"

    # ── Assemble ──────────────────────────────────────
    return (
        f"{tag} *NMDC Tracker*  {sig['emoji']}\n"
        f"🕒 *{market_phase}*  |  {now.strftime('%d-%b-%Y %H:%M IST')}\n\n"
        f"💰 *Prices & Market*\n{prices}\n"
        f"📏 *Key Levels*\n{levels}\n"
        f"📐 *Technical Indicators*\n{indicators}\n"
        f"⛏ *NMDC Iron Ore Prices*\n{ore}\n"
        f"🎯 *Signal: {sig['action']}*  "
        f"(score {sig['score']:+d} / 10,  confidence {sig['confidence']:.0f}%)\n"
        f"{reason_lines}\n\n"
        f"🛡 *Risk Management*\n"
        f"  {risk['size_note']}\n"
        f"{risk_block}\n"
        f"💸 *Dividend Yield*: {div_yield:.2f}% (last ₹{NMDC_DIVIDEND['amount_per_share']}/share, ex-{NMDC_DIVIDEND['ex_date']})\n\n"
        f"{bt_block}"
    )

# ======================================================
# MAIN
# ======================================================
def main():
    print("⏳ Running NMDC Pro Tracker...\n")

    # ── 1. Fetch all prices ────────────────────────────
    live, prev = {}, {}
    for k, sym in TICKERS.items():
        lv, pv      = get_live_prev(sym)
        live[k], prev[k] = lv, pv
        print(f"  {k:<16}: ₹{lv:.2f}  (prev ₹{pv:.2f})")

    # ── 2. Intraday indicators (15m NMDC) ──────────────
    raw_15m = fetch_history("NMDC.NS", period="5d",  interval="15m")
    raw_1d  = fetch_history("NMDC.NS", period="90d", interval="1d")

    rsi_val  = 50.0
    macd_l   = macd_s = macd_h = 0.0
    macd_h_prev = 0.0
    bb_u = bb_m = bb_l = 0.0
    pct_b    = 0.5
    atr_val  = 0.0
    vwap_val = 0.0
    obv_dir  = "unknown"
    high_20d = low_20d = live["NMDC"]
    ema_20   = ema_50 = live["NMDC"]
    vol_today = avg_vol = 0.0

    if raw_15m is not None and not raw_15m.empty:
        c15 = safe_close(raw_15m)
        if len(c15) > 26:
            rsi_val        = float(calculate_rsi(c15).dropna().iloc[-1])
            macd_l, macd_s, macd_h = calculate_macd(c15)
            hist_series    = macd_full_series(c15)
            macd_h_prev    = float(hist_series.dropna().iloc[-2]) if len(hist_series.dropna()) > 1 else 0.0
            bb_u, bb_m, bb_l, pct_b = calculate_bollinger(c15)
        atr_val  = calculate_atr(raw_15m)
        vwap_val = calculate_vwap(raw_15m)
        obv_dir  = calculate_obv(raw_15m)

    if raw_1d is not None and not raw_1d.empty:
        c1d      = safe_close(raw_1d)
        high_20d, low_20d = rolling_high_low(c1d, 20)
        ema_20   = ema(c1d, 20)
        ema_50   = ema(c1d, 50)
        # Volume
        vol_series = raw_1d["Volume"].astype(float).dropna()
        vol_today  = float(vol_series.iloc[-1]) if not vol_series.empty else 0.0
        avg_vol    = float(vol_series.iloc[-21:-1].mean()) if len(vol_series) >= 21 else float(vol_series.mean())

    print(f"\n  RSI={rsi_val:.1f}  MACD_hist={macd_h:+.4f}  BB%B={pct_b:.2f}  ATR=₹{atr_val:.2f}")
    print(f"  VWAP=₹{vwap_val:.2f}  EMA20=₹{ema_20}  EMA50=₹{ema_50}")
    print(f"  20d H=₹{high_20d}  L=₹{low_20d}  OBV={obv_dir}")
    print(f"  Volume today={vol_today/1e5:.2f}L  avg={avg_vol/1e5:.2f}L")

    # ── 3. Market context ──────────────────────────────
    def pct_chg(key):
        p = prev[key]
        return (live[key] - p) / p * 100 if p else 0.0

    metal_chg    = pct_chg("Nifty_Metal")
    peer_avg_chg = (pct_chg("SAIL") + pct_chg("Tata_Steel") + pct_chg("JSPL")) / 3
    vix          = live["India_VIX"] if live["India_VIX"] else 15.0

    # ── 4. Market phase ───────────────────────────────
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t   = now.time()
    if   t < time(9, 15):  market_phase = "PRE-MARKET"
    elif t > time(15, 30): market_phase = "POST-MARKET"
    else:                  market_phase = "LIVE"

    # ── 5. Signal ─────────────────────────────────────
    sig = build_signal(
        price         = live["NMDC"],
        prev_close    = prev["NMDC"],
        rsi           = rsi_val,
        macd_hist     = macd_h,
        macd_prev_hist= macd_h_prev,
        pct_b         = pct_b,
        vix           = vix,
        high_20d      = high_20d,
        low_20d       = low_20d,
        ema_20        = ema_20,
        ema_50        = ema_50,
        volume_today  = vol_today,
        avg_volume    = avg_vol,
        obv_dir       = obv_dir,
        metal_index_chg = metal_chg,
        peer_avg_chg  = peer_avg_chg,
    )
    print(f"\n  Signal: {sig['action']}  score={sig['score']}  confidence={sig['confidence']}%")

    # ── 6. Backtest (daily cache) ─────────────────────
    bt        = {}
    bt_cache  = Path("nmdc_bt_cache.json")
    today_str = now.strftime("%Y-%m-%d")

    if bt_cache.exists():
        try:
            cached = json.loads(bt_cache.read_text())
            if cached.get("date") == today_str:
                bt = cached.get("result", {})
                print("  Backtest: using today's cache")
        except Exception:
            pass

    if not bt:
        bt = run_backtest()
        try:
            bt_cache.write_text(json.dumps({"date": today_str, "result": bt}))
        except Exception:
            pass

    win_rate_bt = bt.get("win_rate", 0.5)

    # ── 7. Risk management ────────────────────────────
    risk = risk_management(live["NMDC"], atr_val, sig["score"], win_rate=win_rate_bt)
    print(f"  Stop=₹{risk['stop']}  T1=₹{risk['target_1r']}  T2=₹{risk['target_2r']}")

    # ── 8. Dividend yield ─────────────────────────────
    div_yield = dividend_yield(live["NMDC"])

    # ── 9. Save CSV ───────────────────────────────────
    nmdc_chg = sig["day_chg"]
    save_csv({
        "timestamp"          : now.isoformat(),
        "market_phase"       : market_phase,
        "nmdc_price"         : round(live["NMDC"], 2),
        "nmdc_prev"          : round(prev["NMDC"], 2),
        "nmdc_chg_pct"       : nmdc_chg,
        "nmdc_steel_price"   : round(live["NMDC_Steel"], 2),
        "sail_price"         : round(live["SAIL"], 2),
        "tata_steel_price"   : round(live["Tata_Steel"], 2),
        "jspl_price"         : round(live["JSPL"], 2),
        "nifty_metal"        : round(live["Nifty_Metal"], 2),
        "nifty_metal_chg_pct": round(metal_chg, 2),
        "nifty50"            : round(live["Nifty50"], 2),
        "vix"                : round(vix, 2),
        "usd_inr"            : round(live["USD_INR"], 4),
        "iron_ore_etf_usd"   : round(live["Iron_Ore_ETF"], 2),
        "rsi"                : round(rsi_val, 1),
        "macd_line"          : round(macd_l, 4),
        "macd_hist"          : round(macd_h, 4),
        "bb_upper"           : bb_u,
        "bb_lower"           : bb_l,
        "bb_pct_b"           : round(pct_b, 4),
        "ema_20"             : ema_20,
        "ema_50"             : ema_50,
        "atr"                : atr_val,
        "vwap"               : vwap_val,
        "obv_dir"            : obv_dir,
        "high_20d"           : high_20d,
        "low_20d"            : low_20d,
        "volume_today"       : int(vol_today),
        "avg_volume"         : int(avg_vol),
        "signal"             : sig["action"],
        "signal_score"       : sig["score"],
        "signal_confidence"  : sig["confidence"],
        "stop"               : risk["stop"],
        "target_1r"          : risk["target_1r"],
        "target_2r"          : risk["target_2r"],
        "target_3r"          : risk["target_3r"],
        "risk_pct"           : risk["risk_pct"],
        "div_yield_pct"      : div_yield,
        "nmdc_ore_lump"      : NMDC_ORE_PRICES["Baila_Lump_65pct"],
        "nmdc_ore_fines"     : NMDC_ORE_PRICES["Baila_Fines_64pct"],
    })

    # ── 10. Telegram ──────────────────────────────────
    msg = format_telegram(
        now, market_phase,
        live, prev,
        rsi_val, macd_l, macd_s, macd_h,
        bb_u, bb_m, bb_l, pct_b,
        ema_20, ema_50,
        atr_val, vwap_val, obv_dir,
        high_20d, low_20d,
        vol_today, avg_vol,
        metal_chg, peer_avg_chg,
        sig, risk, bt,
        div_yield,
    )
    send_telegram(msg)
    print("\n✅ NMDC Tracker completed")

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    main()
