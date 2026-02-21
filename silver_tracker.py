# ================================================================
# 🏦 PSU Bank Tracker — Pro Edition
# Tracks: PNB · Canara Bank (CANBK) · Bank of Baroda (BANKBARODA)
# ================================================================
# ✅ Verified tickers: PNB.NS · CANBK.NS · BANKBARODA.NS
# ✅ Multi-asset context: Nifty Bank, Nifty PSU Bank Index,
#                         SBI (sector bellwether), USD/INR, VIX
# ✅ Indicators per bank: RSI, MACD, Bollinger %B, EMA 20/50,
#                         ATR, VWAP, OBV, Volume surge
# ✅ Banking-specific factors: NIM proxy, NPA trend, P/B ratio
# ✅ 10-factor signal engine per bank (score −10 … +10)
# ✅ Risk management: ATR stop/targets + half-Kelly sizing
# ✅ 30-day RSI+MACD backtest per bank (daily cached)
# ✅ Side-by-side DECISION TABLE in Telegram → pick the best
# ✅ Dividend yield (live-price based, latest declared dividends)
# ✅ Retry decorator on all network calls (3× with backoff)
# ✅ CSV history: one row per run per bank + comparison summary
# ================================================================

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

# ================================================================
# CONFIG
# ================================================================
RSI_OVERSOLD        = 38
RSI_OVERBOUGHT      = 63
VIX_HIGH            = 20.0
ATR_RISK_MULT       = 1.5
PULLBACK_BUY_PCT    = -5.0      # % from 20d high → buy zone
NEAR_HIGH_PCT       = -1.5      # % from 20d high → caution
VOLUME_SURGE_MULT   = 1.5       # volume > 1.5× 20d avg = notable
BT_DAYS             = 30
BT_GAIN_TARGET_PCT  = 7.0       # backtest exit: gain ≥ 7%
MAX_RETRIES         = 3
RETRY_DELAY         = 4         # seconds

# ── Dividend data (latest declared, update when new dividends announced)
# Sources: IndMoney / StockAnalysis — Feb 2026
DIVIDENDS = {
    "PNB": {
        "amount"  : 2.90,           # ₹/share  (FY26 declared May 2025)
        "ex_date" : "2025-06-20",
        "frequency": "annual",
    },
    "CANBK": {
        "amount"  : 4.00,           # ₹/share  (FY25, ex Jun-2025)
        "ex_date" : "2025-06-13",
        "frequency": "annual",
    },
    "BANKBARODA": {
        "amount"  : 8.35,           # ₹/share  (FY26 declared May 2025)
        "ex_date" : "2025-06-06",
        "frequency": "annual",
    },
}

# ── Fundamentals snapshot (Q3 FY26 — update quarterly)
# NIM = Net Interest Margin; Gross NPA %; P/B = Price-to-Book
FUNDAMENTALS = {
    "PNB": {
        "nim_pct"      : 2.52,
        "gross_npa_pct": 3.19,
        "net_npa_pct"  : 0.32,
        "pb_ratio"     : 0.80,
        "roe_pct"      : 12.71,
        "crar_pct"     : 16.77,
        "q3fy26_profit_cr": 5556,
        "q3fy26_yoy_pct"  : 15.7,
        "last_updated" : "Q3 FY26 (Dec 2025)",
    },
    "CANBK": {
        "nim_pct"      : 2.88,
        "gross_npa_pct": 3.73,
        "net_npa_pct"  : 0.99,
        "pb_ratio"     : 0.85,
        "roe_pct"      : 18.20,
        "crar_pct"     : 16.25,
        "q3fy26_profit_cr": 4104,
        "q3fy26_yoy_pct"  : 11.8,
        "last_updated" : "Q3 FY26 (Dec 2025)",
    },
    "BANKBARODA": {
        "nim_pct"      : 3.10,
        "gross_npa_pct": 2.43,
        "net_npa_pct"  : 0.60,
        "pb_ratio"     : 0.90,
        "roe_pct"      : 16.80,
        "crar_pct"     : 16.50,
        "q3fy26_profit_cr": 5443,
        "q3fy26_yoy_pct"  : 4.5,
        "last_updated" : "Q3 FY26 (Dec 2025)",
    },
}

# ── Tickers
BANK_TICKERS = {
    "PNB"        : "PNB.NS",
    "CANBK"      : "CANBK.NS",
    "BANKBARODA" : "BANKBARODA.NS",
}

CONTEXT_TICKERS = {
    "Nifty_Bank"    : "^NSEBANK",
    "Nifty_PSUBank" : "^CNXPSUBANK",
    "SBI"           : "SBIN.NS",       # PSU bank bellwether
    "India_VIX"     : "^INDIAVIX",
    "USD_INR"       : "INR=X",
}

# ================================================================
# RETRY DECORATOR
# ================================================================
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

# ================================================================
# DATA FETCH
# ================================================================
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

# ================================================================
# INDICATORS
# ================================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    avg_g = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_l = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast  = series.ewm(span=fast, adjust=False).mean()
    ema_slow  = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - sig_line
    return (
        round(float(macd_line.iloc[-1]),  4),
        round(float(sig_line.iloc[-1]),   4),
        round(float(histogram.iloc[-1]),  4),
        round(float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0, 4),
    )

def macd_hist_series(series: pd.Series) -> pd.Series:
    ema_f = series.ewm(span=12, adjust=False).mean()
    ema_s = series.ewm(span=26, adjust=False).mean()
    ml    = ema_f - ema_s
    sl    = ml.ewm(span=9, adjust=False).mean()
    return ml - sl

def calculate_bollinger(series: pd.Series, period=20, std_dev=2):
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
    atr   = tr.rolling(period).mean()
    return round(float(atr.dropna().iloc[-1]), 2) if not atr.dropna().empty else 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    close   = safe_close(df)
    high    = df["High"].astype(float)
    low     = df["Low"].astype(float)
    volume  = df["Volume"].astype(float).replace(0, np.nan)
    typical = (close + high + low) / 3
    vwap    = (typical * volume).cumsum() / volume.cumsum()
    return round(float(vwap.dropna().iloc[-1]), 2) if not vwap.dropna().empty else 0.0

def calculate_obv_direction(df: pd.DataFrame) -> str:
    close  = safe_close(df)
    volume = df["Volume"].astype(float)
    obv    = (np.sign(close.diff()) * volume).cumsum()
    valid  = obv.dropna()
    if len(valid) < 5:
        return "unknown"
    return "rising" if float(valid.iloc[-1]) > float(valid.iloc[-5]) else "falling"

def ema_val(series: pd.Series, span: int) -> float:
    return round(float(series.ewm(span=span, adjust=False).mean().iloc[-1]), 2)

def rolling_high_low(series: pd.Series, period=20):
    w = series.iloc[-period:] if len(series) >= period else series
    return round(float(w.max()), 2), round(float(w.min()), 2)

def dividend_yield(price: float, bank_key: str) -> float:
    if price <= 0:
        return 0.0
    amount = DIVIDENDS[bank_key]["amount"]
    return round(amount / price * 100, 2)

# ================================================================
# SIGNAL ENGINE  (10 factors, score −10 … +10)
# ================================================================
def build_signal(
    bank_key: str,
    price: float,
    prev_close: float,
    rsi: float,
    macd_hist: float,
    macd_hist_prev: float,
    pct_b: float,
    atr: float,
    ema_20: float,
    ema_50: float,
    high_20d: float,
    volume_today: float,
    avg_volume: float,
    obv_dir: str,
    psubank_chg: float,     # % change Nifty PSU Bank index
    vix: float,
) -> dict:
    score   = 0
    reasons = []
    day_chg = (price - prev_close) / prev_close * 100 if prev_close else 0.0
    fund    = FUNDAMENTALS[bank_key]

    # ── 1. RSI
    if rsi < RSI_OVERSOLD:
        score += 2;  reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > RSI_OVERBOUGHT:
        score -= 2;  reasons.append(f"RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    # ── 2. MACD crossover
    if macd_hist_prev <= 0 < macd_hist:
        score += 2;  reasons.append("MACD bullish crossover ↑")
    elif macd_hist_prev >= 0 > macd_hist:
        score -= 2;  reasons.append("MACD bearish crossover ↓")
    elif macd_hist > 0:
        score += 1;  reasons.append(f"MACD positive ({macd_hist:+.3f})")
    else:
        score -= 1;  reasons.append(f"MACD negative ({macd_hist:+.3f})")

    # ── 3. Bollinger %B
    if pct_b < 0.20:
        score += 2;  reasons.append(f"Near lower BB (%B={pct_b:.2f})")
    elif pct_b > 0.80:
        score -= 1;  reasons.append(f"Near upper BB (%B={pct_b:.2f})")
    else:
        reasons.append(f"BB mid-zone (%B={pct_b:.2f})")

    # ── 4. EMA trend filter
    if ema_20 > ema_50 and price > ema_20:
        score += 1;  reasons.append("Price > EMA20 > EMA50 (uptrend)")
    elif ema_20 < ema_50 and price < ema_20:
        score -= 1;  reasons.append("Price < EMA20 < EMA50 (downtrend)")
    else:
        reasons.append("EMA trend mixed")

    # ── 5. Pullback from 20d high
    if high_20d > 0:
        pb = (price - high_20d) / high_20d * 100
        if pb <= PULLBACK_BUY_PCT:
            score += 1;  reasons.append(f"Pullback {pb:.1f}% from 20d high")
        elif pb >= NEAR_HIGH_PCT:
            score -= 1;  reasons.append(f"Near 20d high ({pb:.1f}%)")

    # ── 6. Volume confirmation
    if avg_volume > 0:
        vr = volume_today / avg_volume
        if vr >= VOLUME_SURGE_MULT and day_chg > 0:
            score += 1;  reasons.append(f"Volume surge {vr:.1f}× (bullish)")
        elif vr >= VOLUME_SURGE_MULT and day_chg < 0:
            score -= 1;  reasons.append(f"Volume surge {vr:.1f}× (bearish)")

    # ── 7. OBV
    if obv_dir == "rising":
        score += 1;  reasons.append("OBV rising (accumulation)")
    elif obv_dir == "falling":
        score -= 1;  reasons.append("OBV falling (distribution)")

    # ── 8. PSU Bank sector
    if psubank_chg > 0.5:
        score += 1;  reasons.append(f"PSU Bank index up {psubank_chg:+.1f}%")
    elif psubank_chg < -0.5:
        score -= 1;  reasons.append(f"PSU Bank index down {psubank_chg:+.1f}%")

    # ── 9. Fundamentals bonus/penalty
    # Good NIM (>2.8%) and falling NPAs are structural positives
    if fund["nim_pct"] >= 2.9:
        score += 1;  reasons.append(f"NIM healthy ({fund['nim_pct']:.2f}%)")
    if fund["gross_npa_pct"] <= 2.8:
        score += 1;  reasons.append(f"Gross NPA low ({fund['gross_npa_pct']:.2f}%)")
    elif fund["gross_npa_pct"] >= 5.0:
        score -= 1;  reasons.append(f"Gross NPA elevated ({fund['gross_npa_pct']:.2f}%)")

    # P/B < 0.85 = undervalued for a PSU bank
    if fund["pb_ratio"] < 0.85:
        score += 1;  reasons.append(f"P/B undervalued ({fund['pb_ratio']:.2f}×)")

    # ── 10. VIX
    if vix > VIX_HIGH:
        score -= 1;  reasons.append(f"VIX elevated ({vix:.1f}) → risk-off")
    else:
        reasons.append(f"VIX normal ({vix:.1f})")

    # ── Translate
    if score >= 7:    action, emoji = "STRONG BUY",    "🟢🟢"
    elif score >= 4:  action, emoji = "BUY",           "🟢"
    elif score <= -6: action, emoji = "STRONG SELL",   "🔴🔴"
    elif score <= -3: action, emoji = "AVOID / SELL",  "🔴"
    else:             action, emoji = "NEUTRAL / HOLD","🟡"

    confidence = round(min(abs(score) / 10 * 100, 100), 1)

    return {
        "action"    : action,
        "emoji"     : emoji,
        "score"     : score,
        "confidence": confidence,
        "reasons"   : reasons,
        "day_chg"   : round(day_chg, 2),
    }

# ================================================================
# RISK MANAGEMENT
# ================================================================
def risk_management(price: float, atr: float, score: int, win_rate: float = 0.5) -> dict:
    stop = round(price - ATR_RISK_MULT * atr, 2)
    t1   = round(price + ATR_RISK_MULT * atr, 2)
    t2   = round(price + 2 * ATR_RISK_MULT * atr, 2)
    t3   = round(price + 3 * ATR_RISK_MULT * atr, 2)
    rp   = round((price - stop) / price * 100, 2)

    R      = 1.5
    kelly  = max(0.0, win_rate - (1 - win_rate) / R)
    pos_pct = round(min(kelly * 0.5 * 100, 20), 1)   # cap 20% per stock

    if abs(score) >= 7:   size = f"Full (~{pos_pct}% of portfolio)"
    elif abs(score) >= 4: size = f"Half (~{pos_pct * 0.5:.1f}% of portfolio)"
    else:                 size = "No position — wait"

    return {
        "stop" : stop, "t1": t1, "t2": t2, "t3": t3,
        "rp"   : rp,   "size": size, "pos_pct": pos_pct,
    }

# ================================================================
# BACKTESTING
# ================================================================
def run_backtest(symbol: str, days: int = BT_DAYS) -> dict:
    df = fetch_history(symbol, period=f"{days + 15}d", interval="1d")
    if df is None or df.empty:
        return {"error": "No data"}
    close = safe_close(df)
    if len(close) < 30:
        return {"error": f"Only {len(close)} bars"}

    rsi_s  = calculate_rsi(close)
    hist_s = macd_hist_series(close)

    trades, position = [], None
    for i in range(2, len(close)):
        price     = float(close.iloc[i])
        rsi_v     = float(rsi_s.iloc[i])    if not np.isnan(rsi_s.iloc[i])    else 50
        hist_now  = float(hist_s.iloc[i])   if not np.isnan(hist_s.iloc[i])   else 0
        hist_prev = float(hist_s.iloc[i-1]) if not np.isnan(hist_s.iloc[i-1]) else 0
        date_str  = str(close.index[i])[:10]

        if position is None and rsi_v < RSI_OVERSOLD and hist_prev < 0 < hist_now:
            position = {"entry": price, "date": date_str}
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

    wins     = [t for t in trades if t["return_pct"] > 0]
    wr       = len(wins) / len(trades) if trades else 0.5
    total_r  = sum(t["return_pct"] for t in trades)
    avg_r    = total_r / len(trades) if trades else 0.0

    return {
        "total_trades": len(trades),
        "win_rate"    : round(wr, 4),
        "win_rate_pct": round(wr * 100, 1),
        "avg_return"  : round(avg_r, 2),
        "best"        : round(max((t["return_pct"] for t in trades), default=0), 2),
        "worst"       : round(min((t["return_pct"] for t in trades), default=0), 2),
        "trades"      : trades[-2:],
    }

# ================================================================
# ANALYSE ONE BANK — returns full result dict
# ================================================================
def analyse_bank(bank_key: str, symbol: str, psubank_chg: float, vix: float) -> dict:
    print(f"\n  📊 Analysing {bank_key} ({symbol})...")

    live, prev = get_live_prev(symbol)
    print(f"     Price: ₹{live:.2f}  Prev: ₹{prev:.2f}")

    # Intraday (15m)
    raw_15m = fetch_history(symbol, period="5d",  interval="15m")
    raw_1d  = fetch_history(symbol, period="90d", interval="1d")

    rsi_v = 50.0; macd_l = macd_s = macd_h = macd_hp = 0.0
    bb_u = bb_m = bb_l = 0.0; pct_b = 0.5
    atr_v = vwap_v = 0.0; obv_dir = "unknown"
    ema_20 = ema_50 = live; high_20d = low_20d = live
    vol_today = avg_vol = 0.0

    if raw_15m is not None and not raw_15m.empty:
        c15 = safe_close(raw_15m)
        if len(c15) > 26:
            rsi_v              = float(calculate_rsi(c15).dropna().iloc[-1])
            macd_l, macd_s, macd_h, macd_hp = calculate_macd(c15)
            bb_u, bb_m, bb_l, pct_b = calculate_bollinger(c15)
        atr_v  = calculate_atr(raw_15m)
        vwap_v = calculate_vwap(raw_15m)
        obv_dir = calculate_obv_direction(raw_15m)

    if raw_1d is not None and not raw_1d.empty:
        c1d     = safe_close(raw_1d)
        ema_20  = ema_val(c1d, 20)
        ema_50  = ema_val(c1d, 50)
        high_20d, low_20d = rolling_high_low(c1d, 20)
        vol_s   = raw_1d["Volume"].astype(float).dropna()
        vol_today = float(vol_s.iloc[-1]) if not vol_s.empty else 0.0
        avg_vol   = float(vol_s.iloc[-21:-1].mean()) if len(vol_s) >= 21 else float(vol_s.mean())

    sig  = build_signal(
        bank_key, live, prev, rsi_v, macd_h, macd_hp, pct_b,
        atr_v, ema_20, ema_50, high_20d,
        vol_today, avg_vol, obv_dir, psubank_chg, vix,
    )

    # Backtest (daily cache per bank)
    bt_cache  = Path(f"{bank_key.lower()}_bt_cache.json")
    today_str = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
    bt = {}
    if bt_cache.exists():
        try:
            cached = json.loads(bt_cache.read_text())
            if cached.get("date") == today_str:
                bt = cached.get("result", {})
        except Exception:
            pass
    if not bt:
        bt = run_backtest(symbol)
        try:
            bt_cache.write_text(json.dumps({"date": today_str, "result": bt}))
        except Exception:
            pass

    risk = risk_management(live, atr_v, sig["score"], bt.get("win_rate", 0.5))
    div_y = dividend_yield(live, bank_key)

    print(f"     Signal: {sig['action']}  score={sig['score']}  conf={sig['confidence']}%")
    print(f"     Stop=₹{risk['stop']}  T1=₹{risk['t1']}  T2=₹{risk['t2']}")

    return {
        "bank_key" : bank_key,
        "symbol"   : symbol,
        "live"     : live,
        "prev"     : prev,
        "rsi"      : round(rsi_v, 1),
        "macd_l"   : macd_l,
        "macd_h"   : macd_h,
        "pct_b"    : round(pct_b, 2),
        "bb_u"     : bb_u, "bb_l": bb_l,
        "ema_20"   : ema_20, "ema_50": ema_50,
        "atr"      : atr_v, "vwap": vwap_v,
        "high_20d" : high_20d, "low_20d": low_20d,
        "vol_today": vol_today, "avg_vol": avg_vol,
        "obv_dir"  : obv_dir,
        "sig"      : sig,
        "risk"     : risk,
        "bt"       : bt,
        "div_yield": div_y,
        "fund"     : FUNDAMENTALS[bank_key],
    }

# ================================================================
# CSV
# ================================================================
FIELDNAMES = [
    "timestamp", "market_phase", "bank",
    "price", "prev", "chg_pct",
    "rsi", "macd_hist", "bb_pct_b", "ema_20", "ema_50",
    "atr", "vwap", "obv_dir", "high_20d", "low_20d",
    "signal", "score", "confidence",
    "stop", "t1", "t2", "t3", "risk_pct",
    "div_yield_pct",
    "nim_pct", "gross_npa_pct", "pb_ratio", "roe_pct",
    "bt_win_rate", "bt_avg_return",
]

def save_csv(row: dict):
    f = Path("psu_bank_history.csv")
    write_header = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

# ================================================================
# TELEGRAM
# ================================================================
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
                json={"chat_id": chat_id, "text": message,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=15,
            )
            if resp.status_code == 200:
                print("  📨 Telegram sent")
                return
            print(f"  ⚠️  Telegram attempt {attempt}: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️  Telegram attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            time_module.sleep(RETRY_DELAY)
    print("  ❌ Telegram failed")

# ================================================================
# FORMAT: DECISION TABLE (the key output)
# ================================================================
def format_decision_table(results: list, now, market_phase: str, context: dict) -> str:
    sep = "─" * 38

    # ── Context block
    psu_chg  = context.get("psu_chg", 0.0)
    bnk_chg  = context.get("bnk_chg", 0.0)
    sbi_chg  = context.get("sbi_chg", 0.0)
    vix      = context.get("vix", 15.0)
    usd_inr  = context.get("usd_inr", 0.0)

    ctx = (
        "```\n"
        f"{'Nifty PSU Bank':<18} {psu_chg:>+8.2f}%\n"
        f"{'Nifty Bank':<18} {bnk_chg:>+8.2f}%\n"
        f"{'SBI (bellwether)':<18} {sbi_chg:>+8.2f}%\n"
        f"{'VIX':<18} {vix:>9.2f}\n"
        f"{'USD/INR':<18} ₹{usd_inr:>8.4f}\n"
        "```"
    )

    # ── Side-by-side comparison table
    banks = [r["bank_key"] for r in results]
    hdr   = f"{'Metric':<18} " + "  ".join(f"{b:<12}" for b in banks)
    rows  = []

    def row(label, vals):
        return f"{label:<18} " + "  ".join(f"{v:<12}" for v in vals)

    rows.append(row("Price (₹)",      [f"₹{r['live']:.2f}"           for r in results]))
    rows.append(row("Day Chg",        [f"{r['sig']['day_chg']:+.2f}%" for r in results]))
    rows.append(row("Signal",         [r['sig']['action'][:12]        for r in results]))
    rows.append(row("Score (/10)",    [f"{r['sig']['score']:+d}"       for r in results]))
    rows.append(row("Confidence",     [f"{r['sig']['confidence']:.0f}%"for r in results]))
    rows.append(f"{sep}")
    rows.append(row("RSI",            [f"{r['rsi']:.1f}"              for r in results]))
    rows.append(row("MACD Hist",      [f"{r['macd_h']:+.3f}"          for r in results]))
    rows.append(row("BB %B",          [f"{r['pct_b']:.2f}"            for r in results]))
    rows.append(row("OBV",            [r['obv_dir'][:8]               for r in results]))
    rows.append(f"{sep}")
    rows.append(row("ATR (₹)",        [f"₹{r['atr']:.2f}"            for r in results]))
    rows.append(row("Stop Loss",      [f"₹{r['risk']['stop']:.2f}"    for r in results]))
    rows.append(row("Target 1:1",     [f"₹{r['risk']['t1']:.2f}"      for r in results]))
    rows.append(row("Target 1:2",     [f"₹{r['risk']['t2']:.2f}"      for r in results]))
    rows.append(row("Risk %",         [f"{r['risk']['rp']:.2f}%"       for r in results]))
    rows.append(f"{sep}")
    rows.append(row("NIM %",          [f"{r['fund']['nim_pct']:.2f}%"  for r in results]))
    rows.append(row("Gross NPA %",    [f"{r['fund']['gross_npa_pct']:.2f}%"for r in results]))
    rows.append(row("P/B Ratio",      [f"{r['fund']['pb_ratio']:.2f}×" for r in results]))
    rows.append(row("RoE %",          [f"{r['fund']['roe_pct']:.1f}%"  for r in results]))
    rows.append(row("CRAR %",         [f"{r['fund']['crar_pct']:.1f}%" for r in results]))
    rows.append(row("Q3 Profit (Cr)", [f"₹{r['fund']['q3fy26_profit_cr']:,}"for r in results]))
    rows.append(row("Q3 YoY %",       [f"{r['fund']['q3fy26_yoy_pct']:+.1f}%"for r in results]))
    rows.append(f"{sep}")
    rows.append(row("Div Yield",      [f"{r['div_yield']:.2f}%"        for r in results]))
    rows.append(row("BT Win Rate",    [f"{r['bt'].get('win_rate_pct','-')}%" if 'error' not in r['bt'] else 'N/A' for r in results]))
    rows.append(row("BT Avg Ret",     [f"{r['bt'].get('avg_return',0):+.2f}%" if 'error' not in r['bt'] else 'N/A' for r in results]))

    table = "```\n" + hdr + "\n" + sep + "\n" + "\n".join(rows) + "\n```"

    # ── Verdict (ranked by score)
    ranked = sorted(results, key=lambda r: r["sig"]["score"], reverse=True)

    verdict_lines = []
    rank_emoji = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(ranked):
        fund  = r["fund"]
        div_y = r["div_yield"]
        line  = (
            f"{rank_emoji[i]} *{r['bank_key']}*  {r['sig']['emoji']}  "
            f"Score: {r['sig']['score']:+d}  |  "
            f"P/B: {fund['pb_ratio']:.2f}×  |  "
            f"NPA: {fund['gross_npa_pct']:.2f}%  |  "
            f"Div: {div_y:.2f}%\n"
            f"     → _{r['sig']['action']}_  ({r['sig']['confidence']:.0f}% confidence)\n"
            f"     Stop ₹{r['risk']['stop']}  |  T1 ₹{r['risk']['t1']}  |  T2 ₹{r['risk']['t2']}\n"
            f"     {r['risk']['size']}"
        )
        verdict_lines.append(line)

    verdict = "\n\n".join(verdict_lines)

    # Highlight best pick
    best = ranked[0]
    best_summary = (
        f"✅ *Best pick right now: {best['bank_key']}*\n"
        f"Signal score {best['sig']['score']:+d}/10, "
        f"confidence {best['sig']['confidence']:.0f}%  {best['sig']['emoji']}"
    )

    return (
        f"🏦 *PSU Bank Tracker* — Decision Board\n"
        f"🕒 *{market_phase}*  |  {now.strftime('%d-%b-%Y %H:%M IST')}\n\n"
        f"🌐 *Market Context*\n{ctx}\n"
        f"📊 *Side-by-Side Comparison*\n{table}\n"
        f"🎯 *Rankings & Verdicts*\n{verdict}\n\n"
        f"{best_summary}"
    )

def format_bank_detail(r: dict, now) -> str:
    """Individual deep-dive message for one bank."""
    bank = r["bank_key"]
    sig  = r["sig"]
    risk = r["risk"]
    bt   = r["bt"]
    fund = r["fund"]
    sep  = "─" * 28

    reasons = "\n".join(
        f"  {'✅' if any(w in r.lower() for w in ['over','bull','accum','up','healthy','low','normal','under','positive']) else '⚠️'} {r}"
        for r in sig["reasons"]
    )

    bt_block = "N/A"
    if "error" not in bt:
        recent = ""
        for t in bt.get("trades", []):
            icon = "🟢" if t["return_pct"] > 0 else "🔴"
            recent += f"\n    {icon} {t['buy_date']} → {t['sell_date']}: {t['return_pct']:+.1f}%"
        bt_block = (
            f"{bt['total_trades']} trades | WR {bt['win_rate_pct']}% | "
            f"Avg {bt['avg_return']:+.2f}% | "
            f"Best {bt['best']:+.1f}% | Worst {bt['worst']:+.1f}%"
            f"{recent}"
        )

    return (
        f"🏦 *{bank} Deep-Dive*  {sig['emoji']}\n\n"
        f"```\n"
        f"{'Price':<14} ₹{r['live']:>9.2f} ({sig['day_chg']:+.2f}%)\n"
        f"{'RSI (15m)':<14} {r['rsi']:>9.1f}\n"
        f"{'MACD Hist':<14} {r['macd_h']:>+9.4f}\n"
        f"{'BB %B':<14} {r['pct_b']:>9.2f}\n"
        f"{'EMA 20':<14} ₹{r['ema_20']:>9.2f}\n"
        f"{'EMA 50':<14} ₹{r['ema_50']:>9.2f}\n"
        f"{'ATR':<14} ₹{r['atr']:>9.2f}\n"
        f"{'VWAP':<14} ₹{r['vwap']:>9.2f}\n"
        f"{'OBV':<14} {r['obv_dir']:>9}\n"
        f"{'20d High':<14} ₹{r['high_20d']:>9.2f}\n"
        f"{'20d Low':<14} ₹{r['low_20d']:>9.2f}\n"
        f"{sep}\n"
        f"{'NIM %':<14} {fund['nim_pct']:>9.2f}%\n"
        f"{'Gross NPA':<14} {fund['gross_npa_pct']:>9.2f}%\n"
        f"{'Net NPA':<14} {fund['net_npa_pct']:>9.2f}%\n"
        f"{'P/B Ratio':<14} {fund['pb_ratio']:>9.2f}×\n"
        f"{'RoE':<14} {fund['roe_pct']:>9.2f}%\n"
        f"{'CRAR':<14} {fund['crar_pct']:>9.2f}%\n"
        f"{sep}\n"
        f"{'Stop Loss':<14} ₹{risk['stop']:>9.2f}\n"
        f"{'Target 1:1':<14} ₹{risk['t1']:>9.2f}\n"
        f"{'Target 1:2':<14} ₹{risk['t2']:>9.2f}\n"
        f"{'Target 1:3':<14} ₹{risk['t3']:>9.2f}\n"
        f"{'Risk %':<14} {risk['rp']:>9.2f}%\n"
        f"{'Div Yield':<14} {r['div_yield']:>9.2f}%\n"
        f"```\n"
        f"*Signal:* {sig['action']} (score {sig['score']:+d}/10, {sig['confidence']:.0f}%)\n"
        f"{reasons}\n\n"
        f"*Sizing:* {risk['size']}\n\n"
        f"*30d Backtest:* {bt_block}"
    )

# ================================================================
# MAIN
# ================================================================
def main():
    print("⏳ Running PSU Bank Pro Tracker...\n")

    # ── Market context
    print("  📡 Fetching market context...")
    ctx_live, ctx_prev = {}, {}
    for k, sym in CONTEXT_TICKERS.items():
        lv, pv = get_live_prev(sym)
        ctx_live[k], ctx_prev[k] = lv, pv

    def pct(key):
        p = ctx_prev[key]
        return (ctx_live[key] - p) / p * 100 if p else 0.0

    psubank_chg = pct("Nifty_PSUBank")
    vix         = ctx_live["India_VIX"] if ctx_live["India_VIX"] else 15.0
    context     = {
        "psu_chg"  : psubank_chg,
        "bnk_chg"  : pct("Nifty_Bank"),
        "sbi_chg"  : pct("SBI"),
        "vix"      : vix,
        "usd_inr"  : ctx_live["USD_INR"],
    }

    # ── Market phase
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t   = now.time()
    if   t < time(9, 15):  market_phase = "PRE-MARKET"
    elif t > time(15, 30): market_phase = "POST-MARKET"
    else:                  market_phase = "LIVE"

    # ── Analyse each bank
    results = []
    for bank_key, symbol in BANK_TICKERS.items():
        r = analyse_bank(bank_key, symbol, psubank_chg, vix)
        results.append(r)

        # Save individual CSV row
        save_csv({
            "timestamp"   : now.isoformat(),
            "market_phase": market_phase,
            "bank"        : bank_key,
            "price"       : r["live"],
            "prev"        : r["prev"],
            "chg_pct"     : r["sig"]["day_chg"],
            "rsi"         : r["rsi"],
            "macd_hist"   : r["macd_h"],
            "bb_pct_b"    : r["pct_b"],
            "ema_20"      : r["ema_20"],
            "ema_50"      : r["ema_50"],
            "atr"         : r["atr"],
            "vwap"        : r["vwap"],
            "obv_dir"     : r["obv_dir"],
            "high_20d"    : r["high_20d"],
            "low_20d"     : r["low_20d"],
            "signal"      : r["sig"]["action"],
            "score"       : r["sig"]["score"],
            "confidence"  : r["sig"]["confidence"],
            "stop"        : r["risk"]["stop"],
            "t1"          : r["risk"]["t1"],
            "t2"          : r["risk"]["t2"],
            "t3"          : r["risk"]["t3"],
            "risk_pct"    : r["risk"]["rp"],
            "div_yield_pct"   : r["div_yield"],
            "nim_pct"         : r["fund"]["nim_pct"],
            "gross_npa_pct"   : r["fund"]["gross_npa_pct"],
            "pb_ratio"        : r["fund"]["pb_ratio"],
            "roe_pct"         : r["fund"]["roe_pct"],
            "bt_win_rate"     : r["bt"].get("win_rate_pct", ""),
            "bt_avg_return"   : r["bt"].get("avg_return", ""),
        })

    # ── Send Telegram: decision table first (the one you'll act on)
    decision_msg = format_decision_table(results, now, market_phase, context)
    send_telegram(decision_msg)

    # ── Then send individual deep-dives
    for r in results:
        detail_msg = format_bank_detail(r, now)
        send_telegram(detail_msg)
        time_module.sleep(1)   # avoid rate limit

    print("\n✅ PSU Bank Tracker completed")

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    main()
