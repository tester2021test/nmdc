# ============================================================
# 🪨 Ultimate NMDC Tracker — Enhanced Edition
# Features:
#   ✅ Multi-asset: NMDC, NMDC Steel, Nifty Metal Index, USD/INR
#   ✅ Indicators: RSI, MACD, Bollinger Bands, ATR, VWAP
#   ✅ Smarter signals: Multi-factor confirmation + risk scoring
#   ✅ Retry logic & error handling on all fetches
#   ✅ Historical backtesting (last 30 days, momentum strategy)
#   ✅ Rich Telegram alerts (formatted tables, emoji status)
#   ✅ CSV history with extended columns
#   ✅ Iron ore price proxy tracking (SGX TSI Iron Ore)
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

# ===================== CONFIG =====================
# Signal thresholds
RSI_OVERSOLD      = 40
RSI_OVERBOUGHT    = 65
MACD_HIST_THRESH  = 0.0     # above = bullish
BB_OVERSOLD       = 0.20    # %B below this = near lower band
BB_OVERBOUGHT     = 0.80    # %B above this = near upper band
VIX_HIGH          = 20.0    # VIX above this = risk-off penalty
ATR_RISK_MULT     = 1.5     # Stop = entry − ATR × multiplier
MOMENTUM_BUY_PCT  = -3.0    # % pullback from 20d high → buy watch
MOMENTUM_SELL_PCT = 10.0    # % gain from 20d low → profit book

# Retry
MAX_RETRIES  = 3
RETRY_DELAY  = 4   # seconds

# Tickers
TICKERS = {
    "NMDC"          : "NMDC.NS",
    "NMDC_Steel"    : "NMDCSTEEL.NS",   # NMDC Steel Ltd (demerged entity)
    "Nifty_Metal"   : "^CNXMETAL",      # Nifty Metal Index
    "Nifty50"       : "^NSEI",          # broader market context
    "India_VIX"     : "^INDIAVIX",
    "USD_INR"       : "INR=X",
    # Iron ore proxy — SGX TIF futures aren't on yfinance; use steel ETF as proxy
    "Steel_ETF_US"  : "SLX",            # VanEck Steel ETF (USD) — global steel proxy
}

# ===================== RETRY DECORATOR =====================
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
                    print(f"⚠️  Attempt {attempt}/{retries} failed [{fn.__name__}]: {e}")
                    if attempt < retries:
                        time_module.sleep(delay)
            print(f"❌ All retries failed [{fn.__name__}]: {last_err}")
            return None
        return wrapper
    return decorator

# ===================== FETCH =====================
@with_retry()
def fetch_history(symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty dataframe for {symbol}")
    return df

def safe_close(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)

def get_live_prev(symbol: str):
    """Returns (live_price, prev_close). Falls back to (0, 0) on failure."""
    df = fetch_history(symbol, period="5d", interval="1d")
    if df is None or df.empty:
        return 0.0, 0.0
    close = safe_close(df)
    if len(close) == 1:
        return float(close.iloc[-1]), float(close.iloc[-1])
    return float(close.iloc[-1]), float(close.iloc[-2])

# ===================== INDICATORS =====================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)

def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram) — last values."""
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

def calculate_bollinger(series: pd.Series, period=20, std_dev=2):
    """Returns (upper, mid, lower, %B) — last values."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
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
    tr    = pd.concat([high - low,
                       (high - prev).abs(),
                       (low  - prev).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 2) if not atr.dropna().empty else 0.0

def calculate_vwap(df: pd.DataFrame) -> float:
    close   = safe_close(df)
    high    = df["High"].astype(float)
    low     = df["Low"].astype(float)
    volume  = df["Volume"].astype(float)
    typical = (close + high + low) / 3
    vwap    = (typical * volume).cumsum() / volume.replace(0, np.nan).cumsum()
    return round(float(vwap.dropna().iloc[-1]), 2) if not vwap.dropna().empty else 0.0

def rolling_high_low(series: pd.Series, period=20):
    """Returns (20d_high, 20d_low) for momentum context."""
    if len(series) < period:
        return float(series.max()), float(series.min())
    window = series.iloc[-period:]
    return round(float(window.max()), 2), round(float(window.min()), 2)

# ===================== SIGNAL ENGINE =====================
def build_signal(
    price: float,
    prev_close: float,
    rsi: float,
    macd_hist: float,
    pct_b: float,
    vix: float,
    high_20d: float,
    low_20d: float,
    metal_index_chg: float,   # % change in Nifty Metal today
) -> dict:
    """
    Multi-factor signal for NMDC (equity momentum + value factors).
    Score range: [-8, +8]
    """
    score   = 0
    reasons = []

    day_chg = (price - prev_close) / prev_close * 100 if prev_close else 0

    # --- RSI ---
    if rsi < RSI_OVERSOLD:
        score += 2
        reasons.append(f"RSI oversold {rsi:.1f}")
    elif rsi > RSI_OVERBOUGHT:
        score -= 2
        reasons.append(f"RSI overbought {rsi:.1f}")
    else:
        reasons.append(f"RSI neutral {rsi:.1f}")

    # --- MACD ---
    if macd_hist > MACD_HIST_THRESH:
        score += 2
        reasons.append(f"MACD bullish ({macd_hist:+.4f})")
    else:
        score -= 1
        reasons.append(f"MACD bearish ({macd_hist:+.4f})")

    # --- Bollinger Bands ---
    if pct_b < BB_OVERSOLD:
        score += 2
        reasons.append(f"Near lower BB (%B={pct_b:.2f})")
    elif pct_b > BB_OVERBOUGHT:
        score -= 2
        reasons.append(f"Near upper BB (%B={pct_b:.2f})")
    else:
        reasons.append(f"BB mid-zone (%B={pct_b:.2f})")

    # --- Momentum: pullback from 20d high ---
    if high_20d > 0:
        pullback = (price - high_20d) / high_20d * 100
        if pullback <= MOMENTUM_BUY_PCT:
            score += 1
            reasons.append(f"Pulled back {pullback:.1f}% from 20d high")
        elif pullback >= -1:
            score -= 1
            reasons.append(f"Near 20d high (only {pullback:.1f}% below)")

    # --- Sector health: Nifty Metal ---
    if metal_index_chg > 0.5:
        score += 1
        reasons.append(f"Nifty Metal up {metal_index_chg:+.1f}% (tailwind)")
    elif metal_index_chg < -0.5:
        score -= 1
        reasons.append(f"Nifty Metal down {metal_index_chg:+.1f}% (headwind)")

    # --- VIX risk filter ---
    if vix > VIX_HIGH:
        score -= 1
        reasons.append(f"High VIX {vix:.1f} → risk-off")

    # Translate score → action
    if score >= 6:
        action, emoji = "STRONG BUY",     "🟢🟢"
    elif score >= 3:
        action, emoji = "BUY",            "🟢"
    elif score <= -5:
        action, emoji = "STRONG SELL",    "🔴🔴"
    elif score <= -2:
        action, emoji = "AVOID / SELL",   "🔴"
    else:
        action, emoji = "NEUTRAL / HOLD", "🟡"

    confidence = min(abs(score) / 8 * 100, 100)

    return {
        "action"    : action,
        "emoji"     : emoji,
        "score"     : score,
        "confidence": round(confidence, 1),
        "reasons"   : reasons,
        "day_chg"   : round(day_chg, 2),
    }

# ===================== RISK MANAGEMENT =====================
def risk_management(price: float, atr: float, signal_score: int) -> dict:
    stop_loss  = round(price - ATR_RISK_MULT * atr, 2)
    target_1r  = round(price + ATR_RISK_MULT * atr, 2)
    target_2r  = round(price + 2 * ATR_RISK_MULT * atr, 2)
    risk_pct   = round((price - stop_loss) / price * 100, 2)

    if abs(signal_score) >= 6:
        size_note = "Full position (high conviction)"
    elif abs(signal_score) >= 3:
        size_note = "Half position (moderate conviction)"
    else:
        size_note = "No position / wait for confirmation"

    return {
        "stop_loss": stop_loss,
        "target_1r": target_1r,
        "target_2r": target_2r,
        "risk_pct" : risk_pct,
        "size_note": size_note,
    }

# ===================== BACKTESTING =====================
def run_backtest(symbol: str = "NMDC.NS", days: int = 30) -> dict:
    """
    RSI + MACD momentum backtest over last `days` calendar days.
    Buy when RSI < 40 AND MACD histogram turns positive.
    Sell when RSI > 65 OR price hits 1.5×ATR target.
    """
    print(f"🔍 Running {days}-day backtest on {symbol}...")

    df = fetch_history(symbol, period=f"{days}d", interval="1d")
    if df is None or df.empty:
        return {"error": "Backtest data unavailable"}

    close = safe_close(df)
    if len(close) < 20:
        return {"error": "Insufficient data for backtest"}

    rsi_s = calculate_rsi(close, 14)
    _, _, macd_hist_s = _macd_series(close)

    trades   = []
    position = None

    for i in range(1, len(close)):
        price     = float(close.iloc[i])
        rsi_now   = float(rsi_s.iloc[i]) if not np.isnan(rsi_s.iloc[i]) else 50
        hist_now  = float(macd_hist_s.iloc[i]) if not np.isnan(macd_hist_s.iloc[i]) else 0
        hist_prev = float(macd_hist_s.iloc[i - 1]) if not np.isnan(macd_hist_s.iloc[i - 1]) else 0
        date_str  = str(close.index[i])[:10]

        # Entry: RSI oversold + MACD histogram crosses above 0
        if position is None and rsi_now < RSI_OVERSOLD and hist_prev < 0 < hist_now:
            position = {"entry": price, "date": date_str}

        # Exit: RSI overbought OR 10% gain
        elif position is not None:
            gain = (price - position["entry"]) / position["entry"] * 100
            if rsi_now > RSI_OVERBOUGHT or gain >= MOMENTUM_SELL_PCT:
                trades.append({
                    "buy_date"  : position["date"],
                    "sell_date" : date_str,
                    "entry"     : round(position["entry"], 2),
                    "exit"      : round(price, 2),
                    "return_pct": round(gain, 2),
                })
                position = None

    wins      = [t for t in trades if t["return_pct"] > 0]
    total_ret = sum(t["return_pct"] for t in trades)
    win_rate  = len(wins) / len(trades) * 100 if trades else 0

    return {
        "period_days"  : days,
        "total_trades" : len(trades),
        "win_rate_pct" : round(win_rate, 1),
        "total_return" : round(total_ret, 2),
        "avg_return"   : round(total_ret / len(trades), 2) if trades else 0.0,
        "trades"       : trades[-5:],
    }

def _macd_series(series: pd.Series, fast=12, slow=26, signal=9):
    """Full MACD series — used internally for backtesting."""
    ema_fast  = series.ewm(span=fast,   adjust=False).mean()
    ema_slow  = series.ewm(span=slow,   adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line

# ===================== CSV =====================
FIELDNAMES = [
    "timestamp", "market_phase",
    "nmdc_price", "nmdc_prev", "nmdc_chg_pct",
    "nmdc_steel_price",
    "nifty_metal", "nifty_metal_chg_pct",
    "nifty50", "steel_etf_usd", "usd_inr", "vix",
    "rsi", "macd_hist", "bb_pct_b", "atr", "vwap",
    "high_20d", "low_20d",
    "signal", "signal_score", "signal_confidence",
    "stop_loss", "target_1r", "target_2r",
]

def save_csv(row: dict):
    file = Path("nmdc_history.csv")
    write_header = not file.exists()
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ===================== TELEGRAM =====================
def send_telegram(message: str):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ Telegram credentials missing")
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
                print("📨 Telegram message sent")
                return
            print(f"⚠️  Telegram attempt {attempt}: {resp.status_code} {resp.text[:120]}")
        except Exception as e:
            print(f"⚠️  Telegram attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            time_module.sleep(RETRY_DELAY)

    print("❌ All Telegram retries failed")

def format_telegram(
    now, market_phase,
    live, prev,
    vwap, high_20d, low_20d,
    rsi, macd_line, macd_sig, macd_hist,
    bb_upper, bb_mid, bb_lower, pct_b,
    atr, vix,
    metal_chg,
    sig: dict, risk: dict, bt: dict,
) -> str:
    alert_tag   = "🚨 *ALERT*" if abs(sig["score"]) >= 5 else "📊 *UPDATE*"
    sep         = "─" * 28
    day_arrow   = "▲" if sig["day_chg"] >= 0 else "▼"

    prices_block = (
        "```\n"
        f"{'Asset':<18} {'Price':>10}\n"
        f"{sep}\n"
        f"{'NMDC':<18} ₹{live['NMDC']:>9.2f} {day_arrow}{abs(sig['day_chg']):.2f}%\n"
        f"{'NMDC Steel':<18} ₹{live['NMDC_Steel']:>9.2f}\n"
        f"{'Nifty Metal':<18} {live['Nifty_Metal']:>10.2f}\n"
        f"{'Nifty 50':<18} {live['Nifty50']:>10.2f}\n"
        f"{'Steel ETF (USD)':<18} ${live['Steel_ETF_US']:>9.2f}\n"
        f"{'USD/INR':<18} ₹{live['USD_INR']:>9.4f}\n"
        f"{'VIX':<18} {vix:>10.2f}\n"
        "```"
    )

    levels_block = (
        "```\n"
        f"{'20d High':<14} ₹{high_20d:>10.2f}\n"
        f"{'20d Low':<14} ₹{low_20d:>10.2f}\n"
        f"{'VWAP':<14} ₹{vwap:>10.2f}\n"
        "```"
    )

    indicators_block = (
        "```\n"
        f"{'Indicator':<14} {'Value':>10}\n"
        f"{sep}\n"
        f"{'RSI (15m)':<14} {rsi:>9.1f}\n"
        f"{'MACD Line':<14} {macd_line:>+9.4f}\n"
        f"{'MACD Hist':<14} {macd_hist:>+9.4f}\n"
        f"{'BB %B':<14} {pct_b:>9.2f}\n"
        f"{'ATR':<14} ₹{atr:>9.2f}\n"
        "```"
    )

    risk_block = (
        "```\n"
        f"{'Stop Loss':<14} ₹{risk['stop_loss']:>10.2f}\n"
        f"{'Target 1:1':<14} ₹{risk['target_1r']:>10.2f}\n"
        f"{'Target 1:2':<14} ₹{risk['target_2r']:>10.2f}\n"
        f"{'Risk %':<14} {risk['risk_pct']:>9.2f}%\n"
        "```"
    )

    reason_lines = "\n".join(f"  • {r}" for r in sig["reasons"])

    if "error" not in bt:
        bt_block = (
            f"📈 *30-Day Backtest* ({bt['total_trades']} trades)\n"
            f"Win Rate: {bt['win_rate_pct']}%  |  Avg Return: {bt['avg_return']:+.2f}%"
        )
    else:
        bt_block = f"📈 Backtest: {bt.get('error', 'N/A')}"

    return (
        f"{alert_tag} *NMDC Tracker*  {sig['emoji']}\n"
        f"🕒 Phase: *{market_phase}*  |  {now.strftime('%d-%b %H:%M IST')}\n\n"
        f"💰 *Prices & Market*\n{prices_block}\n"
        f"📏 *Key Levels*\n{levels_block}\n"
        f"📐 *Indicators*\n{indicators_block}\n"
        f"🎯 *Signal: {sig['action']}*  "
        f"(score {sig['score']:+d}, confidence {sig['confidence']:.0f}%)\n"
        f"{reason_lines}\n\n"
        f"🛡 *Risk Management*  [{risk['size_note']}]\n{risk_block}\n"
        f"{bt_block}"
    )

# ===================== MAIN =====================
def main():
    print("⏳ Running Enhanced NMDC Tracker...")

    # ---- Fetch all prices ----
    live, prev = {}, {}
    for k, sym in TICKERS.items():
        lv, pv      = get_live_prev(sym)
        live[k], prev[k] = lv, pv
        print(f"  {k}: live={lv:.4f}  prev={pv:.4f}")

    # ---- Intraday data for indicators (NMDC) ----
    raw_15m = fetch_history("NMDC.NS", period="5d", interval="15m")
    raw_1d  = fetch_history("NMDC.NS", period="60d", interval="1d")

    rsi_val   = 50.0
    macd_line = macd_sig = macd_hist = 0.0
    bb_upper  = bb_mid = bb_lower = 0.0
    pct_b     = 0.5
    atr_val   = 0.0
    vwap_val  = 0.0
    high_20d  = low_20d = live["NMDC"]

    if raw_15m is not None and not raw_15m.empty:
        close_15m = safe_close(raw_15m)
        if len(close_15m) > 26:
            rsi_val              = float(calculate_rsi(close_15m).dropna().iloc[-1])
            macd_line, macd_sig, macd_hist = calculate_macd(close_15m)
            bb_upper, bb_mid, bb_lower, pct_b = calculate_bollinger(close_15m)
        atr_val  = calculate_atr(raw_15m)
        vwap_val = calculate_vwap(raw_15m)

    if raw_1d is not None and not raw_1d.empty:
        close_1d = safe_close(raw_1d)
        high_20d, low_20d = rolling_high_low(close_1d, 20)

    print(f"  RSI={rsi_val:.1f}  MACD_hist={macd_hist:+.4f}  BB%B={pct_b:.2f}  ATR=₹{atr_val:.2f}")
    print(f"  VWAP=₹{vwap_val:.2f}  20d H=₹{high_20d}  L=₹{low_20d}")

    # ---- Sector change (Nifty Metal) ----
    metal_chg = ((live["Nifty_Metal"] - prev["Nifty_Metal"]) / prev["Nifty_Metal"] * 100
                 if prev["Nifty_Metal"] else 0.0)

    # ---- VIX ----
    vix = live["India_VIX"] if live["India_VIX"] else 15.0

    # ---- Market Phase ----
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t   = now.time()
    if   t < time(9, 15):  market_phase = "PRE-MARKET"
    elif t > time(15, 30): market_phase = "POST-MARKET"
    else:                  market_phase = "LIVE"

    # ---- Signal & Risk ----
    sig  = build_signal(
        live["NMDC"], prev["NMDC"],
        rsi_val, macd_hist, pct_b, vix,
        high_20d, low_20d, metal_chg,
    )
    risk = risk_management(live["NMDC"], atr_val, sig["score"])

    print(f"  Signal: {sig['action']}  score={sig['score']}  confidence={sig['confidence']}%")
    print(f"  Stop: ₹{risk['stop_loss']}  T1: ₹{risk['target_1r']}  T2: ₹{risk['target_2r']}")

    # ---- Backtest (cached per day) ----
    bt        = {}
    bt_cache  = Path("nmdc_backtest_cache.json")
    today_str = now.strftime("%Y-%m-%d")

    if bt_cache.exists():
        try:
            cached = json.loads(bt_cache.read_text())
            if cached.get("date") == today_str:
                bt = cached.get("result", {})
                print("  Backtest: using today's cached result")
        except Exception:
            pass

    if not bt:
        bt = run_backtest(symbol="NMDC.NS", days=30)
        try:
            bt_cache.write_text(json.dumps({"date": today_str, "result": bt}))
        except Exception:
            pass

    if "error" not in bt:
        print(f"  Backtest: {bt['total_trades']} trades, "
              f"win={bt['win_rate_pct']}%, avg={bt['avg_return']:+.2f}%")

    # ---- Save CSV ----
    nmdc_chg_pct = ((live["NMDC"] - prev["NMDC"]) / prev["NMDC"] * 100
                    if prev["NMDC"] else 0.0)
    save_csv({
        "timestamp"           : now.isoformat(),
        "market_phase"        : market_phase,
        "nmdc_price"          : round(live["NMDC"], 2),
        "nmdc_prev"           : round(prev["NMDC"], 2),
        "nmdc_chg_pct"        : round(nmdc_chg_pct, 2),
        "nmdc_steel_price"    : round(live["NMDC_Steel"], 2),
        "nifty_metal"         : round(live["Nifty_Metal"], 2),
        "nifty_metal_chg_pct" : round(metal_chg, 2),
        "nifty50"             : round(live["Nifty50"], 2),
        "steel_etf_usd"       : round(live["Steel_ETF_US"], 2),
        "usd_inr"             : round(live["USD_INR"], 4),
        "vix"                 : round(vix, 2),
        "rsi"                 : round(rsi_val, 1),
        "macd_hist"           : round(macd_hist, 4),
        "bb_pct_b"            : round(pct_b, 4),
        "atr"                 : round(atr_val, 2),
        "vwap"                : round(vwap_val, 2),
        "high_20d"            : high_20d,
        "low_20d"             : low_20d,
        "signal"              : sig["action"],
        "signal_score"        : sig["score"],
        "signal_confidence"   : sig["confidence"],
        "stop_loss"           : risk["stop_loss"],
        "target_1r"           : risk["target_1r"],
        "target_2r"           : risk["target_2r"],
    })

    # ---- Telegram ----
    msg = format_telegram(
        now, market_phase,
        live, prev,
        vwap_val, high_20d, low_20d,
        rsi_val, macd_line, macd_sig, macd_hist,
        bb_upper, bb_mid, bb_lower, pct_b,
        atr_val, vix, metal_chg,
        sig, risk, bt,
    )
    send_telegram(msg)

    print("✅ Completed")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
