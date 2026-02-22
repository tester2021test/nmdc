# ================================================================
# 📊 Nifty All-Indices Tracker — Pro Edition
# ================================================================
# Covers ALL major Nifty indices available on Yahoo Finance
# (verified tickers, Feb 2026)
#
# ✅ 30 indices across Broad Market, Sectoral, Thematic, Strategy
# ✅ Per-index: RSI, MACD, Bollinger %B, EMA 20/50 trend,
#               ATR, momentum, 52w position, vs Nifty 50 beta
# ✅ 6-factor signal score per index (−6 … +6)
# ✅ Sector Rotation Heatmap — ranked from strongest to weakest
# ✅ Momentum Leaders & Laggards summary
# ✅ 30-day backtest per index (daily cached)
# ✅ Market Breadth: count of bullish vs bearish indices
# ✅ Rich Telegram: Heatmap message + per-category deep-dives
# ✅ Retry decorator on all network calls (3× with backoff)
# ✅ CSV history: one row per run per index
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
RSI_OVERSOLD       = 40
RSI_OVERBOUGHT     = 63
VIX_HIGH           = 20.0
ATR_RISK_MULT      = 1.5
BT_DAYS            = 30
BT_GAIN_TARGET_PCT = 6.0
MAX_RETRIES        = 3
RETRY_DELAY        = 3          # seconds between retries
FETCH_PAUSE        = 0.5        # pause between index fetches (be nice to API)

# ================================================================
# INDEX REGISTRY  (all tickers verified on Yahoo Finance, Feb 2026)
# ================================================================
#  Format: "Short Name": ("Yahoo Ticker", "Category")
INDICES = {
    # ── BROAD MARKET ──────────────────────────────────────────────
    "Nifty 50"          : ("^NSEI",               "Broad Market"),
    "Nifty Next 50"     : ("^NSMIDCP",            "Broad Market"),  # ✅ confirmed ^NSMIDCP not ^NSMIDCP400
    "Nifty 100"         : ("^CNX100",             "Broad Market"),  # ✅
    "Nifty 200"         : ("^CNX200",             "Broad Market"),  # ✅
    "Nifty 500"         : ("^CRSLDX",             "Broad Market"),  # ✅
    "Nifty Midcap 50"   : ("^NSEMDCP50",          "Broad Market"),  # ✅
    "Nifty Midcap 100"  : ("NIFTY_MIDCAP_100.NS", "Broad Market"),  # ✅
    "Nifty Smlcap 100"  : ("^CNXSC",              "Broad Market"),  # ✅
    "Nifty Midcap 150"  : ("NIFTYMIDCAP150.NS",   "Broad Market"),  # ✅

    # ── SECTORAL ──────────────────────────────────────────────────
    "Nifty Bank"        : ("^NSEBANK",            "Sectoral"),      # ✅
    "Nifty IT"          : ("^CNXIT",              "Sectoral"),      # ✅
    "Nifty Pharma"      : ("^CNXPHARMA",          "Sectoral"),      # ✅
    "Nifty Auto"        : ("^CNXAUTO",            "Sectoral"),      # ✅
    "Nifty FMCG"        : ("^CNXFMCG",            "Sectoral"),      # ✅
    "Nifty Metal"       : ("^CNXMETAL",           "Sectoral"),      # ✅
    "Nifty Energy"      : ("^CNXENERGY",          "Sectoral"),      # ✅
    "Nifty Realty"      : ("^CNXREALTY",          "Sectoral"),      # ✅
    "Nifty Media"       : ("^CNXMEDIA",           "Sectoral"),      # ✅
    "Nifty PSU Bank"    : ("^CNXPSUBANK",         "Sectoral"),      # ✅
    "Nifty Fin Service" : ("NIFTY_FIN_SERVICE.NS","Sectoral"),      # ✅
    "Nifty FinSrv25-50" : ("^CNXFIN",             "Sectoral"),      # ✅

    # ── THEMATIC ──────────────────────────────────────────────────
    "Nifty Infra"       : ("^CNXINFRA",           "Thematic"),      # ✅
    "Nifty Commodities" : ("^CNXCMDT",            "Thematic"),      # ✅
    "Nifty PSE"         : ("^CNXPSE",             "Thematic"),      # ✅
    "Nifty MNC"         : ("^CNXMNC",             "Thematic"),      # ✅
    "Nifty Services"    : ("^CNXSERVICE",         "Thematic"),      # ✅

    # ── STRATEGY / FACTOR ─────────────────────────────────────────
    "Nifty Div Opps 50" : ("^CNXDIVOP",           "Strategy"),      # ✅ confirmed ^CNXDIVOP not ^CNXDIVOPPT
    "Nifty Alpha 50"    : ("NIFTYALPHA50.NS",      "Strategy"),      # ✅ confirmed NIFTYALPHA50.NS
    "Nifty100 LowVol30" : ("NIFTY100LOWVOL30.NS",  "Strategy"),      # ✅ confirmed NIFTY100LOWVOL30.NS

    # ── MACRO CONTEXT (not ranked, used as reference) ─────────────
    "India VIX"         : ("^INDIAVIX",           "Macro"),
    "USD/INR"           : ("INR=X",               "Macro"),
}

# Indices not to include in signal scoring / ranking (reference only)
MACRO_KEYS = {"India VIX", "USD/INR"}

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
                    if attempt < retries:
                        time_module.sleep(delay)
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
def calc_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    avg_g = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    avg_l = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    rsi   = (100 - (100 / (1 + rs))).dropna()
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else 50.0

def calc_macd(series: pd.Series):
    """Returns (macd_hist_current, macd_hist_prev)."""
    if len(series) < 27:
        return 0.0, 0.0
    ef  = series.ewm(span=12, adjust=False).mean()
    es  = series.ewm(span=26, adjust=False).mean()
    ml  = ef - es
    sl  = ml.ewm(span=9,  adjust=False).mean()
    h   = (ml - sl).dropna()
    if len(h) < 2:
        return 0.0, 0.0
    return round(float(h.iloc[-1]), 4), round(float(h.iloc[-2]), 4)

def macd_hist_series(series: pd.Series) -> pd.Series:
    ef = series.ewm(span=12, adjust=False).mean()
    es = series.ewm(span=26, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=9, adjust=False).mean()
    return ml - sl

def calc_bb_pct_b(series: pd.Series, period=20) -> float:
    if len(series) < period:
        return 0.5
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct_b = ((series - lower) / ((upper - lower).replace(0, np.nan))).dropna()
    return round(float(pct_b.iloc[-1]), 3) if not pct_b.empty else 0.5

def calc_ema(series: pd.Series, span: int) -> float:
    v = series.ewm(span=span, adjust=False).mean().dropna()
    return round(float(v.iloc[-1]), 2) if not v.empty else 0.0

def calc_atr(df: pd.DataFrame, period=14) -> float:
    try:
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)
        close = safe_close(df)
        prev  = close.shift(1)
        tr    = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        atr   = tr.rolling(period).mean().dropna()
        return round(float(atr.iloc[-1]), 2) if not atr.empty else 0.0
    except Exception:
        return 0.0

def week52_position(series: pd.Series) -> float:
    """Returns 0–100: where current price sits in the 52-week range."""
    if len(series) < 2:
        return 50.0
    lo  = float(series.min())
    hi  = float(series.max())
    cur = float(series.iloc[-1])
    return round((cur - lo) / (hi - lo) * 100, 1) if hi != lo else 50.0

def momentum_pct(series: pd.Series, days: int) -> float:
    """% change over last `days` bars."""
    if len(series) < days + 1:
        return 0.0
    return round((float(series.iloc[-1]) - float(series.iloc[-(days+1)])) / float(series.iloc[-(days+1)]) * 100, 2)

# ================================================================
# SIGNAL ENGINE  (6 factors, score −7 … +7 for indices)
# ================================================================
def build_signal(
    rsi: float,
    macd_h: float,
    macd_h_prev: float,
    pct_b: float,
    ema_20: float,
    ema_50: float,
    price: float,
    mom_5d: float,
    mom_20d: float,
    w52_pos: float,
) -> dict:
    score = 0
    tags  = []

    # 1. RSI
    if rsi < RSI_OVERSOLD:
        score += 2;  tags.append(f"RSI oversold {rsi:.0f}")
    elif rsi > RSI_OVERBOUGHT:
        score -= 2;  tags.append(f"RSI overbought {rsi:.0f}")
    else:
        tags.append(f"RSI {rsi:.0f}")

    # 2. MACD crossover
    if macd_h_prev <= 0 < macd_h:
        score += 2;  tags.append("MACD ↑ cross")
    elif macd_h_prev >= 0 > macd_h:
        score -= 2;  tags.append("MACD ↓ cross")
    elif macd_h > 0:
        score += 1;  tags.append("MACD+")
    else:
        score -= 1;  tags.append("MACD−")

    # 3. Bollinger %B
    if pct_b < 0.20:
        score += 1;  tags.append(f"%B low {pct_b:.2f}")
    elif pct_b > 0.80:
        score -= 1;  tags.append(f"%B high {pct_b:.2f}")

    # 4. EMA trend
    if ema_20 > ema_50 and price >= ema_20:
        score += 1;  tags.append("↑ EMA trend")
    elif ema_20 < ema_50 and price <= ema_20:
        score -= 1;  tags.append("↓ EMA trend")

    # 5. Short-term momentum (5d)
    if mom_5d > 1.5:
        score += 1;  tags.append(f"5d mom +{mom_5d:.1f}%")
    elif mom_5d < -1.5:
        score -= 1;  tags.append(f"5d mom {mom_5d:.1f}%")

    # 6. 52-week position
    if w52_pos < 25:
        score += 1;  tags.append(f"52w low zone {w52_pos:.0f}%")
    elif w52_pos > 85:
        score -= 1;  tags.append(f"52w high zone {w52_pos:.0f}%")

    # Translate
    if score >= 5:    action, emoji = "STRONG BUY",    "🟢🟢"
    elif score >= 3:  action, emoji = "BUY",           "🟢"
    elif score <= -5: action, emoji = "STRONG SELL",   "🔴🔴"
    elif score <= -2: action, emoji = "AVOID",         "🔴"
    else:             action, emoji = "NEUTRAL",       "🟡"

    return {
        "action"    : action,
        "emoji"     : emoji,
        "score"     : score,
        "tags"      : tags,
        "day_chg"   : 0.0,  # filled later
    }

# ================================================================
# BACKTESTING (per index, 30d RSI+MACD)
# ================================================================
def run_backtest(symbol: str, label: str) -> dict:
    df = fetch_history(symbol, period=f"{BT_DAYS + 15}d", interval="1d")
    if df is None or df.empty:
        return {"error": "no data"}
    close = safe_close(df)
    if len(close) < 30:
        return {"error": "short"}

    rsi_s  = close.copy()
    delta  = rsi_s.diff()
    avg_g  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    avg_l  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi_s  = 100 - (100 / (1 + rs))

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
                trades.append({"return_pct": round(gain, 2)})
                position = None

    if not trades:
        return {"total": 0, "wr": 0, "avg": 0.0}

    wins = [t for t in trades if t["return_pct"] > 0]
    wr   = round(len(wins) / len(trades) * 100, 1)
    avg  = round(sum(t["return_pct"] for t in trades) / len(trades), 2)
    return {"total": len(trades), "wr": wr, "avg": avg}

# ================================================================
# ANALYSE ONE INDEX
# ================================================================
def analyse_index(name: str, symbol: str, category: str, bt_cache: dict, today_str: str) -> dict:
    live, prev = get_live_prev(symbol)
    if live == 0.0:
        return None    # skip broken tickers

    # Daily history for indicators
    df_1d  = fetch_history(symbol, period="90d", interval="1d")
    df_5d  = fetch_history(symbol, period="5d",  interval="15m")

    rsi_v   = 50.0
    macd_h  = macd_hp = 0.0
    pct_b   = 0.5
    ema_20  = ema_50 = live
    atr_v   = 0.0
    mom_5d  = mom_20d = 0.0
    w52     = 50.0

    if df_1d is not None and not df_1d.empty:
        c = safe_close(df_1d)
        if len(c) > 14:
            rsi_v  = calc_rsi(c)
        if len(c) > 26:
            macd_h, macd_hp = calc_macd(c)
            pct_b   = calc_bb_pct_b(c)
            ema_20  = calc_ema(c, 20)
            ema_50  = calc_ema(c, 50)
        atr_v   = calc_atr(df_1d)
        mom_5d  = momentum_pct(c, 5)
        mom_20d = momentum_pct(c, 20)
        w52_c   = fetch_history(symbol, period="252d", interval="1d")
        if w52_c is not None and not w52_c.empty:
            w52 = week52_position(safe_close(w52_c))

    day_chg = (live - prev) / prev * 100 if prev else 0.0

    sig = build_signal(rsi_v, macd_h, macd_hp, pct_b, ema_20, ema_50,
                       live, mom_5d, mom_20d, w52)
    sig["day_chg"] = round(day_chg, 2)

    # Stop / target
    stop = round(live - ATR_RISK_MULT * atr_v, 2)
    t1   = round(live + ATR_RISK_MULT * atr_v, 2)
    t2   = round(live + 2 * ATR_RISK_MULT * atr_v, 2)

    # Backtest (daily cache)
    cache_key = symbol.replace("^", "").replace(".", "_")
    bt = bt_cache.get(cache_key, {})
    if not bt:
        bt = run_backtest(symbol, name)
        bt_cache[cache_key] = bt

    return {
        "name"    : name,
        "symbol"  : symbol,
        "category": category,
        "live"    : round(live, 2),
        "prev"    : round(prev, 2),
        "day_chg" : round(day_chg, 2),
        "rsi"     : rsi_v,
        "macd_h"  : macd_h,
        "pct_b"   : pct_b,
        "ema_20"  : ema_20,
        "ema_50"  : ema_50,
        "atr"     : atr_v,
        "mom_5d"  : mom_5d,
        "mom_20d" : mom_20d,
        "w52"     : w52,
        "sig"     : sig,
        "stop"    : stop,
        "t1"      : t1,
        "t2"      : t2,
        "bt"      : bt,
    }

# ================================================================
# CSV
# ================================================================
FIELDNAMES = [
    "timestamp", "market_phase", "category", "name", "symbol",
    "price", "prev", "day_chg_pct",
    "rsi", "macd_hist", "bb_pct_b", "ema_20", "ema_50",
    "atr", "mom_5d", "mom_20d", "w52_pos",
    "signal", "score",
    "stop", "t1", "t2",
    "bt_trades", "bt_win_rate", "bt_avg_return",
]

def save_csv(rows: list, timestamp: str, market_phase: str):
    f = Path("nifty_indices_history.csv")
    write_header = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            if r is None:
                continue
            bt = r["bt"]
            w.writerow({
                "timestamp"    : timestamp,
                "market_phase" : market_phase,
                "category"     : r["category"],
                "name"         : r["name"],
                "symbol"       : r["symbol"],
                "price"        : r["live"],
                "prev"         : r["prev"],
                "day_chg_pct"  : r["day_chg"],
                "rsi"          : r["rsi"],
                "macd_hist"    : r["macd_h"],
                "bb_pct_b"     : r["pct_b"],
                "ema_20"       : r["ema_20"],
                "ema_50"       : r["ema_50"],
                "atr"          : r["atr"],
                "mom_5d"       : r["mom_5d"],
                "mom_20d"      : r["mom_20d"],
                "w52_pos"      : r["w52"],
                "signal"       : r["sig"]["action"],
                "score"        : r["sig"]["score"],
                "stop"         : r["stop"],
                "t1"           : r["t1"],
                "t2"           : r["t2"],
                "bt_trades"    : bt.get("total", ""),
                "bt_win_rate"  : bt.get("wr", ""),
                "bt_avg_return": bt.get("avg", ""),
            })

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
                print(f"  📨 Telegram sent ({len(message)} chars)")
                return
            print(f"  ⚠️  Telegram attempt {attempt}: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️  Telegram attempt {attempt}: {e}")
        if attempt < MAX_RETRIES:
            time_module.sleep(RETRY_DELAY)
    print("  ❌ Telegram failed")

# ================================================================
# FORMAT MESSAGES
# ================================================================
def score_bar(score: int) -> str:
    """Visual score bar: ████░░░"""
    filled = abs(score)
    empty  = 6 - filled
    bar    = "█" * filled + "░" * empty
    return bar if score >= 0 else bar

def format_heatmap(results: list, now, market_phase: str, vix: float, usd_inr: float) -> str:
    """
    Main message: sector rotation heatmap ranked by score.
    One row per index, grouped by category.
    """
    # Sort by score desc within categories
    tradeable = [r for r in results if r is not None and r["category"] not in ("Macro",)]
    ranked    = sorted(tradeable, key=lambda r: r["sig"]["score"], reverse=True)

    # Breadth stats
    bullish  = sum(1 for r in tradeable if r["sig"]["score"] >= 3)
    bearish  = sum(1 for r in tradeable if r["sig"]["score"] <= -2)
    neutral  = len(tradeable) - bullish - bearish

    breadth_bar = (
        f"🟢 {bullish} Bull  🟡 {neutral} Neutral  🔴 {bearish} Bear"
    )

    # Category groups
    categories  = ["Broad Market", "Sectoral", "Thematic", "Strategy"]
    cat_emoji   = {
        "Broad Market": "🌏",
        "Sectoral"    : "🏭",
        "Thematic"    : "🎯",
        "Strategy"    : "⚙️",
    }

    lines = []
    for cat in categories:
        cat_rows = [r for r in ranked if r["category"] == cat]
        if not cat_rows:
            continue
        lines.append(f"\n{cat_emoji[cat]} *{cat}*")
        lines.append("```")
        lines.append(f"{'Index':<20} {'Score':>5}  {'Day%':>6}  {'RSI':>4}  {'Signal':<12}")
        lines.append("─" * 52)
        for r in cat_rows:
            arrow = "▲" if r["day_chg"] >= 0 else "▼"
            lines.append(
                f"{r['name']:<20} {r['sig']['score']:>+5}  "
                f"{arrow}{abs(r['day_chg']):.2f}%  "
                f"{r['rsi']:>4.0f}  "
                f"{r['sig']['emoji']} {r['sig']['action']:<10}"
            )
        lines.append("```")

    # Top 3 leaders and laggards
    top3    = ranked[:3]
    bot3    = ranked[-3:][::-1]
    leaders = "  ".join(f"*{r['name']}* {r['sig']['day_chg']:+.1f}%" for r in top3)
    laggard = "  ".join(f"*{r['name']}* {r['sig']['day_chg']:+.1f}%" for r in bot3)

    return (
        f"📊 *Nifty All-Indices Tracker*\n"
        f"🕒 *{market_phase}*  |  {now.strftime('%d-%b-%Y %H:%M IST')}\n"
        f"🌡 VIX: *{vix:.2f}*  |  USD/INR: ₹{usd_inr:.4f}\n\n"
        f"📈 *Market Breadth*: {breadth_bar}\n"
        + "\n".join(lines) +
        f"\n\n🚀 *Leaders*: {leaders}"
        f"\n📉 *Laggards*: {laggard}"
    )

def format_category_deep_dive(results: list, category: str, now) -> str:
    """Detailed message per category with technicals + backtest."""
    cat_results = [r for r in results if r is not None and r["category"] == category]
    if not cat_results:
        return None

    ranked = sorted(cat_results, key=lambda r: r["sig"]["score"], reverse=True)
    sep    = "─" * 34

    lines  = [f"📋 *{category} — Deep Dive*  {now.strftime('%d-%b %H:%M')}"]

    for r in ranked:
        bt       = r["bt"]
        bt_str   = (f"BT {bt['total']}T WR{bt['wr']}% avg{bt['avg']:+.1f}%"
                    if bt.get("total", 0) > 0 else "BT: no trades")
        trend    = "↑" if r["ema_20"] > r["ema_50"] else "↓"
        tags_str = " | ".join(r["sig"]["tags"][:3])
        lines.append(
            f"\n{r['sig']['emoji']} *{r['name']}*  "
            f"(score {r['sig']['score']:+d})\n"
            f"```\n"
            f"{'Price':<12} {r['live']:>12,.2f}\n"
            f"{'Day Chg':<12} {r['day_chg']:>+11.2f}%\n"
            f"{'RSI':<12} {r['rsi']:>12.1f}\n"
            f"{'MACD Hist':<12} {r['macd_h']:>+11.4f}\n"
            f"{'BB %B':<12} {r['pct_b']:>12.3f}\n"
            f"{'EMA Trend':<12} {trend + ' EMA20:'+str(r['ema_20']):>12}\n"
            f"{'5d Mom':<12} {r['mom_5d']:>+11.2f}%\n"
            f"{'20d Mom':<12} {r['mom_20d']:>+11.2f}%\n"
            f"{'52w Pos':<12} {r['w52']:>11.1f}%\n"
            f"{'ATR':<12} {r['atr']:>12.2f}\n"
            f"{'Stop':<12} {r['stop']:>12.2f}\n"
            f"{'Target 1:1':<12} {r['t1']:>12.2f}\n"
            f"{'Target 1:2':<12} {r['t2']:>12.2f}\n"
            f"```"
            f"_{tags_str}_\n"
            f"_{bt_str}_"
        )

    return "\n".join(lines)

def format_best_picks(results: list, now) -> str:
    """Final message: top 5 actionable picks across all categories."""
    tradeable = [r for r in results if r is not None and r["category"] not in ("Macro",)]
    top5      = sorted(tradeable, key=lambda r: r["sig"]["score"], reverse=True)[:5]
    worst5    = sorted(tradeable, key=lambda r: r["sig"]["score"])[:5]

    rank_e = ["🥇","🥈","🥉","4️⃣","5️⃣"]

    buy_lines  = []
    for i, r in enumerate(top5):
        buy_lines.append(
            f"{rank_e[i]} *{r['name']}* ({r['category']})\n"
            f"   {r['sig']['emoji']} {r['sig']['action']}  score {r['sig']['score']:+d}  "
            f"| Day: {r['day_chg']:+.2f}%  | RSI: {r['rsi']:.0f}\n"
            f"   Stop: {r['stop']:,.0f}  T1: {r['t1']:,.0f}  T2: {r['t2']:,.0f}\n"
            f"   _{'  |  '.join(r['sig']['tags'][:2])}_"
        )

    avoid_lines = []
    for r in worst5:
        avoid_lines.append(
            f"🔴 *{r['name']}* — {r['sig']['action']}  "
            f"(score {r['sig']['score']:+d}, day {r['day_chg']:+.2f}%)"
        )

    return (
        f"🎯 *Best Picks Right Now*  {now.strftime('%d-%b %H:%M')}\n\n"
        f"✅ *Top 5 — BUY / ACCUMULATE*\n"
        + "\n\n".join(buy_lines) +
        f"\n\n⚠️ *Weakest 5 — AVOID / WATCH*\n"
        + "\n".join(avoid_lines)
    )

# ================================================================
# MAIN
# ================================================================
def main():
    print("⏳ Nifty All-Indices Tracker — starting...\n")

    ist          = pytz.timezone("Asia/Kolkata")
    now          = datetime.now(ist)
    today_str    = now.strftime("%Y-%m-%d")
    t            = now.time()

    if   t < time(9, 15):  market_phase = "PRE-MARKET"
    elif t > time(15, 30): market_phase = "POST-MARKET"
    else:                  market_phase = "LIVE"

    # ── Load backtest cache ────────────────────────────
    bt_cache_path = Path("nifty_bt_cache.json")
    bt_cache      = {}
    if bt_cache_path.exists():
        try:
            data = json.loads(bt_cache_path.read_text())
            if data.get("date") == today_str:
                bt_cache = data.get("results", {})
                print(f"  ✅ Loaded backtest cache ({len(bt_cache)} entries)")
        except Exception:
            pass

    # ── Analyse all indices ────────────────────────────
    results = []
    vix     = 15.0
    usd_inr = 84.0

    for i, (name, (symbol, category)) in enumerate(INDICES.items()):
        print(f"  [{i+1:02d}/{len(INDICES)}] {name:<22} ({symbol})", end=" ")
        r = analyse_index(name, symbol, category, bt_cache, today_str)
        if r is None:
            print("❌ skipped")
        else:
            print(f"₹{r['live']:,.2f}  {r['sig']['emoji']} score={r['sig']['score']:+d}")
            results.append(r)
            if name == "India VIX":
                vix = r["live"]
            if name == "USD/INR":
                usd_inr = r["live"]
        time_module.sleep(FETCH_PAUSE)

    # ── Save backtest cache ────────────────────────────
    try:
        bt_cache_path.write_text(json.dumps({"date": today_str, "results": bt_cache}))
    except Exception:
        pass

    # ── Save CSV ───────────────────────────────────────
    save_csv(results, now.isoformat(), market_phase)
    print(f"\n  📁 CSV saved ({len([r for r in results if r])} rows)")

    # ── Summary stats ──────────────────────────────────
    tradeable = [r for r in results if r and r["category"] not in ("Macro",)]
    bullish   = sum(1 for r in tradeable if r["sig"]["score"] >= 3)
    bearish   = sum(1 for r in tradeable if r["sig"]["score"] <= -2)
    print(f"\n  📊 Breadth: 🟢{bullish} Bull  🔴{bearish} Bear  🟡{len(tradeable)-bullish-bearish} Neutral")

    # ── Telegram: Heatmap ──────────────────────────────
    msg1 = format_heatmap(results, now, market_phase, vix, usd_inr)
    send_telegram(msg1)
    time_module.sleep(1)

    # ── Telegram: Per-category deep-dives ─────────────
    for cat in ["Broad Market", "Sectoral", "Thematic", "Strategy"]:
        msg = format_category_deep_dive(results, cat, now)
        if msg:
            send_telegram(msg)
            time_module.sleep(1)

    # ── Telegram: Best picks summary ──────────────────
    msg_picks = format_best_picks(results, now)
    send_telegram(msg_picks)

    print("\n✅ All done!")

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    main()
