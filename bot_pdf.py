import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
import threading
import time
from datetime import datetime, date
import telebot  # Requires: pip install pyTelegramBotAPI

# Fix Windows terminal encoding for emojis/symbols
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. CONFIGURATION & TELEGRAM BOT SETUP
# ==========================================
TRADE_LOG_FILE = "active_trades_learning.json"
LEARNING_STATE_FILE = "learning_state.json"

TELEGRAM_BOT_TOKEN = "Your API"
MY_CHAT_ID = Your ID 

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Account & Risk Parameters ($10,000 Account Friendly)
ACCOUNT_SIZE = 10000.0
MAX_RISK_DOLLARS = 250.0     # Fixed 2.5% risk per trade (auto-scales shares with ATR)
MIN_REWARD_RATIO = 2.0       # Minimum 2:1 R:R

# 11 Major S&P 500 Sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication"
}


# ==========================================
# 2. SELF-LEARNING STATE & PERSISTENCE
# ==========================================
def load_learning_state():
    default_state = {
        "atr_multiplier": 0.50,          # Learns between 0.35 and 1.25
        "premature_stopouts": 0,         # Counter for liquidity sweep fake-outs
        "long_history": [],              # List of booleans [True, False, ...]
        "short_history": [],             # List of booleans [True, False, ...]
        "audit_watchlist": {},           # Stopped-out trades watched for 3 days
        "last_adjustment_note": "Initialized at baseline 0.50x ATR buffer."
    }
    if os.path.exists(LEARNING_STATE_FILE):
        try:
            with open(LEARNING_STATE_FILE, 'r') as f:
                data = json.load(f)
                for k, v in default_state.items():
                    if k not in data:
                        data[k] = v
                return data
        except Exception:
            return default_state
    return default_state


def save_learning_state(state):
    try:
        with open(LEARNING_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving learning state: {e}")


def get_directional_stats(state):
    long_hist = state.get("long_history", [])[-10:]
    short_hist = state.get("short_history", [])[-10:]

    long_wr = round((sum(long_hist) / len(long_hist)) * 100, 1) if long_hist else 65.0
    short_wr = round((sum(short_hist) / len(short_hist)) * 100, 1) if short_hist else 65.0

    long_strict = (len(long_hist) >= 4 and long_wr < 50.0)
    short_strict = (len(short_hist) >= 4 and short_wr < 50.0)

    return long_wr, short_wr, long_strict, short_strict


def load_active_trades():
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_active_trades(trades):
    try:
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trades, f, indent=4)
    except Exception as e:
        print(f"Error saving active trades: {e}")


def get_sp500_and_nasdaq_tickers():
    """Fetches combined S&P 500 and Nasdaq 100 tickers dynamically."""
    tickers = set()
    try:
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url_sp500)
        sp500_list = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        tickers.update(sp500_list)
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")

    try:
        url_ndx = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables_ndx = pd.read_html(url_ndx)
        for table in tables_ndx:
            col = 'Ticker' if 'Ticker' in table.columns else ('Symbol' if 'Symbol' in table.columns else None)
            if col:
                ndx_list = table[col].str.replace('.', '-', regex=False).tolist()
                tickers.update(ndx_list)
                break
    except Exception as e:
        print(f"Error fetching Nasdaq 100 list: {e}")

    if not tickers:
        return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "QCOM", "AVGO", "U", "PLTR"]

    return list(tickers)


def get_stock_name(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('longName') or t.info.get('shortName') or ticker
    except Exception:
        return ticker


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ==========================================
# 3. ANTI-FAKEOUT 11-SECTOR ROTATION ENGINE (5D / 10D / 20D)
# ==========================================
def analyze_sector_rotation():
    """Analyzes 5D, 10D, and 20D returns + SPY relative strength + volume to filter out false 5-day bounces."""
    try:
        symbols = list(SECTOR_ETFS.keys()) + ["SPY"]
        raw_data = yf.download(symbols, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True)

        if raw_data is None or raw_data.empty:
            return "⚠️ Could not retrieve sector ETF data right now. Please try again."

        spy_df = raw_data["SPY"].dropna()
        spy_5d = ((float(spy_df['Close'].iloc[-1]) - float(spy_df['Close'].iloc[-6])) / float(spy_df['Close'].iloc[-6])) * 100
        spy_10d = ((float(spy_df['Close'].iloc[-1]) - float(spy_df['Close'].iloc[-11])) / float(spy_df['Close'].iloc[-11])) * 100
        spy_20d = ((float(spy_df['Close'].iloc[-1]) - float(spy_df['Close'].iloc[-21])) / float(spy_df['Close'].iloc[-21])) * 100

        true_inflows = []
        early_turns = []
        fakeout_bounces = []
        confirmed_outflows = []
        neutral_list = []

        all_metrics = []

        for etf, name in SECTOR_ETFS.items():
            if etf not in raw_data:
                continue
            df = raw_data[etf].dropna()
            if len(df) < 22:
                continue

            close_now = float(df['Close'].iloc[-1])
            ret_5d = ((close_now - float(df['Close'].iloc[-6])) / float(df['Close'].iloc[-6])) * 100
            ret_10d = ((close_now - float(df['Close'].iloc[-11])) / float(df['Close'].iloc[-11])) * 100
            ret_20d = ((close_now - float(df['Close'].iloc[-21])) / float(df['Close'].iloc[-21])) * 100

            rs_5d = ret_5d - spy_5d
            rs_10d = ret_10d - spy_10d

            vol_5d = float(df['Volume'].iloc[-5:].mean())
            vol_20d = float(df['Volume'].iloc[-20:].mean())
            vol_ratio = round(vol_5d / vol_20d, 2) if vol_20d > 0 else 1.0

            # Composite score weighing 5D (35%), 10D (40%), and 20D (25%)
            score = (ret_5d * 0.35) + (ret_10d * 0.40) + (ret_20d * 0.25) + (rs_10d * 0.15)

            item = {
                "etf": etf,
                "name": name,
                "ret_5d": round(ret_5d, 2),
                "ret_10d": round(ret_10d, 2),
                "ret_20d": round(ret_20d, 2),
                "rs_5d": round(rs_5d, 2),
                "vol_ratio": vol_ratio,
                "score": round(score, 2)
            }
            all_metrics.append(item)

            # Classify accurately to prevent 5-day fake-out traps
            if ret_5d > 0.25 and ret_10d > 0.25 and ret_20d > 0:
                true_inflows.append(item)
            elif ret_5d > 0.35 and ret_10d > 0.25 and ret_20d <= 0:
                early_turns.append(item)
            elif ret_5d > 0.50 and ret_10d < -0.25 and ret_20d < -0.50:
                fakeout_bounces.append(item)
            elif ret_5d < -0.25 and ret_10d < -0.25 and ret_20d < 0:
                confirmed_outflows.append(item)
            else:
                neutral_list.append(item)

        all_metrics.sort(key=lambda x: x["score"], reverse=True)
        true_inflows.sort(key=lambda x: x["score"], reverse=True)
        early_turns.sort(key=lambda x: x["score"], reverse=True)
        confirmed_outflows.sort(key=lambda x: x["score"])

        lines = [
            "📊 *ANTI-FAKEOUT SECTOR ROTATION (5D / 10D / 20D)*",
            f"_(Benchmark `SPY`: 5D `{spy_5d:+.2f}%` | 10D `{spy_10d:+.2f}%` | 20D `{spy_20d:+.2f}%`)_\n"
        ]

        # 1. True Confirmed Inflows
        lines.append("🟢 *CONFIRMED INSTITUTIONAL INFLOWS (5D + 10D + 20D Aligned):*")
        if true_inflows:
            for s in true_inflows:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%` | Vol `{s['vol_ratio']}x`"
                )
        else:
            lines.append("   • _No sectors have all 3 timeframes aligned bullish right now._")

        # 2. Early Emerging Turnarounds
        if early_turns:
            lines.append("\n🟡 *EARLY EMERGING ROTATION (5D & 10D Turning Up):*")
            for s in early_turns:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        # 3. False 5-Day Bounces (Traps)
        if fakeout_bounces:
            lines.append("\n⚠️ *POTENTIAL 5-DAY FAKE-OUT BOUNCE (10D & 20D Still Bearish — Do Not Chase):*")
            for s in fakeout_bounces:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        # 4. Neutral
        if neutral_list:
            lines.append("\n⚪ *NEUTRAL / MIXED:*")
            for s in neutral_list:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        # 5. Confirmed Outflows
        lines.append("\n🔴 *CONFIRMED INSTITUTIONAL OUTFLOWS (5D + 10D + 20D Bearish):*")
        if confirmed_outflows:
            for s in confirmed_outflows:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%` | Vol `{s['vol_ratio']}x`"
                )
        else:
            lines.append("   • _No sectors have all 3 timeframes aligned bearish right now._")

        # Actionable Summary
        top_in = all_metrics[:2]
        top_out = all_metrics[-2:]
        in_str = " & ".join([f"{x['name']} (`{x['etf']}`)" for x in top_in])
        out_str = " & ".join([f"{x['name']} (`{x['etf']}`)" for x in reversed(top_out)])

        lines.append(
            f"\n🔄 *NET MONEY FLOW VERDICT*:\n"
            f"Money is rotating *OUT OF* {out_str} ➡️ *INTO* {in_str}."
        )

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error analyzing sector rotation: {str(e)}"


# ==========================================
# 4. GRADE A+ OTE + CANDLE-CLOSE ENTRY RULES
# ==========================================
def analyze_single_ticker(ticker, preloaded_df=None):
    state = load_learning_state()
    atr_mult = state.get("atr_multiplier", 0.50)
    long_wr, short_wr, long_strict, short_strict = get_directional_stats(state)

    try:
        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if df is None or df.empty:
            return {"status": "error", "message": f"Could not retrieve historical data for {ticker}."}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna().copy()
        if len(df) < 50:
            return {"status": "error", "message": f"Insufficient price history for {ticker}."}

        swing_high = df['High'].rolling(window=30).max()
        swing_low = df['Low'].rolling(window=30).min()

        # 1. Fair Value Gaps (FVG)
        df['Bullish_FVG'] = (df['Low'] > df['High'].shift(2))
        df['Bearish_FVG'] = (df['High'] < df['Low'].shift(2))

        # 2. Market Structure Breaks (MSB) with displacement
        body_size = abs(df['Close'] - df['Open'])
        avg_body = body_size.rolling(window=14).mean()
        displacement = body_size > (avg_body * 1.2)

        df['Bullish_MSB'] = (df['Close'] > df['High'].shift(1)) & displacement
        df['Bearish_MSB'] = (df['Close'] < df['Low'].shift(1)) & displacement

        # 3. Fibonacci Golden Pocket (OTE) Zones [0.618 to 0.66 retracement]
        range_span = swing_high - swing_low
        fib_618 = swing_high - (range_span * 0.618)
        fib_660 = swing_high - (range_span * 0.660)

        bear_fib_618 = swing_low + (range_span * 0.618)
        bear_fib_660 = swing_low + (range_span * 0.660)

        df['In_Bullish_GP'] = (df['Low'] <= fib_618) & (df['Close'] >= fib_660)
        df['In_Bearish_GP'] = (df['High'] >= bear_fib_618) & (df['Close'] <= bear_fib_660)

        # 4. Individual Stock ATR & RSI
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)

        today = df.iloc[-1]
        entry = round(float(today['Close']), 2)
        ticker_atr = float(today['ATR']) if not pd.isna(today['ATR']) else (entry * 0.02)
        current_rsi = round(float(today['RSI']), 1) if not pd.isna(today['RSI']) else 50.0
        company_name = get_stock_name(ticker)
        atr_buffer = round(ticker_atr * atr_mult, 2)

        # Near-term 5-day, 10-day & 15-day structural levels
        recent_10_high = round(float(df['High'].iloc[-10:].max()), 2)
        recent_10_low = round(float(df['Low'].iloc[-10:].min()), 2)
        recent_5_high = round(float(df['High'].iloc[-5:].max()), 2)
        recent_5_low = round(float(df['Low'].iloc[-5:].min()), 2)
        recent_15_high = round(float(df['High'].iloc[-15:].max()), 2)
        recent_15_low = round(float(df['Low'].iloc[-15:].min()), 2)

        # --- CHECK LONG A+ SETUP ---
        if bool(today['Bullish_FVG']) and bool(today['Bullish_MSB']) and bool(today['In_Bullish_GP']):
            if not (long_strict and current_rsi > 42.0):
                stop_loss = round(recent_15_low - atr_buffer, 2)
                if stop_loss >= entry:
                    stop_loss = round(entry - (ticker_atr * (1.0 + atr_mult)), 2)

                risk = abs(entry - stop_loss)
                target = round(entry + (risk * MIN_REWARD_RATIO), 2)

                days = max(3, min(round(abs(target - entry) / (ticker_atr * 0.45)), 25))
                tf_str = f"~{days} to {days + 5} trading days (~{max(1, round(days/5))} to {round((days+5)/5)} weeks)"

                return {
                    "status": "passed",
                    "side": "LONG 📈",
                    "ticker": ticker,
                    "name": company_name,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "target": target,
                    "timeframe": tf_str,
                    "entry_action": f"Enter LONG after 1D (Daily) candle close above `${entry}` (or on a 4H bullish candle close holding `${entry}`)",
                    "reason": (
                        f"Bullish Fair Value Gap (FVG) and Market Structure Break (MSB) confirmed on 1D close inside the OTE "
                        f"0.618-0.66 Golden Pocket, protected by a {atr_mult}x ATR (${atr_buffer}) liquidity-sweep buffer."
                    )
                }

        # --- CHECK SHORT A+ SETUP ---
        if bool(today['Bearish_FVG']) and bool(today['Bearish_MSB']) and bool(today['In_Bearish_GP']):
            if not (short_strict and current_rsi < 58.0):
                stop_loss = round(recent_15_high + atr_buffer, 2)
                if stop_loss <= entry:
                    stop_loss = round(entry + (ticker_atr * (1.0 + atr_mult)), 2)

                risk = abs(stop_loss - entry)
                target = round(entry - (risk * MIN_REWARD_RATIO), 2)

                days = max(3, min(round(abs(entry - target) / (ticker_atr * 0.45)), 25))
                tf_str = f"~{days} to {days + 5} trading days (~{max(1, round(days/5))} to {round((days+5)/5)} weeks)"

                return {
                    "status": "passed",
                    "side": "SHORT 📉",
                    "ticker": ticker,
                    "name": company_name,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "target": target,
                    "timeframe": tf_str,
                    "entry_action": f"Enter SHORT after 1D (Daily) candle close below `${entry}` (or on a 4H bearish candle close rejecting `${entry}`)",
                    "reason": (
                        f"Bearish Fair Value Gap (FVG) and Market Structure Break (MSB) confirmed on 1D close inside the premium OTE "
                        f"0.618-0.66 Golden Pocket, protected by a {atr_mult}x ATR (${atr_buffer}) liquidity-sweep buffer."
                    )
                }

        # --- DIAGNOSTIC REJECTION REASONS ---
        failed_conditions = []
        if not (bool(today['Bullish_FVG']) or bool(today['Bearish_FVG'])):
            failed_conditions.append("No active Fair Value Gap (FVG) imbalance formed on recent 1D candles.")
        if not (bool(today['Bullish_MSB']) or bool(today['Bearish_MSB'])):
            failed_conditions.append("No energetic Market Structure Break (MSB) displacement present on 1D close.")
        if not (bool(today['In_Bullish_GP']) or bool(today['In_Bearish_GP'])):
            failed_conditions.append("Price is currently outside the optimal 0.618 - 0.66 Fibonacci Golden Pocket (OTE) retracement zone.")
        if long_strict and current_rsi > 42.0:
            failed_conditions.append(f"Self-Learning Long Filter active (Win Rate {long_wr}% < 50%): Waiting for RSI <= 42 (Current: {current_rsi}).")
        if short_strict and current_rsi < 58.0:
            failed_conditions.append(f"Self-Learning Short Filter active (Win Rate {short_wr}% < 50%): Waiting for RSI >= 58 (Current: {current_rsi}).")

        # --- REALISTIC NEAR-TERM BULLISH & BEARISH GAME PLAN (Explicit 4H / 1D Close Rules) ---
        local_span = max(recent_10_high - recent_10_low, ticker_atr * 1.5)

        # 1. Bullish Scenario
        local_bull_gp = round(recent_10_high - (local_span * 0.638), 2)
        bull_entry = round(max(local_bull_gp, entry - (ticker_atr * 1.1)), 2)
        bull_breakout_trigger = round(min(recent_5_high, entry + (ticker_atr * 1.15)), 2)
        if bull_breakout_trigger <= bull_entry:
            bull_breakout_trigger = round(bull_entry + (ticker_atr * 0.75), 2)

        bull_sl = round(bull_entry - (ticker_atr * (1.0 + atr_mult)), 2)
        bull_tp = round(bull_entry + (abs(bull_entry - bull_sl) * MIN_REWARD_RATIO), 2)

        if bull_entry < entry:
            bull_label = "Pullback OTE"
            bull_entry_action = (
                f"Enter LONG after a *4H candle close* bouncing above `${bull_entry}` ({bull_label}) "
                f"*OR* after a *1D (Daily) candle close* above `${bull_breakout_trigger}`"
            )
        else:
            bull_label = "Reclaim OTE"
            bull_entry_action = (
                f"Enter LONG after a *4H candle close* holding above `${bull_entry}` ({bull_label}) "
                f"*OR* after a *1D (Daily) candle close* above `${bull_breakout_trigger}`"
            )

        bull_reason = (
            f"Confirms institutional buyers are holding the OTE zone on a 4H close or breaking resistance on a full 1D close "
            f"with a Bullish MSB + FVG (ignore 5-min wicks)."
        )

        # 2. Bearish Scenario
        local_bear_gp = round(recent_10_low + (local_span * 0.638), 2)
        bear_bounce_entry = round(min(local_bear_gp, entry + (ticker_atr * 1.1)), 2)
        bear_breakdown_trigger = round(max(recent_5_low, entry - (ticker_atr * 1.15)), 2)
        if bear_breakdown_trigger >= bear_bounce_entry:
            bear_breakdown_trigger = round(bear_bounce_entry - (ticker_atr * 0.75), 2)

        bear_entry = bear_bounce_entry
        bear_sl = round(bear_entry + (ticker_atr * (1.0 + atr_mult)), 2)
        bear_tp = round(bear_entry - (abs(bear_sl - bear_entry) * MIN_REWARD_RATIO), 2)

        if bear_entry > entry:
            bear_label = "Supply Retest"
            bear_entry_action = (
                f"Enter SHORT after a *4H candle close* rejecting below `${bear_entry}` ({bear_label}) "
                f"*OR* after a *1D (Daily) candle close* below `${bear_breakdown_trigger}`"
            )
        else:
            bear_label = "Breakdown Retest"
            bear_entry_action = (
                f"Enter SHORT after a *4H candle close* staying below `${bear_entry}` ({bear_label}) "
                f"*OR* after a *1D (Daily) candle close* below `${bear_breakdown_trigger}`"
            )

        bear_reason = (
            f"Confirms sellers are rejecting premium OTE supply on a 4H close or snapping support on a full 1D close "
            f"with a Bearish MSB + FVG (ignore 5-min wicks)."
        )

        return {
            "status": "rejected",
            "ticker": ticker,
            "name": company_name,
            "entry": entry,
            "atr_mult": atr_mult,
            "reasons": failed_conditions if failed_conditions else ["Setup does not meet strict institutional Grade A+ confluence requirements."],
            "bull_plan": {
                "entry_action": bull_entry_action,
                "sl": bull_sl,
                "tp": bull_tp,
                "reason": bull_reason
            },
            "bear_plan": {
                "entry_action": bear_entry_action,
                "sl": bear_sl,
                "tp": bear_tp,
                "reason": bear_reason
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error analyzing {ticker}: {str(e)}"}


# ==========================================
# 5. TRADE MONITORING & POST-STOP-OUT AUDIT
# ==========================================
def monitor_and_learn():
    active_trades = load_active_trades()
    state = load_learning_state()
    alerts = []
    updated_trades = {}

    for ticker, trade in active_trades.items():
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                updated_trades[ticker] = trade
                continue

            high_price = float(hist['High'].iloc[-1])
            low_price = float(hist['Low'].iloc[-1])

            entry = trade['Entry']
            target = trade['Target_Price']
            stop_loss = trade['Stop_Loss']
            side = trade.get('Side', 'LONG 📈')
            is_long = "LONG" in side

            if is_long:
                if high_price >= target:
                    pnl_pct = round(((target - entry) / entry) * 100, 2)
                    state["long_history"].append(True)
                    alerts.append(f"🏆 *LONG TARGET HIT!* {ticker} reached `${target}` (Entry: `${entry}`). Gain: `+{pnl_pct}%` 🎉")
                elif low_price <= stop_loss:
                    pnl_pct = round(((stop_loss - entry) / entry) * 100, 2)
                    state["long_history"].append(False)
                    state["audit_watchlist"][ticker] = {
                        "Side": "LONG",
                        "Target_Price": target,
                        "Stopped_Date": str(date.today()),
                        "Days_Watched": 0
                    }
                    alerts.append(f"🧱🚗 *LONG STOP LOSS HIT!* {ticker} hit `${stop_loss}` (`{pnl_pct}%`). Bot is auditing for liquidity sweep fake-outs.")
                else:
                    updated_trades[ticker] = trade
            else:
                if low_price <= target:
                    pnl_pct = round(((entry - target) / entry) * 100, 2)
                    state["short_history"].append(True)
                    alerts.append(f"🏆 *SHORT TARGET HIT!* {ticker} reached `${target}` (Entry: `${entry}`). Gain: `+{pnl_pct}%` 🎉")
                elif high_price >= stop_loss:
                    pnl_pct = round(((entry - stop_loss) / entry) * 100, 2)
                    state["short_history"].append(False)
                    state["audit_watchlist"][ticker] = {
                        "Side": "SHORT",
                        "Target_Price": target,
                        "Stopped_Date": str(date.today()),
                        "Days_Watched": 0
                    }
                    alerts.append(f"🧱🚗 *SHORT STOP LOSS HIT!* {ticker} hit `${stop_loss}`. Bot is auditing for liquidity sweep fake-outs.")
                else:
                    updated_trades[ticker] = trade
        except Exception:
            updated_trades[ticker] = trade

    remaining_watchlist = {}
    for ticker, audit in state.get("audit_watchlist", {}).items():
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                remaining_watchlist[ticker] = audit
                continue

            high_price = float(hist['High'].iloc[-1])
            low_price = float(hist['Low'].iloc[-1])
            orig_target = audit["Target_Price"]
            days_watched = audit.get("Days_Watched", 0) + 1

            fakeout_detected = False
            if audit["Side"] == "LONG" and high_price >= orig_target:
                fakeout_detected = True
            elif audit["Side"] == "SHORT" and low_price <= orig_target:
                fakeout_detected = True

            if fakeout_detected:
                state["premature_stopouts"] = state.get("premature_stopouts", 0) + 1
                old_mult = state["atr_multiplier"]
                new_mult = min(1.25, round(old_mult + 0.15, 2))
                state["atr_multiplier"] = new_mult
                note = f"Auto-Learned from {ticker} fake-out sweep: Increased ATR buffer from {old_mult}x to {new_mult}x."
                state["last_adjustment_note"] = note
                alerts.append(f"🧠 *SELF-LEARNING UPDATE:* {ticker} reversed and hit original target `${orig_target}` after stopping out. {note}")
            elif days_watched < 3:
                audit["Days_Watched"] = days_watched
                remaining_watchlist[ticker] = audit
        except Exception:
            remaining_watchlist[ticker] = audit

    all_recent = (state.get("long_history", [])[-5:] + state.get("short_history", [])[-5:])
    if len(all_recent) >= 8:
        combined_wr = (sum(all_recent) / len(all_recent)) * 100
        if combined_wr >= 72.0 and state["atr_multiplier"] > 0.50:
            old_mult = state["atr_multiplier"]
            new_mult = max(0.50, round(old_mult - 0.10, 2))
            state["atr_multiplier"] = new_mult
            state["last_adjustment_note"] = f"High win rate ({round(combined_wr, 1)}%): Tightened ATR buffer from {old_mult}x to {new_mult}x."

    state["audit_watchlist"] = remaining_watchlist
    save_learning_state(state)
    save_active_trades(updated_trades)
    return alerts


def execute_aplus_scan():
    report_sections = []

    learning_alerts = monitor_and_learn()
    if learning_alerts:
        report_sections.append("🚨 *TRADE EXIT & LEARNING ALERTS*\n\n" + "\n\n".join(learning_alerts))

    tickers = get_sp500_and_nasdaq_tickers()
    found_setups = []
    active_trades = load_active_trades()

    print(f"🔍 Scanning {len(tickers)} tickers across S&P 500 and Nasdaq 100...")

    batch_size = 50
    data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            batch_data = yf.download(batch, period="6mo", interval="1d", group_by="ticker", progress=False, threads=True)
            if not batch_data.empty:
                if isinstance(batch_data.columns, pd.MultiIndex):
                    for t in batch:
                        if t in batch_data:
                            data[t] = batch_data[t]
                else:
                    if len(batch) == 1:
                        data[batch[0]] = batch_data
        except Exception:
            pass
        time.sleep(0.3)

    for ticker, df_ticker in data.items():
        result = analyze_single_ticker(ticker, preloaded_df=df_ticker)
        if result.get("status") == "passed":
            side = result["side"]
            entry = result["entry"]
            sl = result["stop_loss"]
            tp = result["target"]
            tf = result["timeframe"]
            entry_act = result["entry_action"]
            reason = result["reason"]
            c_name = result["name"]

            risk = abs(entry - sl)
            shares = round(MAX_RISK_DOLLARS / risk, 2) if risk > 0 else 1
            total_val = round(shares * entry, 2)
            if total_val > (ACCOUNT_SIZE * 0.35):
                total_val = round(ACCOUNT_SIZE * 0.35, 2)
                shares = round(total_val / entry, 2)

            active_trades[ticker] = {
                "Name": c_name,
                "Side": side,
                "Entry": entry,
                "Stop_Loss": sl,
                "Target_Price": tp,
                "Timeframe": tf,
                "Shares": shares,
                "Date_Added": str(date.today())
            }

            found_setups.append(
                f"🚨 *{ticker}* - {c_name} ({side} Grade A+ Setup)\n"
                f"   • *Reason*: {reason}\n"
                f"   • 🚪 *Entry Action*: {entry_act}\n"
                f"   • *Est. Time Frame*: `{tf}`\n"
                f"   • 🧱🚗 Stop Loss: `${sl}`\n"
                f"   • 🏆 Price Target: `${tp}`\n"
                f"   • Position Size: `{shares} shares` (`${total_val}`)"
            )

    save_active_trades(active_trades)

    if found_setups:
        report_sections.append("🎯 *GRADE A+ S&P/NASDAQ OTE SETUPS (SAVED)*\n\n" + "\n\n".join(found_setups[:10]))
    else:
        report_sections.append("🎯 *S&P/NASDAQ GRADE A+ SCAN*\nNo stocks met strict A+ criteria across S&P 500 / Nasdaq 100 today.")

    final_message = "\n\n".join(report_sections)
    return final_message[:4000] + "\n...(truncated)" if len(final_message) > 4000 else final_message


def morning_scheduler():
    sent_today = None
    while True:
        now = datetime.now()
        if now.weekday() < 5 and now.hour == 8 and now.minute == 0 and sent_today != now.date():
            sent_today = now.date()
            bot.send_message(MY_CHAT_ID, "☀️ Pre-Market Self-Learning Grade A+ Scan Starting...")
            report = execute_aplus_scan()
            bot.send_message(MY_CHAT_ID, report, parse_mode="Markdown")
        time.sleep(30)


# ==========================================
# 6. TELEGRAM COMMAND & TICKER HANDLERS
# ==========================================
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    print(f"📩 Incoming from {message.chat.id}: {message.text}")

    if message.chat.id != MY_CHAT_ID:
        bot.reply_to(message, f"⛔ Unauthorized Chat ID: {message.chat.id}")
        return

    text = (message.text or "").strip().lower()
    raw_text = (message.text or "").strip()

    if text in ['/start', 'start', '/help', 'help']:
        bot.reply_to(
            message,
            "🤖 *Self-Learning S&P 500 & Nasdaq 100 A+ Bot Online!*\n\n"
            "• Type any ticker (e.g., `MSFT`, `CRWV`) to get instant A+ analysis & 4H/1D candle-close trigger rules!\n"
            "• `sector` - See anti-fakeout 11-Sector rotation (5D / 10D / 20D institutional money flow)\n"
            "• `scan` - Run full market scan & self-learning audit\n"
            "• `stats` - View live Long/Short win rates & ATR learning status\n"
            "• `entry` - Show entries (🚪)\n"
            "• `stop loss` - Show stop losses (🧱🚗)\n"
            "• `price target` - Show targets & timeframes (🏆)",
            parse_mode="Markdown"
        )

    elif text in ['/sector', 'sector', 'sectors']:
        bot.reply_to(message, "📊 Analyzing 5D / 10D / 20D institutional money flow across all 11 S&P 500 sectors...")
        sector_report = analyze_sector_rotation()
        bot.send_message(MY_CHAT_ID, sector_report, parse_mode="Markdown")

    elif text in ['/stats', 'stats']:
        state = load_learning_state()
        long_wr, short_wr, long_strict, short_strict = get_directional_stats(state)
        atr_mult = state.get("atr_multiplier", 0.50)
        fakeouts = state.get("premature_stopouts", 0)
        note = state.get("last_adjustment_note", "No adjustments yet.")

        msg = (
            f"🧠 *SELF-LEARNING ENGINE STATUS*\n\n"
            f"• *Current ATR Stop-Loss Buffer*: `{atr_mult}x ATR`\n"
            f"• *Liquidity Sweep Fake-Outs Caught*: `{fakeouts}`\n"
            f"• *Long Win Rate (Rolling)*: `{long_wr}%` ({'Strict RSI Filter Active' if long_strict else 'Normal Mode'})\n"
            f"• *Short Win Rate (Rolling)*: `{short_wr}%` ({'Strict RSI Filter Active' if short_strict else 'Normal Mode'})\n\n"
            f"📝 *Latest Adaptation*: _{note}_"
        )
        bot.send_message(MY_CHAT_ID, msg, parse_mode="Markdown")

    elif text in ['/scan', 'scan']:
        bot.reply_to(message, "🔍 Scanning S&P 500 and Nasdaq 100 with Self-Learning ATR & Directional Filters... (~15 seconds)")
        try:
            report = execute_aplus_scan()
            bot.send_message(MY_CHAT_ID, report, parse_mode="Markdown")
            print("✅ Scan report sent to Telegram!")
        except Exception as e:
            print(f"❌ Scan Error: {e}")
            bot.send_message(MY_CHAT_ID, f"❌ Error during scan: {e}")

    elif text in ['entry']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return

        response_lines = ["🚪 *ACTIVE TRADE ENTRIES*"]
        for ticker, data in trades.items():
            response_lines.append(f"🚪 *{ticker}* ({data.get('Side', 'LONG')})\n  Entry: `${data.get('Entry')}`")
        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['stop loss', 'stoploss']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return

        response_lines = ["🧱🚗 *ACTIVE STOP LOSS LEVELS*"]
        for ticker, data in trades.items():
            response_lines.append(f"🧱🚗 *{ticker}* ({data.get('Side', 'LONG')})\n  Stop Loss: `${data.get('Stop_Loss')}`")
        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['price target', 'pricetarget', 'target']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return

        response_lines = ["🏆 *ACTIVE PRICE TARGETS & TIMEFRAMES*"]
        for ticker, data in trades.items():
            response_lines.append(f"🏆 *{ticker}* ({data.get('Side', 'LONG')})\n  Target: `${data.get('Target_Price')}`\n  ⏱️ *Est. Time*: `{data.get('Timeframe')}`")
        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    else:
        ticker = raw_text.upper()
        bot.reply_to(message, f"🔍 Analyzing `{ticker}` with Self-Learning A+ Engine...")

        result = analyze_single_ticker(ticker)

        if result.get("status") == "passed":
            msg = (
                f"✅ *SETUP QUALIFIED ({result['side']})*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({result['name']})\n"
                f"   • *Reason*: {result['reason']}\n"
                f"   • 🚪 *Entry Action*: {result['entry_action']}\n"
                f"   • *Est. Time Frame*: `{result['timeframe']}`\n"
                f"   • 🧱🚗 Stop Loss: `${result['stop_loss']}`\n"
                f"   • 🏆 Price Target: `${result['target']}`"
            )
            bot.send_message(MY_CHAT_ID, msg, parse_mode="Markdown")

        elif result.get("status") == "rejected":
            reasons_text = "\n".join([f"   • {r}" for r in result['reasons']])
            bull = result['bull_plan']
            bear = result['bear_plan']
            atr_m = result['atr_mult']

            msg = (
                f"❌ *SETUP REJECTED (NO POSITION YET)*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({result['name']})\n"
                f"💰 *Current Price*: `${result['entry']}`\n\n"
                f"🛑 *Reasons for Skipping Today*:\n{reasons_text}\n\n"
                f"🔮 *GAME PLAN: WHEN TO ENTER*\n\n"
                f"📈 *Bullish Case (Will Go LONG when):*\n"
                f"   • 🚪 *Entry Action*: {bull['entry_action']}\n"
                f"   • *Why*: {bull['reason']}\n"
                f"   • 🧱🚗 *Planned Stop Loss*: `${bull['sl']}` (`{atr_m}x ATR` buffer)\n"
                f"   • 🏆 *Planned Target*: `${bull['tp']}`\n\n"
                f"📉 *Bearish Case (Will SHORT when):*\n"
                f"   • 🚪 *Entry Action*: {bear['entry_action']}\n"
                f"   • *Why*: {bear['reason']}\n"
                f"   • 🧱🚗 *Planned Stop Loss*: `${bear['sl']}` (`{atr_m}x ATR` buffer)\n"
                f"   • 🏆 *Planned Target*: `${bear['tp']}`"
            )
            bot.send_message(MY_CHAT_ID, msg, parse_mode="Markdown")
        else:
            bot.send_message(MY_CHAT_ID, f"⚠️ {result.get('message', 'Could not analyze ticker.')}")


if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)

    threading.Thread(target=morning_scheduler, daemon=True).start()
    print("🤖 Self-Learning S&P/Nasdaq Grade A+ Bot online! Send 'sector', 'scan', 'stats', or any ticker on Telegram...")
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
