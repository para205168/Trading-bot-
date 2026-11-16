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
CURRENT_HOLDINGS = ["U", "NVDA", "AAPL"]
TRADE_LOG_FILE = "active_trades_ondemand.json"

TELEGRAM_BOT_TOKEN = "ENTER YOUR API HERE"
MY_CHAT_ID = ENTER YOUR ID HERE

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Account & Risk Parameters ($1,200 Account Friendly)
ACCOUNT_SIZE = 1200.0
MAX_RISK_DOLLARS = 30.0     
MIN_REWARD_RATIO = 2.0      # Minimum 2:1 R:R per PDF specification
MIN_AVG_VOLUME = 1_000_000


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
        df_sp500 = tables[0]
        sp500_list = df_sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
        tickers.update(sp500_list)
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")

    try:
        url_ndx = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables_ndx = pd.read_html(url_ndx)
        for table in tables_ndx:
            if 'Ticker' in table.columns:
                ndx_list = table['Ticker'].str.replace('.', '-', regex=False).tolist()
                tickers.update(ndx_list)
                break
            elif 'Symbol' in table.columns:
                ndx_list = table['Symbol'].str.replace('.', '-', regex=False).tolist()
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
        name = t.info.get('longName') or t.info.get('shortName') or ticker
        return name
    except Exception:
        return ticker


# ==========================================
# 2. PDF GRADE A+ OTE & ANALYSIS LOGIC
# ==========================================
def analyze_single_ticker(ticker):
    """
    Analyzes a specific ticker for both Long and Short Grade A+ setups 
    according to PDF institutional guidelines (FVG, MSB, OTE Golden Pocket)[cite: 15, 20].
    Returns detailed diagnostics including reasons if skipped.
    """
    try:
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
        
        # 2. Market Structure Breaks (MSB) with displacement[cite: 15, 20]
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

        today = df.iloc[-1]
        entry = round(float(today['Close']), 2)
        company_name = get_stock_name(ticker)

        # Volatility ATR for Timeframe Estimation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean().iloc[-1]

        # Check Long A+ Setup
        is_bullish_fvg = bool(today['Bullish_FVG'])
        is_bullish_msb = bool(today['Bullish_MSB'])
        in_bull_gp = bool(today['In_Bullish_GP'])

        if is_bullish_fvg and is_bullish_msb and in_bull_gp:
            recent_ssl = round(float(df['Low'].iloc[-15:].min()), 2)
            stop_loss = round(recent_ssl * 0.99, 2)
            if stop_loss >= entry:
                stop_loss = round(entry * 0.97, 2)
            risk = abs(entry - stop_loss)
            target = round(entry + (risk * MIN_REWARD_RATIO), 2)
            
            days = max(3, min(round(abs(target - entry) / (float(atr) * 0.45)), 25))
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
                "reason": "Bullish Fair Value Gap (FVG) and Market Structure Break (MSB) confirmed, with price retracing perfectly into the OTE 0.618-0.66 Golden Pocket zone supporting a 2:1 R:R ratio[cite: 15, 20]."
            }

        # Check Short A+ Setup
        is_bearish_fvg = bool(today['Bearish_FVG'])
        is_bearish_msb = bool(today['Bearish_MSB'])
        in_bear_gp = bool(today['In_Bearish_GP'])

        if is_bearish_fvg and is_bearish_msb and in_bear_gp:
            recent_bsl = round(float(df['High'].iloc[-15:].max()), 2)
            stop_loss = round(recent_bsl * 1.01, 2)
            if stop_loss <= entry:
                stop_loss = round(entry * 1.03, 2)
            risk = abs(stop_loss - entry)
            target = round(entry - (risk * MIN_REWARD_RATIO), 2)
            
            days = max(3, min(round(abs(entry - target) / (float(atr) * 0.45)), 25))
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
                "reason": "Bearish Fair Value Gap (FVG) and Market Structure Break (MSB) confirmed, with price rejecting from the premium OTE 0.618-0.66 Golden Pocket retracement zone supporting a 2:1 R:R ratio[cite: 15, 20]."
            }

        # If it failed conditions, construct specific reasons why position was rejected
        failed_conditions = []
        if not (is_bullish_fvg or is_bearish_fvg):
            failed_conditions.append("No active Fair Value Gap (FVG) imbalance formed on recent candles[cite: 5, 20].")
        if not (is_bullish_msb or is_bearish_msb):
            failed_conditions.append("No energetic Market Structure Break (MSB) displacement present[cite: 15, 20].")
        if not (in_bull_gp or in_bear_gp):
            failed_conditions.append("Price is currently outside the optimal 0.618 - 0.66 Fibonacci Golden Pocket (OTE) retracement zone[cite: 20].")

        if not failed_conditions:
            failed_conditions.append("Setup does not meet strict institutional Grade A+ confluence requirements.")

        return {
            "status": "rejected",
            "ticker": ticker,
            "name": company_name,
            "entry": entry,
            "reasons": failed_conditions
        }

    except Exception as e:
        return {"status": "error", "message": f"Error analyzing {ticker}: {str(e)}"}


def monitor_active_trades():
    active_trades = load_active_trades()
    if not active_trades:
        return []

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

            if "LONG" in side:
                if high_price >= target:
                    pnl_pct = round(((target - entry) / entry) * 100, 2)
                    alerts.append(f"🏆 *LONG TARGET HIT!* {ticker} reached Target ${target} (Entry: ${entry}). Gain: +{pnl_pct}% 🎉")
                elif low_price <= stop_loss:
                    pnl_pct = round(((stop_loss - entry) / entry) * 100, 2)
                    alerts.append(f"🧱🚗 *LONG STOP LOSS HIT!* {ticker} hit Stop Loss ${stop_loss}. Loss: {pnl_pct}%.")
                else:
                    updated_trades[ticker] = trade
            else:
                if low_price <= target:
                    pnl_pct = round(((entry - target) / entry) * 100, 2)
                    alerts.append(f"🏆 *SHORT TARGET HIT!* {ticker} reached Target ${target} (Entry: ${entry}). Gain: +{pnl_pct}% 🎉")
                elif high_price >= stop_loss:
                    pnl_pct = round(((entry - stop_loss) / entry) * 100, 2)
                    alerts.append(f"🧱🚗 *SHORT STOP LOSS HIT!* {ticker} hit Stop Loss ${stop_loss}.")
                else:
                    updated_trades[ticker] = trade
        except Exception:
            updated_trades[ticker] = trade

    save_active_trades(updated_trades)
    return alerts


def execute_aplus_scan():
    report_sections = []

    trade_alerts = monitor_active_trades()
    if trade_alerts:
        report_sections.append("🚨 *TRADE EXIT ALERTS (PT / SL HIT)*\n\n" + "\n\n".join(trade_alerts))

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

    for ticker in data.keys():
        result = analyze_single_ticker(ticker)
        if result.get("status") == "passed":
            side = result["side"]
            entry = result["entry"]
            sl = result["stop_loss"]
            tp = result["target"]
            tf = result["timeframe"]
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
                f"   • *Est. Time Frame*: `{tf}`\n"
                f"   • Entry (🚪 Gate): `${entry}`\n"
                f"   • Stop Loss (🧱🚗 Wall + Car): `${sl}`\n"
                f"   • Price Target (🏆 Trophy): `${tp}`\n"
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
            bot.send_message(MY_CHAT_ID, "☀️ Pre-Market S&P/Nasdaq Grade A+ OTE Scan Starting...")
            report = execute_aplus_scan()
            bot.send_message(MY_CHAT_ID, report, parse_mode="Markdown")
        time.sleep(30)


# ==========================================
# 3. TELEGRAM COMMAND & TICKER HANDLERS
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
            "🤖 S&P 500 & Nasdaq 100 A+ OTE Bot Online!\n\n"
            "• Type any ticker (e.g., `NVDA`, `AAPL`, `MSFT`) to analyze it instantly!\n"
            "• `scan` - Run full market scan\n"
            "• `entry` - Show entry prices (🚪 Gate)\n"
            "• `stop loss` - Show stop loss levels (🧱🚗 Wall + Car)\n"
            "• `price target` - Show price targets & timeframes (🏆 Trophy)",
            parse_mode="Markdown"
        )

    elif text in ['/scan', 'scan']:
        bot.reply_to(message, "🔍 Scanning S&P 500 and Nasdaq 100 for Grade A+ OTE setups... (~15 seconds)")
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
            response_lines.append(f"🚪 *{ticker}* ({data.get('Side', 'LONG')})\n  Entry Gate: `${data.get('Entry')}`")
        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['stop loss', 'stoploss']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return
        
        response_lines = ["🧱🚗 *ACTIVE STOP LOSS LEVELS*"]
        for ticker, data in trades.items():
            response_lines.append(f"🧱🚗 *{ticker}* ({data.get('Side', 'LONG')})\n  Stop Loss Barrier: `${data.get('Stop_Loss')}`")
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
        # User entered a stock ticker directly (e.g., "NVDA" or "AAPL")
        ticker = raw_text.upper()
        bot.reply_to(message, f"🔍 Analyzing `{ticker}` against institutional A+ criteria...")
        
        result = analyze_single_ticker(ticker)
        
        if result.get("status") == "passed":
            msg = (
                f"✅ *SETUP QUALIFIED ({result['side']})*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({result['name']})\n"
                f"   • *Reason*: {result['reason']}\n"
                f"   • *Est. Time Frame*: `{result['timeframe']}`\n"
                f"   • Entry (🚪 Gate): `${result['entry']}`\n"
                f"   • Stop Loss (🧱🚗 Wall + Car): `${result['stop_loss']}`\n"
                f"   • Price Target (🏆 Trophy): `${result['target']}`"
            )
            bot.send_message(MY_CHAT_ID, msg, parse_mode="Markdown")
        
        elif result.get("status") == "rejected":
            reasons_text = "\n".join([f"   • {r}" for r in result['reasons']])
            msg = (
                f"❌ *SETUP REJECTED (NO POSITION)*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({result['name']})\n"
                f"💰 *Current Price*: `${result['entry']}`\n\n"
                f"🛑 *Reasons for Skipping*:\n{reasons_text}"
            )
            bot.send_message(MY_CHAT_ID, msg, parse_mode="Markdown")
        else:
            bot.send_message(MY_CHAT_ID, f"⚠️ {result.get('message', 'Could not analyze ticker.')}")


if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)
    
    threading.Thread(target=morning_scheduler, daemon=True).start()
    print("🤖 On-Demand S&P/Nasdaq Grade A+ Bot online! Send any ticker (e.g. NVDA) on Telegram...")
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
