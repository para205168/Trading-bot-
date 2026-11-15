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
TRADE_LOG_FILE = "active_trades.json"

TELEGRAM_BOT_TOKEN = "9033182301:AAGwJg0OFiFHRwAA5SG7ONdylCEK68BNIXY"
MY_CHAT_ID = 9091284226

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Filter Thresholds ($1,200 Account Friendly)
MIN_AVG_VOLUME = 1_000_000  
EARNINGS_SKIP_DAYS = 7      
EARNINGS_ALERT_DAYS = 2     
ACCOUNT_SIZE = 1200.0
MAX_RISK_DOLLARS = 30.0     


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


def get_watchlist_tickers():
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", 
        "INTC", "QCOM", "AVGO", "ADBE", "CRM", "PYPL", "SHOP", "U", "PLTR", 
        "COIN", "MARA", "BA", "DIS", "JPM", "BAC", "XOM", "CVX", "WMT", "COST", 
        "JNJ", "PFE", "LLY", "UNH", "CAT", "GE", "UBER", "ABNB", "RBLX", "SQ"
    ]


def get_stock_name(ticker):
    try:
        t = yf.Ticker(ticker)
        name = t.info.get('longName') or t.info.get('shortName') or ticker
        return name
    except Exception:
        return ticker


def get_days_to_earnings(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if not cal:
            return None
        earnings_dates = cal.get('Earnings Date', []) if isinstance(cal, dict) else []
        if not earnings_dates:
            return None

        today = date.today()
        upcoming_dates = []
        for d in earnings_dates:
            if isinstance(d, datetime):
                d = d.date()
            if isinstance(d, date) and d >= today:
                upcoming_dates.append(d)

        if not upcoming_dates:
            return None
        return (min(upcoming_dates) - today).days
    except Exception:
        return None


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

            if high_price >= target:
                pnl_pct = round(((target - entry) / entry) * 100, 2)
                # 🏆 Trophy for Price Target
                alerts.append(f"🏆 *PRICE TARGET HIT!* {ticker} reached Target Price ${target} (Entry: ${entry}). Gain: +{pnl_pct}% 🎉")
            elif low_price <= stop_loss:
                pnl_pct = round(((stop_loss - entry) / entry) * 100, 2)
                # 🧱🚗 Wall in front of the Car for Stop Loss Hit
                alerts.append(f"🧱🚗 *STOP LOSS HIT!* {ticker} hit Stop Loss ${stop_loss} (Entry: ${entry}). Loss: {pnl_pct}%. Exit trade.")
            else:
                updated_trades[ticker] = trade
        except Exception:
            updated_trades[ticker] = trade

    save_active_trades(updated_trades)
    return alerts


def calculate_indicators(df):
    df['EMA_34'] = df['Close'].ewm(span=34, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    return df


def check_market_regime():
    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy['EMA_50'] = spy['Close'].ewm(span=50, adjust=False).mean()
        spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
        latest = spy.iloc[-1]
        is_bullish = (latest['Close'] > latest['EMA_50']) and (latest['EMA_50'] > latest['SMA_200'])
        return is_bullish, round(latest['Close'], 2), round(latest['EMA_50'], 2)
    except Exception:
        return True, 0, 0


def execute_full_scan():
    report_sections = []

    trade_alerts = monitor_active_trades()
    if trade_alerts:
        report_sections.append("🚨 *TRADE EXIT ALERTS (PT / SL HIT)*\n" + "\n".join(trade_alerts))

    market_bullish, spy_close, spy_ema50 = check_market_regime()
    if not market_bullish:
        report_sections.append(f"🚨 *MARKET FILTER PAUSED*\nSPY (${spy_close}) is below its 50 EMA (${spy_ema50}). Skipping new long entries.")
        return "\n\n".join(report_sections)

    tickers = get_watchlist_tickers()
    found_setups = []
    skipped_earnings = []
    active_trades = load_active_trades()

    print("🔍 Scanning watchlist leaders...")
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna().copy()
            if len(df) < 200:
                continue

            df = calculate_indicators(df)
            today = df.iloc[-1]

            if today['Vol_SMA_20'] < MIN_AVG_VOLUME:
                continue

            is_uptrend = (today['Close'] > today['EMA_50']) and (today['Close'] > today['SMA_200'])
            is_momentum_ok = (today['RSI'] > 50) and (today['MACD'] > today['MACD_Signal'])

            if is_uptrend and is_momentum_ok:
                days_to_earnings = get_days_to_earnings(ticker)
                if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_SKIP_DAYS:
                    skipped_earnings.append(f"{ticker} ({days_to_earnings}d)")
                    continue

                entry = round(float(today['Close']), 2)
                recent_swing_low = round(float(df['Low'].iloc[-10:].min()), 2)
                stop_loss = round(min(recent_swing_low, float(today['EMA_50']) * 0.98), 2)
                
                if stop_loss >= entry:
                    stop_loss = round(entry * 0.97, 2)

                risk_per_share = abs(entry - stop_loss)
                if risk_per_share == 0:
                    continue
                
                target_price = round(entry + (risk_per_share * 2.0), 2)
                company_name = get_stock_name(ticker)

                shares = round(MAX_RISK_DOLLARS / risk_per_share, 2)
                total_val = round(shares * entry, 2)
                if total_val > (ACCOUNT_SIZE * 0.35):
                    total_val = round(ACCOUNT_SIZE * 0.35, 2)
                    shares = round(total_val / entry, 2)

                active_trades[ticker] = {
                    "Name": company_name,
                    "Entry": entry,
                    "Stop_Loss": stop_loss,
                    "Target_Price": target_price,
                    "Shares": shares,
                    "Date_Added": str(date.today())
                }

                found_setups.append(
                    f"🚨 *{ticker}* - {company_name}\n"
                    f"   • Entry: `${entry}`\n"
                    f"   • Stop Loss: `${stop_loss}`\n"
                    f"   • Price Target: `${target_price}`\n"
                    f"   • Size: `{shares} shares` (`${total_val}`)"
                )
        except Exception:
            continue

    save_active_trades(active_trades)

    if found_setups:
        report_sections.append("🎯 *HIGH WIN-RATE SETUPS (SAVED)*\n\n" + "\n\n".join(found_setups[:10]))
    else:
        report_sections.append("🎯 *HIGH WIN-RATE SETUPS*\nNo qualifying setups found today.")

    if skipped_earnings:
        report_sections.append(f"🚨 *Skipped (Earnings soon)*: {', '.join(skipped_earnings[:5])}")

    final_message = "\n\n".join(report_sections)
    return final_message[:4000] + "\n...(truncated)" if len(final_message) > 4000 else final_message


def morning_scheduler():
    sent_today = None
    while True:
        now = datetime.now()
        if now.weekday() < 5 and now.hour == 8 and now.minute == 0 and sent_today != now.date():
            sent_today = now.date()
            bot.send_message(MY_CHAT_ID, "☀️ Pre-Market Auto-Scan Starting...")
            report = execute_full_scan()
            bot.send_message(MY_CHAT_ID, report, parse_mode="Markdown")
        time.sleep(30)


# ==========================================
# TELEGRAM COMMAND HANDLERS
# ==========================================

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    print(f"📩 Incoming from {message.chat.id}: {message.text}")

    if message.chat.id != MY_CHAT_ID:
        bot.reply_to(message, f"⛔ Unauthorized Chat ID: {message.chat.id}")
        return

    text = (message.text or "").strip().lower()

    if text in ['/start', 'start', '/help', 'help']:
        bot.reply_to(
            message,
            "🤖 Swing Trading Bot Online!\n\n"
            "Commands you can type:\n"
            "• `scan` - Run market scan and save active trades\n"
            "• `entry` - Show entry prices (🚪 Gate)\n"
            "• `stop loss` - Show stop loss levels (🧱🚗 Wall + Car)\n"
            "• `price target` - Show price targets (🏆 Trophy)\n"
            "• `holdings` - Check earnings warnings (🚨 Siren)",
            parse_mode="Markdown"
        )

    elif text in ['/scan', 'scan']:
        bot.reply_to(message, "🔍 Scanning market leaders... (~5 seconds)")
        try:
            report = execute_full_scan()
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
        
        # 🚪 Gate Sign for Entry
        response_lines = ["🚪 *ACTIVE TRADE ENTRIES*"]
        for ticker, data in trades.items():
            try:
                hist = yf.download(ticker, period="1d", progress=False)
                current_price = round(float(hist['Close'].iloc[-1]), 2) if not hist.empty else "N/A"
            except Exception:
                current_price = "N/A"
            
            name = data.get("Name", ticker)
            entry = data.get("Entry")
            response_lines.append(f"🚪 *{ticker}* ({name})\n  Current Price: `${current_price}` | Entry Gate: `${entry}`")

        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['stop loss', 'stoploss']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return
        
        # 🧱🚗 Wall in front of the Car for Stop Loss
        response_lines = ["🧱🚗 *ACTIVE STOP LOSS LEVELS*"]
        for ticker, data in trades.items():
            try:
                hist = yf.download(ticker, period="1d", progress=False)
                current_price = round(float(hist['Close'].iloc[-1]), 2) if not hist.empty else "N/A"
            except Exception:
                current_price = "N/A"
            
            name = data.get("Name", ticker)
            sl = data.get("Stop_Loss")
            response_lines.append(f"🧱🚗 *{ticker}* ({name})\n  Current Price: `${current_price}` | Stop Loss Barrier: `${sl}`")

        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['price target', 'pricetarget', 'target']:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(message, "📁 No active trades saved yet. Run `scan` first.")
            return
        
        # 🏆 Trophy for Price Target
        response_lines = ["🏆 *ACTIVE PRICE TARGETS*"]
        for ticker, data in trades.items():
            try:
                hist = yf.download(ticker, period="1d", progress=False)
                current_price = round(float(hist['Close'].iloc[-1]), 2) if not hist.empty else "N/A"
            except Exception:
                current_price = "N/A"
            
            name = data.get("Name", ticker)
            tp = data.get("Target_Price")
            response_lines.append(f"🏆 *{ticker}* ({name})\n  Current Price: `${current_price}` | Target Cup: `${tp}`")

        bot.send_message(MY_CHAT_ID, "\n\n".join(response_lines), parse_mode="Markdown")

    elif text in ['/holdings', 'holdings']:
        bot.reply_to(message, "⏳ Checking your holdings for earnings updates...")
        warnings = []
        for ticker in CURRENT_HOLDINGS:
            days_now = get_days_to_earnings(ticker)
            if days_now is not None and 0 <= days_now <= EARNINGS_ALERT_DAYS:
                when = "TODAY" if days_now == 0 else f"in {days_now} day(s)"
                warnings.append(f"🚨 *{ticker}* reports earnings {when}!")

        if warnings:
            bot.send_message(message.chat.id, "🚨 *EARNINGS WARNINGS*\n\n" + "\n".join(warnings), parse_mode="Markdown")
        else:
            bot.send_message(message.chat.id, "✅ All your current holdings are clear of earnings for the next few days.")


if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)
    
    threading.Thread(target=morning_scheduler, daemon=True).start()
    print("🤖 Bot online with wall + car (🧱🚗), gate (🚪), and trophy (🏆) emojis! Send a command on Telegram...")
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)