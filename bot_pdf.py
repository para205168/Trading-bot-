from datetime import date, datetime, timedelta
import json
import os
import sys
import threading
import time
from zoneinfo import ZoneInfo
from fpdf import FPDF  # Requires: pip install fpdf2
import numpy as np
import pandas as pd
import requests
import telebot  # Requires: pip install pyTelegramBotAPI
import yfinance as yf

# Fix Windows terminal encoding for emojis/symbols
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ==========================================
# 1. CONFIGURATION & TELEGRAM BOT SETUP
# ==========================================
TRADE_LOG_FILE = "active_trades_learning.json"
LEARNING_STATE_FILE = "learning_state.json"
JOURNAL_LOG_FILE = "trade_journal.json"

CT_TZ = ZoneInfo("America/Chicago")

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "Your API here"
)
MY_CHAT_ID = int(os.getenv("MY_CHAT_ID", "YOur ID here"))

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Account & Confidence-Based Capital Allocation ($10,000 Cash Account - LONG ONLY)
STARTING_CAPITAL = 10000.0
HIGH_CONF_RESERVE = (
    2500.0  # Protected cash reserved exclusively for HIGH confidence setups
)
MIN_REWARD_RATIO = 2.0  # Minimum 2:1 R:R

# Sweet-Spot Stop-Loss Tuning (Anti-Fakeout Floor + Tight-Target Ceiling)
BASE_MIN_STOP_ATR = (
    1.15  # Minimum distance from entry so daily noise doesn't stop you out
)
BASE_MAX_STOP_ATR = (
    1.45  # Maximum distance from entry so 2:1 targets stay realistic
)

CONFIDENCE_TIERS = {
    "HIGH": {
        "label": "🔥 HIGH CONFIDENCE",
        "short_tag": "HIGH",
        "risk_pct": 0.025,  # 2.5% risk ($250 on $10k)
        "max_alloc_pct": 0.35,  # Max 35% position ($3,500)
        "can_use_reserve": True,
    },
    "MEDIUM": {
        "label": "⚡ MEDIUM CONFIDENCE",
        "short_tag": "MEDIUM",
        "risk_pct": 0.015,  # 1.5% risk ($150 on $10k)
        "max_alloc_pct": 0.20,  # Max 20% position ($2,000)
        "can_use_reserve": False,
    },
    "LOW": {
        "label": "🔹 LOW CONFIDENCE",
        "short_tag": "LOW",
        "risk_pct": 0.0075,  # 0.75% risk ($75 on $10k)
        "max_alloc_pct": 0.10,  # Max 10% position ($1,000)
        "can_use_reserve": False,
    },
}

# Boroden Fibonacci Tuning Parameters
BORODEN_ZONE_TOLERANCE = 0.0075  # 0.75% zone width for Symmetry/Cluster hit
BORODEN_SWING_ORDER = 4  # Bars left/right to confirm swing high/low

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
    "XLC": "Communication",
}


# ==========================================
# 2. SAFE MARKDOWN, TIMEZONE & STOP HELPERS
# ==========================================
def now_ct() -> datetime:
    """Returns current timezone-aware datetime in US Central Time (CST/CDT)."""
    return datetime.now(CT_TZ)


def sanitize_md(text: str) -> str:
    """Strips characters that break Telegram legacy Markdown formatting."""
    if not text:
        return ""
    return (
        str(text)
        .replace("*", "")
        .replace("_", " ")
        .replace("`", "'")
        .replace("[", "(")
        .replace("]", ")")
    )


def sanitize_pdf_text(text: str) -> str:
    """Strips non-Latin-1 characters (like emojis) so standard PDF fonts render cleanly."""
    if not text:
        return ""
    return str(text).encode("latin-1", "ignore").decode("latin-1").strip()


def send_safe_markdown(chat_id, text: str):
    """Sends a Markdown message safely, falling back to plain text if Telegram rejects an entity."""
    if not text:
        return
    try:
        bot.send_message(chat_id, text, parse_mode="Markdown")
    except Exception:
        clean_text = text.replace("*", "").replace("_", "").replace("`", "")
        bot.send_message(chat_id, clean_text)


def calculate_sweet_spot_stop(
    entry: float,
    pivot_low: float,
    ticker_atr: float,
    atr_mult: float,
) -> tuple[float, float]:
    """Calculates the Sweet-Spot Reversal-Pivot Stop Loss.

    Anchors below the 2-bar reversal pivot low with a wick buffer, then clamps
    the stop distance strictly between [min_atr_dist, max_atr_dist] so stops are
    neither too tight (fake-out prone) nor too wide (unrealistic 2:1 targets).
    Returns (stop_loss, effective_atr_distance_mult).
    """
    learning_shift = max(0.0, atr_mult - 0.50)
    min_stop_mult = round(BASE_MIN_STOP_ATR + learning_shift, 2)
    max_stop_mult = round(BASE_MAX_STOP_ATR + learning_shift, 2)

    wick_buffer = ticker_atr * (atr_mult * 0.5)
    raw_pivot_stop = pivot_low - wick_buffer

    min_allowed_stop = entry - (ticker_atr * max_stop_mult)
    max_allowed_stop = entry - (ticker_atr * min_stop_mult)

    clamped_stop = max(min_allowed_stop, min(raw_pivot_stop, max_allowed_stop))
    stop_loss = round(max(0.01, clamped_stop), 2)
    eff_mult = (
        round((entry - stop_loss) / ticker_atr, 2) if ticker_atr > 0 else 1.25
    )
    return stop_loss, eff_mult


# ==========================================
# 3. SELF-LEARNING STATE, CAPITAL & JOURNAL PERSISTENCE
# ==========================================
def load_learning_state():
    default_state = {
        "atr_multiplier": 0.50,
        "premature_stopouts": 0,
        "long_history": [],
        "audit_watchlist": {},
        "last_adjustment_note": "Sweet-Spot Pivot Stop active (1.15x - 1.45x ATR band).",
    }
    if os.path.exists(LEARNING_STATE_FILE):
        try:
            with open(LEARNING_STATE_FILE, "r") as f:
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
        with open(LEARNING_STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving learning state: {e}")


def get_long_stats(state):
    long_hist = state.get("long_history", [])[-10:]
    long_wr = (
        round((sum(long_hist) / len(long_hist)) * 100, 1) if long_hist else 66.0
    )
    long_strict = len(long_hist) >= 4 and long_wr < 50.0
    return long_wr, long_strict


def load_active_trades():
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_active_trades(trades):
    try:
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(trades, f, indent=4)
    except Exception as e:
        print(f"Error saving active trades: {e}")


def load_journal_entries():
    if os.path.exists(JOURNAL_LOG_FILE):
        try:
            with open(JOURNAL_LOG_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_journal_entries(entries):
    try:
        with open(JOURNAL_LOG_FILE, "w") as f:
            json.dump(entries, f, indent=4)
    except Exception as e:
        print(f"Error saving journal entries: {e}")


def get_capital_status(active_trades=None):
    """Calculates live account equity, invested capital, and available cash for new trades."""
    if active_trades is None:
        active_trades = load_active_trades()

    entries = load_journal_entries()
    realized_pnl = sum(
        float(e.get("PnL_Dollars", 0.0))
        for e in entries
        if e.get("Status") in ("WIN", "LOSS")
    )
    total_equity = round(STARTING_CAPITAL + realized_pnl, 2)

    invested = 0.0
    for tr in active_trades.values():
        if "SHORT" in tr.get("Side", "LONG"):
            continue
        pos_val = tr.get("Position_Value")
        if pos_val is None:
            pos_val = float(tr.get("Shares", 0.0)) * float(tr.get("Entry", 0.0))
        invested += float(pos_val)

    invested = round(invested, 2)
    available_cash = round(max(0.0, total_equity - invested), 2)
    general_cash = round(max(0.0, available_cash - HIGH_CONF_RESERVE), 2)

    return {
        "total_equity": total_equity,
        "realized_pnl": round(realized_pnl, 2),
        "invested": invested,
        "available_cash": available_cash,
        "general_cash": general_cash,
        "reserve": HIGH_CONF_RESERVE,
    }


def allocate_position_by_confidence(
    entry: float, stop_loss: float, tier_key: str, cap_status: dict
):
    """Sizes a LONG position based on confidence tier (HIGH/MEDIUM/LOW) and remaining cash."""
    tier = CONFIDENCE_TIERS[tier_key]
    equity = cap_status["total_equity"]
    usable_cash = (
        cap_status["available_cash"]
        if tier["can_use_reserve"]
        else cap_status["general_cash"]
    )

    if usable_cash < entry or usable_cash < 100.0:
        return 0.0, 0.0, 0.0, usable_cash

    risk_per_share = abs(entry - stop_loss)
    target_risk_dollars = equity * tier["risk_pct"]
    max_tier_dollars = equity * tier["max_alloc_pct"]

    risk_based_dollars = (
        (target_risk_dollars / risk_per_share) * entry
        if risk_per_share > 0
        else max_tier_dollars
    )
    alloc_dollars = min(risk_based_dollars, max_tier_dollars, usable_cash)

    shares = round(alloc_dollars / entry, 2)
    actual_dollars = round(shares * entry, 2)
    actual_risk = round(shares * risk_per_share, 2)

    if shares <= 0 or actual_dollars < 50.0:
        return 0.0, 0.0, 0.0, usable_cash

    return shares, actual_dollars, actual_risk, usable_cash


def log_journal_entry(trade_dict):
    """Appends a new LONG trade setup to the persistent journal if not already logged today."""
    entries = load_journal_entries()
    today_str = str(now_ct().date())
    for item in entries:
        if (
            item.get("Ticker") == trade_dict["Ticker"]
            and item.get("Date_Opened") == today_str
            and item.get("Status") == "OPEN"
        ):
            return
    entries.append(trade_dict)
    save_journal_entries(entries)


def update_journal_exit(ticker, status, exit_price, pnl_pct, pnl_dollars):
    """Marks the matching OPEN journal entry as WIN or LOSS when closed."""
    entries = load_journal_entries()
    today_str = str(now_ct().date())
    updated = False
    for item in reversed(entries):
        if item.get("Ticker") == ticker and item.get("Status") == "OPEN":
            item["Status"] = status
            item["Exit_Price"] = round(exit_price, 2)
            item["Date_Closed"] = today_str
            item["PnL_Pct"] = round(pnl_pct, 2)
            item["PnL_Dollars"] = round(pnl_dollars, 2)
            updated = True
            break
    if updated:
        save_journal_entries(entries)


def mark_journal_fakeout(ticker):
    """Flags a stopped-out trade in the journal if it later reversed to hit its original target."""
    entries = load_journal_entries()
    for item in reversed(entries):
        if item.get("Ticker") == ticker and item.get("Status") == "LOSS":
            item["Notes"] = (
                "Liquidity sweep fake-out: reversed & hit target within 3 days."
            )
            save_journal_entries(entries)
            break


# ==========================================
# 4. WEEKLY PDF JOURNAL GENERATOR
# ==========================================
class WeeklyJournalPDF(FPDF):

    def header(self):
        self.set_fill_color(24, 42, 69)
        self.rect(0, 0, 210, 26, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.set_xy(10, 6)
        self.cell(
            0, 7, "WEEKLY SWING TRADING JOURNAL (LONG-ONLY)", ln=True, align="L"
        )
        self.set_font("Helvetica", "", 9)
        self.set_x(10)
        self.cell(
            0,
            5,
            "Confidence-Weighted Capital Allocation ($10k Account) | Sweet-Spot Pivot Stops",
            ln=True,
            align="L",
        )
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            8,
            f"Generated by SandeepPick_bot | Page {self.page_no()}",
            align="C",
        )


def generate_weekly_journal_pdf() -> tuple[str, str]:
    """Builds a weekly PDF report covering Monday through Friday of the current week."""
    current_dt = now_ct()
    today_date = current_dt.date()
    monday_date = today_date - timedelta(days=today_date.weekday())
    friday_date = monday_date + timedelta(days=4)

    entries = load_journal_entries()
    active_trades = load_active_trades()
    state = load_learning_state()
    cap = get_capital_status(active_trades)

    if not entries and active_trades:
        for tkr, tr in active_trades.items():
            if "SHORT" in tr.get("Side", "LONG"):
                continue
            sh = float(tr.get("Shares", 0.0))
            ent = float(tr.get("Entry", 0.0))
            entries.append({
                "Ticker": tkr,
                "Name": tr.get("Name", tkr),
                "Confidence": tr.get("Confidence", "MEDIUM"),
                "Confidence_Score": tr.get("Confidence_Score", 70),
                "Engine": tr.get("Engine", "Dual-Engine"),
                "Date_Opened": tr.get("Date_Added", str(today_date)),
                "Entry": ent,
                "Stop_Loss": tr.get("Stop_Loss", 0.0),
                "Target_Price": tr.get("Target_Price", 0.0),
                "Shares": sh,
                "Position_Value": tr.get(
                    "Position_Value",
                    round(sh * ent, 2),
                ),
                "Status": "OPEN",
                "Exit_Price": None,
                "Date_Closed": None,
                "PnL_Pct": 0.0,
                "PnL_Dollars": 0.0,
                "Notes": "",
            })
        save_journal_entries(entries)

    weekly_trades = []
    for item in entries:
        d_open = item.get("Date_Opened", "")
        d_close = item.get("Date_Closed") or ""
        is_this_week = (
            str(monday_date) <= d_open <= str(friday_date)
            or (d_close and str(monday_date) <= d_close <= str(friday_date))
            or item.get("Status") == "OPEN"
        )
        if is_this_week:
            weekly_trades.append(item)

    open_tickers = list({
        t["Ticker"] for t in weekly_trades if t.get("Status") == "OPEN"
    })
    live_prices = {}
    if open_tickers:
        try:
            px_data = yf.download(
                open_tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if not px_data.empty:
                if len(open_tickers) == 1:
                    tkr = open_tickers[0]
                    live_prices[tkr] = round(
                        float(px_data["Close"].dropna().iloc[-1]), 2
                    )
                else:
                    for tkr in open_tickers:
                        if tkr in px_data and not px_data[tkr].dropna().empty:
                            live_prices[tkr] = round(
                                float(px_data[tkr]["Close"].dropna().iloc[-1]),
                                2,
                            )
        except Exception:
            pass

    total_setups = len(weekly_trades)
    wins = [t for t in weekly_trades if t.get("Status") == "WIN"]
    losses = [t for t in weekly_trades if t.get("Status") == "LOSS"]
    opens = [t for t in weekly_trades if t.get("Status") == "OPEN"]

    closed_count = len(wins) + len(losses)
    weekly_wr = (
        round((len(wins) / closed_count) * 100, 1) if closed_count > 0 else 0.0
    )
    realized_pnl = round(
        sum(float(t.get("PnL_Dollars", 0.0)) for t in wins + losses), 2
    )

    unrealized_pnl = 0.0
    for t in opens:
        tkr = t["Ticker"]
        curr_px = live_prices.get(tkr, t.get("Entry", 0.0))
        shares = float(t.get("Shares", 0.0))
        entry = float(t.get("Entry", 0.0))
        unrealized_pnl += (curr_px - entry) * shares
    unrealized_pnl = round(unrealized_pnl, 2)

    long_wr, long_strict = get_long_stats(state)
    atr_mult = state.get("atr_multiplier", 0.50)
    shift = max(0.0, atr_mult - 0.50)
    min_band = round(BASE_MIN_STOP_ATR + shift, 2)
    max_band = round(BASE_MAX_STOP_ATR + shift, 2)
    fakeouts = state.get("premature_stopouts", 0)
    note = sanitize_pdf_text(
        state.get("last_adjustment_note", "Baseline active.")
    )

    pdf = WeeklyJournalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        7,
        f"Reporting Week: {monday_date.strftime('%b %d, %Y')} - {friday_date.strftime('%b %d, %Y')} (Generated {current_dt.strftime('%I:%M %p CT')})",
        ln=True,
    )
    pdf.ln(2)

    # KPI Summary Boxes (2 rows x 3 columns)
    box_w = 61
    box_h = 18
    gap = 3.5
    start_x = 10
    y1 = pdf.get_y()

    kpis_row1 = [
        (
            "ACCOUNT EQUITY / CASH",
            f"${cap['total_equity']:,.0f} (${cap['available_cash']:,.0f} free)",
            (240, 245, 252),
        ),
        (
            "CLOSED WIN RATE",
            f"{weekly_wr}% ({len(wins)}W / {len(losses)}L)",
            (236, 250, 242),
        ),
        (
            "TOTAL INVESTED",
            f"{len(opens)} Open (${cap['invested']:,.2f})",
            (252, 248, 235),
        ),
    ]
    for idx, (label, val, bg) in enumerate(kpis_row1):
        x = start_x + idx * (box_w + gap)
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(200, 210, 220)
        pdf.rect(x, y1, box_w, box_h, "DF")
        pdf.set_xy(x + 3, y1 + 2.5)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(95, 105, 115)
        pdf.cell(box_w - 6, 4, label, ln=True)
        pdf.set_x(x + 3)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(20, 30, 45)
        pdf.cell(box_w - 6, 7, val, ln=True)

    y2 = y1 + box_h + 3
    kpis_row2 = [
        (
            "REALIZED WEEKLY P&L",
            f"${realized_pnl:+,.2f}",
            (236, 250, 242) if realized_pnl >= 0 else (253, 238, 238),
        ),
        (
            "UNREALIZED OPEN P&L",
            f"${unrealized_pnl:+,.2f}",
            (240, 245, 252),
        ),
        (
            "SWEET-SPOT STOP BAND",
            f"{min_band}x-{max_band}x ATR ({fakeouts} sweeps)",
            (245, 242, 252),
        ),
    ]
    for idx, (label, val, bg) in enumerate(kpis_row2):
        x = start_x + idx * (box_w + gap)
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(200, 210, 220)
        pdf.rect(x, y2, box_w, box_h, "DF")
        pdf.set_xy(x + 3, y2 + 2.5)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(95, 105, 115)
        pdf.cell(box_w - 6, 4, label, ln=True)
        pdf.set_x(x + 3)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(20, 30, 45)
        pdf.cell(box_w - 6, 7, val, ln=True)

    pdf.set_y(y2 + box_h + 6)

    # Self-Learning Engine Note Box
    pdf.set_fill_color(246, 248, 250)
    pdf.set_draw_color(210, 218, 226)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 50, 80)
    pdf.cell(
        0,
        6,
        f"  Self-Learning Engine Status (Rolling Win Rate: {long_wr}% | High-Conf Reserve: ${HIGH_CONF_RESERVE:,.0f})",
        border="LTR",
        ln=True,
        fill=True,
    )
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(60, 65, 75)
    pdf.multi_cell(
        0,
        5,
        f"  Latest Adaptation: {note}",
        border="LBR",
        fill=True,
    )
    pdf.ln(5)

    # Trade Table Header (Includes Shares Bought & Total Invested $ per stock - Total width = 190mm)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(24, 42, 69)
    pdf.cell(
        0, 7, "WEEKLY LONG TRADE LOG, SHARES & INVESTED CAPITAL", ln=True
    )

    col_widths = [15, 16, 33, 17, 17, 17, 15, 20, 14, 26]
    headers = [
        "Ticker",
        "Conf.",
        "Engine",
        "Entry",
        "Stop",
        "Target",
        "Shares",
        "Invested",
        "Status",
        "Cur/Exit P&L",
    ]

    pdf.set_fill_color(24, 42, 69)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7.4)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    if not weekly_trades:
        pdf.set_text_color(80, 80, 80)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(
            sum(col_widths),
            10,
            "No LONG setups were triggered or closed during this week.",
            border=1,
            align="C",
            ln=True,
        )
    else:
        pdf.set_font("Helvetica", "", 7.3)
        for row_idx, t in enumerate(weekly_trades):
            if row_idx % 2 == 0:
                pdf.set_fill_color(255, 255, 255)
            else:
                pdf.set_fill_color(247, 249, 252)

            tkr = sanitize_pdf_text(t.get("Ticker", ""))
            conf = sanitize_pdf_text(t.get("Confidence", "MED"))
            eng = sanitize_pdf_text(t.get("Engine", ""))
            if len(eng) > 19:
                eng = eng[:16] + "..."
            entry = float(t.get("Entry", 0.0))
            sl = float(t.get("Stop_Loss", 0.0))
            tp = float(t.get("Target_Price", 0.0))
            shares = float(t.get("Shares", 0.0))
            invested_amt = float(
                t.get("Position_Value", round(entry * shares, 2))
            )
            status = sanitize_pdf_text(t.get("Status", "OPEN"))

            if status == "OPEN":
                cur_px = live_prices.get(tkr, entry)
                pnl_d = (cur_px - entry) * shares
                pnl_str = f"${cur_px:.2f} ({pnl_d:+.0f})"
            else:
                ex_px = float(t.get("Exit_Price") or entry)
                pnl_d = float(t.get("PnL_Dollars", 0.0))
                pnl_str = f"${ex_px:.2f} ({pnl_d:+.0f})"

            pdf.set_text_color(25, 25, 25)
            pdf.set_font("Helvetica", "B", 7.6)
            pdf.cell(col_widths[0], 7, tkr, border=1, align="C", fill=True)

            if conf == "HIGH":
                pdf.set_text_color(18, 128, 60)
            elif conf == "MEDIUM":
                pdf.set_text_color(30, 85, 165)
            else:
                pdf.set_text_color(110, 110, 110)
            pdf.cell(col_widths[1], 7, conf, border=1, align="C", fill=True)

            pdf.set_text_color(25, 25, 25)
            pdf.set_font("Helvetica", "", 7.2)
            pdf.cell(
                col_widths[2], 7, " " + eng, border=1, align="L", fill=True
            )
            pdf.cell(
                col_widths[3],
                7,
                f"${entry:.2f}",
                border=1,
                align="R",
                fill=True,
            )
            pdf.cell(
                col_widths[4], 7, f"${sl:.2f}", border=1, align="R", fill=True
            )
            pdf.cell(
                col_widths[5], 7, f"${tp:.2f}", border=1, align="R", fill=True
            )
            pdf.cell(
                col_widths[6],
                7,
                f"{shares:.2f}",
                border=1,
                align="R",
                fill=True,
            )
            pdf.set_font("Helvetica", "B", 7.3)
            pdf.cell(
                col_widths[7],
                7,
                f"${invested_amt:,.2f}",
                border=1,
                align="R",
                fill=True,
            )

            if status == "WIN":
                pdf.set_text_color(18, 128, 60)
                pdf.set_font("Helvetica", "B", 7.3)
            elif status == "LOSS":
                pdf.set_text_color(185, 35, 35)
                pdf.set_font("Helvetica", "B", 7.3)
            else:
                pdf.set_text_color(165, 110, 15)
                pdf.set_font("Helvetica", "B", 7.3)
            pdf.cell(col_widths[8], 7, status, border=1, align="C", fill=True)

            pdf.set_text_color(25, 25, 25)
            pdf.set_font("Helvetica", "", 7.1)
            pdf.cell(col_widths[9], 7, pnl_str, border=1, align="R", fill=True)
            pdf.ln()

    notes_list = [
        f"- {t['Ticker']} ({t['Date_Opened']}): {sanitize_pdf_text(t['Notes'])}"
        for t in weekly_trades
        if t.get("Notes")
    ]
    if notes_list:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(24, 42, 69)
        pdf.cell(0, 6, "AUDIT & EXECUTION NOTES", ln=True)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(50, 50, 50)
        for n in notes_list:
            pdf.multi_cell(0, 5, n)

    filename = f"Weekly_Trading_Journal_{today_date}.pdf"
    pdf.output(filename)

    caption = (
        f"📓 *Weekly Trading Journal ({monday_date.strftime('%b %d')} – {friday_date.strftime('%b %d, %Y')})*\n"
        f"• *Account Equity*: `${cap['total_equity']:,.2f}` (`${cap['invested']:,.2f}` Invested | `${cap['available_cash']:,.2f}` Free Cash)\n"
        f"• *Setups Tracked*: `{total_setups}` (`{len(wins)}W` / `{len(losses)}L` / `{len(opens)} Open`)\n"
        f"• *Closed Win Rate*: `{weekly_wr}%` | *Realized P&L*: `${realized_pnl:+,.2f}`"
    )
    return filename, caption


# ==========================================
# 5. TICKER LIST & TECHNICAL INDICATORS
# ==========================================
def get_sp500_and_nasdaq_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = set()
    try:
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html_sp = requests.get(url_sp500, headers=headers, timeout=15).text
        tables = pd.read_html(html_sp)
        sp500_list = (
            tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        )
        tickers.update(sp500_list)
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")

    try:
        url_ndx = "https://en.wikipedia.org/wiki/Nasdaq-100"
        html_ndx = requests.get(url_ndx, headers=headers, timeout=15).text
        tables_ndx = pd.read_html(html_ndx)
        for table in tables_ndx:
            col = (
                "Ticker"
                if "Ticker" in table.columns
                else ("Symbol" if "Symbol" in table.columns else None)
            )
            if col:
                ndx_list = (
                    table[col].str.replace(".", "-", regex=False).tolist()
                )
                tickers.update(ndx_list)
                break
    except Exception as e:
        print(f"Error fetching Nasdaq 100 list: {e}")

    if not tickers:
        return [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "AMD",
            "NFLX",
            "QCOM",
            "AVGO",
            "U",
            "PLTR",
        ]

    return sorted(list(tickers))


def get_stock_name(ticker):
    try:
        t = yf.Ticker(ticker)
        raw_name = t.info.get("longName") or t.info.get("shortName") or ticker
        return sanitize_md(raw_name)
    except Exception:
        return ticker


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_cci(df, period):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def find_swings(df, order=BORODEN_SWING_ORDER):
    highs = df["High"].values
    lows = df["Low"].values
    swing_highs = []
    swing_lows = []

    for i in range(order, len(df) - order):
        if highs[i] == np.max(highs[i - order : i + order + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == np.min(lows[i - order : i + order + 1]):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


# ==========================================
# 6. ANTI-FAKEOUT 11-SECTOR ROTATION ENGINE (5D / 10D / 20D)
# ==========================================
def analyze_sector_rotation():
    try:
        symbols = list(SECTOR_ETFS.keys()) + ["SPY"]
        raw_data = yf.download(
            symbols,
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if raw_data is None or raw_data.empty:
            return "⚠️ Could not retrieve sector ETF data right now. Please try again."

        spy_df = raw_data["SPY"].dropna()
        spy_5d = (
            (float(spy_df["Close"].iloc[-1]) - float(spy_df["Close"].iloc[-6]))
            / float(spy_df["Close"].iloc[-6])
        ) * 100
        spy_10d = (
            (
                float(spy_df["Close"].iloc[-1])
                - float(spy_df["Close"].iloc[-11])
            )
            / float(spy_df["Close"].iloc[-11])
        ) * 100
        spy_20d = (
            (
                float(spy_df["Close"].iloc[-1])
                - float(spy_df["Close"].iloc[-21])
            )
            / float(spy_df["Close"].iloc[-21])
        ) * 100

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

            close_now = float(df["Close"].iloc[-1])
            ret_5d = (
                (close_now - float(df["Close"].iloc[-6]))
                / float(df["Close"].iloc[-6])
            ) * 100
            ret_10d = (
                (close_now - float(df["Close"].iloc[-11]))
                / float(df["Close"].iloc[-11])
            ) * 100
            ret_20d = (
                (close_now - float(df["Close"].iloc[-21]))
                / float(df["Close"].iloc[-21])
            ) * 100

            rs_5d = ret_5d - spy_5d
            rs_10d = ret_10d - spy_10d

            vol_5d = float(df["Volume"].iloc[-5:].mean())
            vol_20d = float(df["Volume"].iloc[-20:].mean())
            vol_ratio = round(vol_5d / vol_20d, 2) if vol_20d > 0 else 1.0

            score = (
                (ret_5d * 0.35)
                + (ret_10d * 0.40)
                + (ret_20d * 0.25)
                + (rs_10d * 0.15)
            )

            item = {
                "etf": etf,
                "name": name,
                "ret_5d": round(ret_5d, 2),
                "ret_10d": round(ret_10d, 2),
                "ret_20d": round(ret_20d, 2),
                "rs_5d": round(rs_5d, 2),
                "vol_ratio": vol_ratio,
                "score": round(score, 2),
            }
            all_metrics.append(item)

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
            f"_(Benchmark `SPY`: 5D `{spy_5d:+.2f}%` | 10D `{spy_10d:+.2f}%` | 20D `{spy_20d:+.2f}%`)_\n",
        ]

        lines.append(
            "🟢 *CONFIRMED INSTITUTIONAL INFLOWS (5D + 10D + 20D Aligned):*"
        )
        if true_inflows:
            for s in true_inflows:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%` | Vol `{s['vol_ratio']}x`"
                )
        else:
            lines.append(
                "   • _No sectors have all 3 timeframes aligned bullish right now._"
            )

        if early_turns:
            lines.append(
                "\n🟡 *EARLY EMERGING ROTATION (5D & 10D Turning Up):*"
            )
            for s in early_turns:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        if fakeout_bounces:
            lines.append(
                "\n⚠️ *POTENTIAL 5-DAY FAKE-OUT BOUNCE (10D & 20D Still Bearish — Do Not Chase):*"
            )
            for s in fakeout_bounces:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        if neutral_list:
            lines.append("\n⚪ *NEUTRAL / MIXED:*")
            for s in neutral_list:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%`"
                )

        lines.append(
            "\n🔴 *CONFIRMED INSTITUTIONAL OUTFLOWS (5D + 10D + 20D Bearish):*"
        )
        if confirmed_outflows:
            for s in confirmed_outflows:
                lines.append(
                    f"   • *{s['name']} (`{s['etf']}`)*: 5D `{s['ret_5d']:+.2f}%` | 10D `{s['ret_10d']:+.2f}%` | 20D `{s['ret_20d']:+.2f}%` | Vol `{s['vol_ratio']}x`"
                )
        else:
            lines.append(
                "   • _No sectors have all 3 timeframes aligned bearish right now._"
            )

        top_in = all_metrics[:2]
        top_out = all_metrics[-2:]
        in_str = " & ".join([f"{x['name']} (`{x['etf']}`)" for x in top_in])
        out_str = " & ".join(
            [f"{x['name']} (`{x['etf']}`)" for x in reversed(top_out)]
        )

        lines.append(
            f"\n🔄 *NET MONEY FLOW VERDICT*:\n"
            f"Money is rotating *OUT OF* {out_str} ➡️ *INTO* {in_str}."
        )

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error analyzing sector rotation: {str(e)}"


# ==========================================
# 7. BORODEN SYMMETRY & CLUSTER ENGINE (LONG ONLY)
# ==========================================
def evaluate_boroden_fib_long(
    df, current_price, ticker_atr, atr_mult, pivot_low
):
    if len(df) < 120:
        return None, None

    df = df.copy()
    df["EMA_34"] = df["Close"].ewm(span=34, adjust=False).mean()
    df["CCI_14"] = calculate_cci(df, 14)
    df["CCI_50"] = calculate_cci(df, 50)

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(curr["CCI_14"]) or pd.isna(curr["CCI_50"]):
        return None, None

    recent_df = df.iloc[-90:].reset_index(drop=True)
    swing_highs, swing_lows = find_swings(recent_df, order=BORODEN_SWING_ORDER)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None, None

    last_high_idx, last_high_price = swing_highs[-1]
    prior_pullbacks = []
    for h_idx, h_val in swing_highs[:-1]:
        following_lows = [l_val for l_idx, l_val in swing_lows if l_idx > h_idx]
        if following_lows:
            drop = h_val - min(following_lows)
            if drop > 0:
                prior_pullbacks.append(drop)

    long_sym_levels = [
        last_high_price - drop for drop in prior_pullbacks[-4:]
    ]
    nearest_long_sym = (
        round(min(long_sym_levels, key=lambda x: abs(x - current_price)), 2)
        if long_sym_levels
        else None
    )

    if curr["Close"] > curr["EMA_34"] and curr["CCI_50"] > 0 and long_sym_levels:
        if (
            curr["Low"] < last_high_price * 0.985
            and curr["Close"] < last_high_price
        ):
            retracement_levels = []
            for l_idx, l_val in swing_lows[-3:]:
                if l_idx < last_high_idx and l_val < last_high_price:
                    swing_range = last_high_price - l_val
                    for ratio in (0.382, 0.50, 0.618, 0.786):
                        retracement_levels.append(
                            last_high_price - (swing_range * ratio)
                        )

            all_fib_levels = long_sym_levels + retracement_levels
            pullback_low = float(recent_df.iloc[last_high_idx:]["Low"].min())

            matched_sym = [
                lvl
                for lvl in long_sym_levels
                if abs(pullback_low - lvl) / lvl <= BORODEN_ZONE_TOLERANCE
            ]
            cluster_levels = [
                lvl
                for lvl in all_fib_levels
                if abs(pullback_low - lvl) / lvl <= BORODEN_ZONE_TOLERANCE
            ]

            is_sym = len(matched_sym) >= 1
            is_cluster = len(cluster_levels) >= 3 and is_sym

            cci_trigger = (
                prev["CCI_14"] < 0 and curr["CCI_14"] > prev["CCI_14"]
            ) or (prev["CCI_14"] <= 0 and curr["CCI_14"] > 0)
            bullish_candle = (
                curr["Close"] > curr["Open"] or curr["Close"] > prev["Close"]
            )

            if (is_sym or is_cluster) and cci_trigger and bullish_candle:
                zone_levels = cluster_levels if is_cluster else matched_sym
                zone_low = round(min(zone_levels), 2)
                zone_high = round(max(zone_levels), 2)

                stop_loss, eff_stop_mult = calculate_sweet_spot_stop(
                    current_price, pivot_low, ticker_atr, atr_mult
                )

                pullback_range = last_high_price - pullback_low
                target_1272 = round(pullback_low + (pullback_range * 1.272), 2)
                target_1618 = round(pullback_low + (pullback_range * 1.618), 2)
                min_rr_target = round(
                    current_price
                    + (abs(current_price - stop_loss) * MIN_REWARD_RATIO),
                    2,
                )

                setup_label = (
                    f"Boroden Fib Cluster ({len(cluster_levels)} levels + 1.00 Symmetry)"
                    if is_cluster
                    else "Boroden 100% Symmetry Pullback"
                )
                return {
                    "setup_label": setup_label,
                    "is_cluster": is_cluster,
                    "cluster_count": len(cluster_levels),
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "stop_loss": stop_loss,
                    "eff_stop_mult": eff_stop_mult,
                    "target": max(target_1272, min_rr_target),
                    "target_1272": target_1272,
                    "target_1618": target_1618,
                    "ema_34": round(float(curr["EMA_34"]), 2),
                    "cci_14": round(float(curr["CCI_14"]), 1),
                    "cci_50": round(float(curr["CCI_50"]), 1),
                }, nearest_long_sym

    return None, nearest_long_sym


# ==========================================
# 8. CONFIDENCE SCORING + DUAL ENGINE (LONG ONLY)
# ==========================================
def score_setup_confidence(
    ict_long_ok: bool,
    boroden_res: dict | None,
    current_rsi: float,
    vol_ratio: float,
):
    """Rates a qualified LONG setup from 0 to 100 and assigns HIGH, MEDIUM, or LOW tier."""
    score = 45

    if ict_long_ok and boroden_res is not None:
        score += 40
    elif boroden_res is not None:
        if boroden_res.get("is_cluster"):
            c_count = boroden_res.get("cluster_count", 3)
            score += 22 + min((c_count - 3) * 4, 12)
        else:
            score += 8
    elif ict_long_ok:
        score += 20

    if boroden_res is not None:
        cci_50 = boroden_res.get("cci_50", 0)
        if cci_50 >= 60:
            score += 10
        elif cci_50 >= 30:
            score += 6
        elif cci_50 > 0:
            score += 2

    if vol_ratio >= 1.25:
        score += 6
    elif vol_ratio >= 1.0:
        score += 3

    if 32.0 <= current_rsi <= 52.0:
        score += 5
    elif current_rsi > 65.0:
        score -= 5

    score = max(35, min(100, int(round(score))))

    if score >= 80:
        return "HIGH", score
    elif score >= 62:
        return "MEDIUM", score
    else:
        return "LOW", score


def analyze_single_ticker(ticker, preloaded_df=None, fetch_name=True):
    state = load_learning_state()
    atr_mult = state.get("atr_multiplier", 0.50)
    long_wr, long_strict = get_long_stats(state)

    try:
        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)

        if df is None or df.empty:
            return {
                "status": "error",
                "message": f"Could not retrieve historical data for {ticker}.",
            }

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if len(df) < 50:
            return {
                "status": "error",
                "message": f"Insufficient price history for {ticker}.",
            }

        swing_high = df["High"].rolling(window=30).max()
        swing_low = df["Low"].rolling(window=30).min()

        df["Bullish_FVG"] = df["Low"] > df["High"].shift(2)

        body_size = abs(df["Close"] - df["Open"])
        avg_body = body_size.rolling(window=14).mean()
        displacement = body_size > (avg_body * 1.2)
        df["Bullish_MSB"] = (df["Close"] > df["High"].shift(1)) & displacement

        range_span = swing_high - swing_low
        fib_618 = swing_high - (range_span * 0.618)
        fib_660 = swing_high - (range_span * 0.660)
        df["In_Bullish_GP"] = (df["Low"] <= fib_618) & (df["Close"] >= fib_660)

        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        df["ATR"] = (
            pd.concat([high_low, high_close, low_close], axis=1)
            .max(axis=1)
            .rolling(14)
            .mean()
        )
        df["RSI"] = calculate_rsi(df["Close"], 14)

        today = df.iloc[-1]
        current_price = round(float(today["Close"]), 2)
        ticker_atr = (
            float(today["ATR"])
            if not pd.isna(today["ATR"])
            else (current_price * 0.02)
        )
        current_rsi = (
            round(float(today["RSI"]), 1) if not pd.isna(today["RSI"]) else 50.0
        )

        pivot_low_2bar = float(df["Low"].iloc[-2:].min())

        vol_today = float(today.get("Volume", 0.0))
        vol_avg20 = float(df["Volume"].iloc[-20:].mean())
        vol_ratio = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 1.0

        recent_10_high = round(float(df["High"].iloc[-10:].max()), 2)
        recent_10_low = round(float(df["Low"].iloc[-10:].min()), 2)
        recent_5_high = round(float(df["High"].iloc[-5:].max()), 2)

        ict_long_ok = (
            bool(today["Bullish_FVG"])
            and bool(today["Bullish_MSB"])
            and bool(today["In_Bullish_GP"])
        )
        if long_strict and current_rsi > 42.0:
            ict_long_ok = False

        boroden_res, nearest_long_sym = evaluate_boroden_fib_long(
            df, current_price, ticker_atr, atr_mult, pivot_low_2bar
        )
        boroden_long_ok = boroden_res is not None
        if long_strict and current_rsi > 42.0:
            boroden_long_ok = False

        # --- LONG SETUP PASSED ---
        if ict_long_ok or boroden_long_ok:
            company_name = get_stock_name(ticker) if fetch_name else ticker
            conf_tier, conf_score = score_setup_confidence(
                ict_long_ok, boroden_res, current_rsi, vol_ratio
            )

            sweet_stop, eff_stop_mult = calculate_sweet_spot_stop(
                current_price, pivot_low_2bar, ticker_atr, atr_mult
            )

            if ict_long_ok and boroden_long_ok:
                engine_tag = "🔥 DOUBLE CONFLUENCE LONG (ICT A+ & Boroden Fib)"
                stop_loss = sweet_stop
                risk = abs(current_price - stop_loss)
                target = max(
                    round(current_price + (risk * MIN_REWARD_RATIO), 2),
                    boroden_res["target"],
                )
                reason = (
                    f"ICT Bullish FVG + MSB inside 0.618-0.66 Golden Pocket ALIGNED with {boroden_res['setup_label']} "
                    f"(`${boroden_res['zone_low']}-${boroden_res['zone_high']}`), >34 EMA (`${boroden_res['ema_34']}`), "
                    f"CCI(14/50): `{boroden_res['cci_14']}/{boroden_res['cci_50']}` | Sweet-Spot Stop (`{eff_stop_mult}x ATR`)."
                )
            elif ict_long_ok:
                engine_tag = "Grade A+ ICT OTE"
                stop_loss = sweet_stop
                risk = abs(current_price - stop_loss)
                target = round(current_price + (risk * MIN_REWARD_RATIO), 2)
                reason = (
                    f"Bullish Fair Value Gap (FVG) and Market Structure Break (MSB) confirmed on 1D close inside the OTE "
                    f"0.618-0.66 Golden Pocket, protected by a `{eff_stop_mult}x ATR` Sweet-Spot Reversal-Pivot stop."
                )
            else:
                engine_tag = boroden_res["setup_label"]
                stop_loss = boroden_res["stop_loss"]
                eff_stop_mult = boroden_res["eff_stop_mult"]
                target = boroden_res["target"]
                reason = (
                    f"{boroden_res['setup_label']} tested `${boroden_res['zone_low']}-${boroden_res['zone_high']}` "
                    f"with 14-CCI (`{boroden_res['cci_14']}`) hooking up in an uptrend (>34 EMA `${boroden_res['ema_34']}`, "
                    f"50-CCI `{boroden_res['cci_50']}`) | Sweet-Spot Stop (`{eff_stop_mult}x ATR`)."
                )

            days = max(
                3,
                min(round(abs(target - current_price) / (ticker_atr * 0.45)), 25),
            )
            tf_str = f"~{days} to {days + 5} trading days (~{max(1, round(days/5))} to {round((days+5)/5)} weeks)"

            return {
                "status": "passed",
                "side": "LONG 📈",
                "confidence": conf_tier,
                "confidence_score": conf_score,
                "engine": engine_tag,
                "ticker": ticker,
                "name": company_name,
                "current_price": current_price,
                "entry": current_price,
                "stop_loss": stop_loss,
                "target": target,
                "timeframe": tf_str,
                "entry_action": f"Enter LONG after 1D (Daily) candle close above `${current_price}` (or on a 4H bullish candle close holding `${current_price}`)",
                "reason": reason,
            }

        # --- DIAGNOSTIC REJECTION REASONS ---
        if not fetch_name:
            return {"status": "rejected"}

        company_name = get_stock_name(ticker)
        failed_conditions = []
        if not bool(today["Bullish_FVG"]):
            failed_conditions.append(
                "ICT: No active Bullish Fair Value Gap (FVG) imbalance formed on recent 1D candles."
            )
        if not bool(today["Bullish_MSB"]):
            failed_conditions.append(
                "ICT: No energetic Bullish Market Structure Break (MSB) displacement on 1D close."
            )
        if not bool(today["In_Bullish_GP"]):
            failed_conditions.append(
                "ICT: Price is outside the Bullish 0.618 - 0.66 Fibonacci Golden Pocket (OTE) zone."
            )

        sym_note = (
            f"Long Symmetry Support: `${nearest_long_sym}`"
            if nearest_long_sym
            else "No valid prior swing symmetry levels nearby."
        )
        failed_conditions.append(
            f"Boroden Fib: Waiting for 14-CCI hook up at Symmetry/Cluster zone in an uptrend (>34 EMA). ({sym_note})"
        )

        if long_strict and current_rsi > 42.0:
            failed_conditions.append(
                f"Self-Learning Long Filter active (Win Rate {long_wr}% < 50%): Waiting for RSI <= 42 (Current: {current_rsi})."
            )

        local_span = max(recent_10_high - recent_10_low, ticker_atr * 1.5)
        local_bull_gp = round(recent_10_high - (local_span * 0.638), 2)
        if nearest_long_sym and nearest_long_sym < current_price:
            bull_entry = round(max(local_bull_gp, nearest_long_sym), 2)
        else:
            bull_entry = round(
                max(local_bull_gp, current_price - (ticker_atr * 1.1)), 2
            )

        bull_breakout_trigger = round(
            min(recent_5_high, current_price + (ticker_atr * 1.15)), 2
        )
        if bull_breakout_trigger <= bull_entry:
            bull_breakout_trigger = round(bull_entry + (ticker_atr * 0.75), 2)

        shift = max(0.0, atr_mult - 0.50)
        sweet_plan_mult = round(1.25 + shift, 2)
        bull_sl = round(bull_entry - (ticker_atr * sweet_plan_mult), 2)
        bull_tp = round(
            bull_entry + (abs(bull_entry - bull_sl) * MIN_REWARD_RATIO), 2
        )

        bull_label = (
            "Pullback OTE / Symmetry"
            if bull_entry < current_price
            else "Reclaim OTE / Symmetry"
        )
        bull_entry_action = (
            f"Enter LONG after a *4H candle close* holding above `${bull_entry}` ({bull_label}) "
            f"*OR* after a *1D (Daily) candle close* above `${bull_breakout_trigger}`"
        )
        bull_reason = (
            f"Confirms buyers defending OTE / Boroden Symmetry support on a 4H close with 14-CCI hooking up, "
            f"or breaking resistance on a 1D close with Bullish MSB + FVG."
        )

        return {
            "status": "rejected",
            "ticker": ticker,
            "name": company_name,
            "current_price": current_price,
            "sweet_plan_mult": sweet_plan_mult,
            "reasons": failed_conditions,
            "bull_plan": {
                "entry_action": bull_entry_action,
                "sl": bull_sl,
                "tp": bull_tp,
                "reason": bull_reason,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing {ticker}: {str(e)}",
        }


# ==========================================
# 9. TRADE MONITORING & CONFIDENCE-SORTED SCANNER
# ==========================================
def monitor_and_learn():
    active_trades = load_active_trades()
    state = load_learning_state()
    alerts = []
    updated_trades = {}

    for ticker, trade in active_trades.items():
        if "SHORT" in trade.get("Side", "LONG"):
            continue

        try:
            hist = yf.download(
                ticker, period="5d", interval="1d", progress=False
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                updated_trades[ticker] = trade
                continue

            high_price = float(hist["High"].iloc[-1])
            low_price = float(hist["Low"].iloc[-1])

            entry = trade["Entry"]
            target = trade["Target_Price"]
            stop_loss = trade["Stop_Loss"]
            shares = float(trade.get("Shares", 1.0))

            if high_price >= target:
                pnl_pct = round(((target - entry) / entry) * 100, 2)
                pnl_dollars = round((target - entry) * shares, 2)
                state["long_history"].append(True)
                update_journal_exit(ticker, "WIN", target, pnl_pct, pnl_dollars)
                alerts.append(
                    f"🏆 *LONG TARGET HIT!* {ticker} reached `${target}` (Entry: `${entry}`). Gain: `+{pnl_pct}%` (`+${pnl_dollars}`) 🎉"
                )
            elif low_price <= stop_loss:
                pnl_pct = round(((stop_loss - entry) / entry) * 100, 2)
                pnl_dollars = round((stop_loss - entry) * shares, 2)
                state["long_history"].append(False)
                update_journal_exit(
                    ticker, "LOSS", stop_loss, pnl_pct, pnl_dollars
                )
                state["audit_watchlist"][ticker] = {
                    "Side": "LONG",
                    "Target_Price": target,
                    "Stopped_Date": str(now_ct().date()),
                    "Days_Watched": 0,
                }
                alerts.append(
                    f"🧱🚗 *LONG STOP LOSS HIT!* {ticker} hit `${stop_loss}` (`{pnl_pct}%` / `${pnl_dollars}`). Bot is auditing for liquidity sweep fake-outs."
                )
            else:
                updated_trades[ticker] = trade
        except Exception:
            updated_trades[ticker] = trade

    remaining_watchlist = {}
    for ticker, audit in state.get("audit_watchlist", {}).items():
        if audit.get("Side") != "LONG":
            continue
        try:
            hist = yf.download(
                ticker, period="5d", interval="1d", progress=False
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                remaining_watchlist[ticker] = audit
                continue

            high_price = float(hist["High"].iloc[-1])
            orig_target = audit["Target_Price"]
            days_watched = audit.get("Days_Watched", 0) + 1

            if high_price >= orig_target:
                state["premature_stopouts"] = (
                    state.get("premature_stopouts", 0) + 1
                )
                old_mult = state["atr_multiplier"]
                new_mult = min(0.90, round(old_mult + 0.15, 2))
                state["atr_multiplier"] = new_mult
                shift = round(new_mult - 0.50, 2)
                note = f"Auto-Learned from {ticker} fake-out sweep: Shifted Sweet-Spot Stop band to {round(BASE_MIN_STOP_ATR + shift, 2)}x-{round(BASE_MAX_STOP_ATR + shift, 2)}x ATR."
                state["last_adjustment_note"] = note
                mark_journal_fakeout(ticker)
                alerts.append(
                    f"🧠 *SELF-LEARNING UPDATE:* {ticker} reversed and hit original target `${orig_target}` after stopping out. {note}"
                )
            elif days_watched < 3:
                audit["Days_Watched"] = days_watched
                remaining_watchlist[ticker] = audit
        except Exception:
            remaining_watchlist[ticker] = audit

    recent_longs = state.get("long_history", [])[-10:]
    if len(recent_longs) >= 8:
        long_wr = (sum(recent_longs) / len(recent_longs)) * 100
        if long_wr >= 72.0 and state["atr_multiplier"] > 0.50:
            old_mult = state["atr_multiplier"]
            new_mult = max(0.50, round(old_mult - 0.10, 2))
            state["atr_multiplier"] = new_mult
            shift = round(new_mult - 0.50, 2)
            state["last_adjustment_note"] = (
                f"High win rate ({round(long_wr, 1)}%): Tightened Sweet-Spot Stop band back to {round(BASE_MIN_STOP_ATR + shift, 2)}x-{round(BASE_MAX_STOP_ATR + shift, 2)}x ATR."
            )

    state["audit_watchlist"] = remaining_watchlist
    save_learning_state(state)
    save_active_trades(updated_trades)
    return alerts


def execute_aplus_scan(auto_mode=False, already_alerted=None):
    if already_alerted is None:
        already_alerted = set()

    messages_to_send = []

    learning_alerts = monitor_and_learn()
    if learning_alerts:
        messages_to_send.append(
            "🚨 *TRADE EXIT & LEARNING ALERTS*\n\n"
            + "\n\n".join(learning_alerts)
        )

    tickers = get_sp500_and_nasdaq_tickers()
    raw_candidates = []
    active_trades = load_active_trades()

    mode_label = "AUTO" if auto_mode else "MANUAL"
    print(
        f"🔍 [{mode_label}] Scanning {len(tickers)} tickers (LONG ONLY + Sweet-Spot Stops + Confidence Allocator)..."
    )

    batch_size = 50
    data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            batch_data = yf.download(
                batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
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
        if auto_mode and (
            ticker in already_alerted or ticker in active_trades
        ):
            continue

        result = analyze_single_ticker(
            ticker, preloaded_df=df_ticker, fetch_name=False
        )
        if result.get("status") == "passed":
            raw_candidates.append(result)

    raw_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)

    found_setups = []
    skipped_cash_setups = []
    newly_found_tickers = []
    today_str = str(now_ct().date())

    for res in raw_candidates:
        ticker = res["ticker"]
        c_name = sanitize_md(get_stock_name(ticker))
        side = res["side"]
        conf_key = res["confidence"]
        conf_score = res["confidence_score"]
        tier_info = CONFIDENCE_TIERS[conf_key]
        engine = sanitize_md(res["engine"])
        curr_price = res["current_price"]
        entry = res["entry"]
        sl = res["stop_loss"]
        tp = res["target"]
        tf = res["timeframe"]
        entry_act = res["entry_action"]
        reason = res["reason"]

        cap_status = get_capital_status(active_trades)
        shares, total_val, actual_risk, usable_cash = (
            allocate_position_by_confidence(entry, sl, conf_key, cap_status)
        )

        if ticker in active_trades and not auto_mode:
            existing = active_trades[ticker]
            shares = existing.get("Shares", shares)
            total_val = existing.get("Position_Value", total_val)
            actual_risk = round(shares * abs(entry - sl), 2)
        elif shares <= 0:
            skipped_cash_setups.append(
                f"⏸️ *{ticker}* ({tier_info['label']} `[{conf_score}/100]` | Price `${curr_price}`): "
                f"Skipped allocation — protecting `${HIGH_CONF_RESERVE:,.0f}` High-Confidence reserve (Usable cash: `${usable_cash:,.2f}`)."
            )
            continue
        else:
            active_trades[ticker] = {
                "Name": c_name,
                "Side": side,
                "Confidence": conf_key,
                "Confidence_Score": conf_score,
                "Engine": engine,
                "Current_Price": curr_price,
                "Entry": entry,
                "Stop_Loss": sl,
                "Target_Price": tp,
                "Timeframe": tf,
                "Shares": shares,
                "Position_Value": total_val,
                "Date_Added": today_str,
            }

            log_journal_entry({
                "Ticker": ticker,
                "Name": c_name,
                "Confidence": conf_key,
                "Confidence_Score": conf_score,
                "Engine": engine,
                "Date_Opened": today_str,
                "Entry": entry,
                "Stop_Loss": sl,
                "Target_Price": tp,
                "Shares": shares,
                "Position_Value": total_val,
                "Status": "OPEN",
                "Exit_Price": None,
                "Date_Closed": None,
                "PnL_Pct": 0.0,
                "PnL_Dollars": 0.0,
                "Notes": "",
            })

        newly_found_tickers.append(ticker)
        found_setups.append(
            f"🚨 *{ticker}* - {c_name}\n"
            f"   • 🎯 *Rating*: *{tier_info['label']}* (`Score: {conf_score}/100`)\n"
            f"   • 💰 *Current Price*: `${curr_price}` | *Engine*: _{engine}_\n"
            f"   • *Reason*: {reason}\n"
            f"   • 🚪 *Entry Action*: {entry_act}\n"
            f"   • *Est. Time Frame*: `{tf}`\n"
            f"   • 🧱🚗 Stop Loss: `${sl}` | 🏆 Target: `${tp}`\n"
            f"   • 💵 *Allocated Size*: `{shares} shares` (`${total_val:,.2f}` | Max Risk: `${actual_risk:,.2f}`)"
        )

    save_active_trades(active_trades)
    final_cap = get_capital_status(active_trades)

    if auto_mode and not found_setups and not learning_alerts:
        print(
            f"💤 [{now_ct().strftime('%H:%M')}] Auto-scan complete: No new setups found."
        )
        return [], newly_found_tickers

    if found_setups:
        header = (
            f"⚡ *LIVE AUTO-ALERT: NEW LONG SETUP!* (Free Cash: `${final_cap['available_cash']:,.2f}`)"
            if auto_mode
            else f"🎯 *S&P/NASDAQ LONG SETUPS ({len(found_setups)} ALLOCATED)*\n"
            f"💵 *Account*: `${final_cap['total_equity']:,.2f}` | *Invested*: `${final_cap['invested']:,.2f}` | *Free Cash*: `${final_cap['available_cash']:,.2f}` (incl. `${HIGH_CONF_RESERVE:,.0f}` High-Conf Reserve)"
        )
        current_chunk = header
        for block in found_setups[:15]:
            if len(current_chunk) + len(block) + 4 > 3800:
                messages_to_send.append(current_chunk)
                current_chunk = block
            else:
                current_chunk += "\n\n" + block
        if skipped_cash_setups:
            skip_block = "\n".join(skipped_cash_setups[:5])
            if len(current_chunk) + len(skip_block) + 4 > 3800:
                messages_to_send.append(current_chunk)
                current_chunk = skip_block
            else:
                current_chunk += "\n\n" + skip_block
        if current_chunk:
            messages_to_send.append(current_chunk)
    elif not auto_mode:
        messages_to_send.append(
            f"🎯 *S&P/NASDAQ LONG DUAL-ENGINE SCAN*\n"
            f"No new stocks met strict ICT A+ or Boroden Fibonacci criteria today.\n"
            f"💵 *Free Cash Available*: `${final_cap['available_cash']:,.2f}` (`${final_cap['general_cash']:,.2f}` general + `${HIGH_CONF_RESERVE:,.0f}` High-Conf reserve)."
        )

    return messages_to_send, newly_found_tickers


def auto_market_scanner():
    """Automatically scans every hour during US market hours (8 AM - 3 PM CT)."""
    last_scan_slot = None
    alerted_date = None
    alerted_today = set()

    scan_hours_ct = {8, 9, 10, 11, 12, 13, 14, 15}

    while True:
        now = now_ct()

        if alerted_date != now.date():
            alerted_date = now.date()
            alerted_today = set()

        current_slot = (now.date(), now.hour)
        if (
            now.weekday() < 5
            and now.hour in scan_hours_ct
            and now.minute == 0
            and last_scan_slot != current_slot
        ):
            last_scan_slot = current_slot
            try:
                is_auto = now.hour != 8
                if not is_auto:
                    bot.send_message(
                        MY_CHAT_ID,
                        "☀️ Pre-Market LONG-ONLY Confidence-Ranked Scan Starting ($10k Account + Sweet-Spot Stops)...",
                    )

                reports, new_tickers = execute_aplus_scan(
                    auto_mode=is_auto, already_alerted=alerted_today
                )
                alerted_today.update(new_tickers)

                for msg_part in reports:
                    send_safe_markdown(MY_CHAT_ID, msg_part)
                    time.sleep(0.3)

                if reports:
                    print(f"✅ Sent auto-alert to Telegram for: {new_tickers}")
            except Exception as e:
                print(f"❌ Auto-Scanner Error: {e}")

        time.sleep(20)


# ==========================================
# 10. TELEGRAM COMMAND & TICKER HANDLERS
# ==========================================
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    print(f"📩 Incoming from {message.chat.id}: {message.text}")

    if message.chat.id != MY_CHAT_ID:
        bot.reply_to(message, f"⛔ Unauthorized Chat ID: {message.chat.id}")
        return

    text = (message.text or "").strip().lower()
    raw_text = (message.text or "").strip()

    if text in ["/start", "start", "/help", "help"]:
        send_safe_markdown(
            MY_CHAT_ID,
            "🤖 *Self-Learning S&P 500 & Nasdaq 100 LONG-ONLY Bot Online!*\n"
            "_(Confidence-Ranked Allocator + Sweet-Spot Pivot Stops for $10,000 Account)_\n\n"
            "• `scan` - Scan & rank setups (HIGH / MEDIUM / LOW confidence) with auto cash allocation\n"
            "• `cash` - View live $10k account equity, invested capital & High-Confidence reserve\n"
            "• `journal` - Download Weekly Trading Journal PDF (Available Fridays after 3:00 PM CT)\n"
            "• `sector` - Anti-fakeout 11-Sector rotation (5D / 10D / 20D money flow)\n"
            "• `stats` - View live Long win rate & Sweet-Spot ATR stop status\n"
            "• `entry` / `stop loss` / `price target` - View active position levels\n"
            "• Type any ticker (e.g., `MSFT`, `NVDA`) for instant rating & 4H/1D trigger rules",
        )

    elif text in ["/cash", "cash", "balance", "capital"]:
        cap = get_capital_status()
        msg = (
            f"💵 *$10,000 ACCOUNT CAPITAL & RESERVE STATUS*\n\n"
            f"• *Total Account Equity*: `${cap['total_equity']:,.2f}` (Realized P&L: `${cap['realized_pnl']:+,.2f}`)\n"
            f"• *Currently Invested*: `${cap['invested']:,.2f}`\n"
            f"• *Total Free Cash*: `${cap['available_cash']:,.2f}`\n"
            f"   ├─ *General Cash (Medium/Low Plays)*: `${cap['general_cash']:,.2f}`\n"
            f"   └─ *Protected High-Confidence Reserve*: `${cap['reserve']:,.2f}`\n\n"
            f"📊 *Allocation Tiers:*\n"
            f"• 🔥 *HIGH (80–100)*: `2.5%` risk (`$250`), up to `35%` (`$3,500`)\n"
            f"• ⚡ *MEDIUM (62–79)*: `1.5%` risk (`$150`), up to `20%` (`$2,000`)\n"
            f"• 🔹 *LOW (<62)*: `0.75%` risk (`$75`), up to `10%` (`$1,000`)"
        )
        send_safe_markdown(MY_CHAT_ID, msg)

    elif text in ["/journal", "journal", "/journal force", "journal force"]:
        current_dt = now_ct()
        is_friday_after_3pm = (
            current_dt.weekday() == 4 and current_dt.hour >= 15
        )
        force_override = "force" in text

        if not is_friday_after_3pm and not force_override:
            day_name = current_dt.strftime("%A")
            time_str = current_dt.strftime("%I:%M %p CT")
            send_safe_markdown(
                MY_CHAT_ID,
                f"⏳ *Weekly Journal PDF Locked Until Friday 3:00 PM CT*\n\n"
                f"• *Current Time*: `{day_name}, {time_str}`\n"
                f"• Your weekly journal automatically unlocks every *Friday after 3:00 PM CT* once the market closes for the week.\n"
                f"_(Tip: Type `journal force` if you want to generate an early preview right now.)_",
            )
            return

        bot.reply_to(
            message,
            "📓 Generating your Weekly Trading Journal PDF with shares, invested capital & live P&L...",
        )
        try:
            pdf_path, caption = generate_weekly_journal_pdf()
            with open(pdf_path, "rb") as doc:
                bot.send_document(
                    MY_CHAT_ID, doc, caption=caption, parse_mode="Markdown"
                )
            print(f"✅ Sent Weekly Journal PDF: {pdf_path}")
        except Exception as e:
            print(f"❌ Journal PDF Error: {e}")
            bot.send_message(MY_CHAT_ID, f"❌ Error generating journal PDF: {e}")

    elif text in ["/sector", "sector", "sectors"]:
        bot.reply_to(
            message,
            "📊 Analyzing 5D / 10D / 20D institutional money flow across all 11 S&P 500 sectors...",
        )
        sector_report = analyze_sector_rotation()
        send_safe_markdown(MY_CHAT_ID, sector_report)

    elif text in ["/stats", "stats"]:
        state = load_learning_state()
        long_wr, long_strict = get_long_stats(state)
        atr_mult = state.get("atr_multiplier", 0.50)
        shift = round(max(0.0, atr_mult - 0.50), 2)
        min_band = round(BASE_MIN_STOP_ATR + shift, 2)
        max_band = round(BASE_MAX_STOP_ATR + shift, 2)
        fakeouts = state.get("premature_stopouts", 0)
        note = sanitize_md(
            state.get("last_adjustment_note", "No adjustments yet.")
        )

        msg = (
            f"🧠 *SELF-LEARNING ENGINE STATUS (LONG ONLY)*\n\n"
            f"• *Sweet-Spot Stop Band*: `{min_band}x – {max_band}x ATR` (Reversal-Pivot Anchored)\n"
            f"• *Liquidity Sweep Fake-Outs Caught*: `{fakeouts}`\n"
            f"• *Long Win Rate (Rolling)*: `{long_wr}%` ({'Strict RSI Filter Active' if long_strict else 'Normal Mode'})\n\n"
            f"📝 *Latest Adaptation*: _{note}_"
        )
        send_safe_markdown(MY_CHAT_ID, msg)

    elif text in ["/scan", "scan"]:
        bot.reply_to(
            message,
            "🔍 Scanning S&P 500 & Nasdaq 100 with Sweet-Spot Stops & Confidence Allocator ($10k Account)...",
        )
        try:
            reports, _ = execute_aplus_scan(auto_mode=False)
            for msg_part in reports:
                send_safe_markdown(MY_CHAT_ID, msg_part)
                time.sleep(0.3)
            print("✅ Manual scan report sent to Telegram!")
        except Exception as e:
            print(f"❌ Scan Error: {e}")
            bot.send_message(MY_CHAT_ID, f"❌ Error during scan: {e}")

    elif text in ["entry"]:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(
                message, "📁 No active trades saved yet. Run `scan` first."
            )
            return

        response_lines = ["🚪 *ACTIVE LONG ENTRIES (BY CONFIDENCE)*"]
        for ticker, data in trades.items():
            if "SHORT" in data.get("Side", "LONG"):
                continue
            conf = data.get("Confidence", "MED")
            alloc = data.get("Position_Value", 0.0)
            response_lines.append(
                f"🚪 *{ticker}* (`{conf}` | Invested: `${alloc:,.2f}`)\n  Current/Entry: `${data.get('Entry')}` ({data.get('Shares')} shares)"
            )
        send_safe_markdown(MY_CHAT_ID, "\n\n".join(response_lines))

    elif text in ["stop loss", "stoploss"]:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(
                message, "📁 No active trades saved yet. Run `scan` first."
            )
            return

        response_lines = ["🧱🚗 *ACTIVE LONG STOP LOSS LEVELS*"]
        for ticker, data in trades.items():
            if "SHORT" in data.get("Side", "LONG"):
                continue
            response_lines.append(
                f"🧱🚗 *{ticker}* (`{data.get('Confidence', 'MED')}`)\n  Stop Loss: `${data.get('Stop_Loss')}`"
            )
        send_safe_markdown(MY_CHAT_ID, "\n\n".join(response_lines))

    elif text in ["price target", "pricetarget", "target"]:
        trades = load_active_trades()
        if not trades:
            bot.reply_to(
                message, "📁 No active trades saved yet. Run `scan` first."
            )
            return

        response_lines = ["🏆 *ACTIVE LONG PRICE TARGETS & TIMEFRAMES*"]
        for ticker, data in trades.items():
            if "SHORT" in data.get("Side", "LONG"):
                continue
            response_lines.append(
                f"🏆 *{ticker}* (`{data.get('Confidence', 'MED')}`)\n  Target: `${data.get('Target_Price')}`\n  ⏱️ *Est. Time*: `{data.get('Timeframe')}`"
            )
        send_safe_markdown(MY_CHAT_ID, "\n\n".join(response_lines))

    else:
        ticker = raw_text.upper()
        bot.reply_to(
            message,
            f"🔍 Analyzing `{ticker}` (1-Year Data) with Sweet-Spot Stop & Confidence Engine...",
        )

        result = analyze_single_ticker(ticker, fetch_name=True)

        if result.get("status") == "passed":
            c_name = sanitize_md(result["name"])
            eng = sanitize_md(result["engine"])
            conf_key = result["confidence"]
            conf_score = result["confidence_score"]
            tier_info = CONFIDENCE_TIERS[conf_key]

            cap_status = get_capital_status()
            shares, total_val, actual_risk, usable_cash = (
                allocate_position_by_confidence(
                    result["entry"], result["stop_loss"], conf_key, cap_status
                )
            )

            alloc_line = (
                f"`{shares} shares` (`${total_val:,.2f}` invested | Max Risk: `${actual_risk:,.2f}`)"
                if shares > 0
                else f"`$0` (Insufficient usable cash: `${usable_cash:,.2f}`; `${HIGH_CONF_RESERVE:,.0f}` reserved for HIGH plays)"
            )

            msg = (
                f"✅ *LONG SETUP QUALIFIED ({result['side']})*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({c_name})\n"
                f"🎯 *Rating*: *{tier_info['label']}* (`Score: {conf_score}/100`)\n"
                f"💰 *Current Price*: `${result['current_price']}`\n"
                f"⚙️ *Engine*: _{eng}_\n"
                f"   • *Reason*: {result['reason']}\n"
                f"   • 🚪 *Entry Action*: {result['entry_action']}\n"
                f"   • *Est. Time Frame*: `{result['timeframe']}`\n"
                f"   • 🧱🚗 Stop Loss: `${result['stop_loss']}`\n"
                f"   • 🏆 Price Target: `${result['target']}`\n"
                f"   • 💵 *Recommended Allocation*: {alloc_line}"
            )
            send_safe_markdown(MY_CHAT_ID, msg)

        elif result.get("status") == "rejected":
            c_name = sanitize_md(result["name"])
            reasons_text = "\n".join([f"   • {r}" for r in result["reasons"]])
            bull = result["bull_plan"]
            plan_mult = result["sweet_plan_mult"]

            msg = (
                f"❌ *LONG SETUP REJECTED (NO POSITION YET)*\n"
                f"📌 *Ticker*: `{result['ticker']}` ({c_name})\n"
                f"💰 *Current Price*: `${result['current_price']}`\n\n"
                f"🛑 *Reasons for Skipping Today*:\n{reasons_text}\n\n"
                f"🔮 *GAME PLAN: WHEN TO GO LONG*\n\n"
                f"📈 *Bullish Trigger Plan:*\n"
                f"   • 🚪 *Entry Action*: {bull['entry_action']}\n"
                f"   • *Why*: {bull['reason']}\n"
                f"   • 🧱🚗 *Planned Stop Loss*: `${bull['sl']}` (`{plan_mult}x ATR` Sweet-Spot)\n"
                f"   • 🏆 *Planned Target*: `${bull['tp']}`"
            )
            send_safe_markdown(MY_CHAT_ID, msg)
        else:
            bot.send_message(
                MY_CHAT_ID,
                f"⚠️ {result.get('message', 'Could not analyze ticker.')}",
            )


if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)

    threading.Thread(target=auto_market_scanner, daemon=True).start()
    print(
        "🤖 Self-Learning LONG-ONLY Bot (Sweet-Spot Stops + Confidence Allocator + Weekly PDF Journal) online!"
    )
    bot.infinity_polling(skip_pending=True, timeout=20, long_polling_timeout=20)
