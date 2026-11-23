# ==============================================================================
# 1. IMPORTS, CONFIGURATION & TOKEN RESOLUTION
# ==============================================================================
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import html
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from typing import Any, Optional
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # Headless rendering for Telegram PNG charts
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import requests
import yfinance as yf
from telegram import Update
from telegram.error import RetryAfter, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Secrets must come from .env or environment variables (never hardcoded in source)
TELEGRAM_BOT_TOKEN_FALLBACK = ""
TELEGRAM_CHAT_ID_FALLBACK = ""

STARTING_CAPITAL = 20000.00
JOURNAL_FILE = Path("trade_journal.json")
HOLDINGS_FILE = Path("holdings.json")
TICKER_LEARNER_FILE = Path("ticker_learner.json")
LEGACY_MISTAKES_FILE = Path("mistake_learner.json")
ALERT_STATE_FILE = Path("alert_state.json")
BOT_NAME = "SandeepPick_bot"
CT_TZ = ZoneInfo("America/Chicago")
ET_TZ = ZoneInfo("America/New_York")
BATCH_SIZE = 75
TELEGRAM_MAX_LEN = 4000
TELEGRAM_CAPTION_MAX_LEN = 980

# Scheduler intervals
AUTO_SCAN_INTERVAL_SECONDS = 1 * 3600      # Auto-scan for entry alerts every 1 hour
EXIT_MONITOR_INTERVAL_SECONDS = 4 * 3600   # Check bought positions for exit/stop-loss every 4 hours
HEADS_UP_COOLDOWN_HOURS = 12               # Cooldown between low-priority Heads-Up Watch alerts per ticker
AUTO_BUY_DEFAULT = True                    # Automatically track triggered entry signals in holdings.json
MAX_AUTO_OPEN_POSITIONS = 5                # Max concurrent auto-bought positions
AUTO_POSITION_PCT = 0.10                   # 10% of STARTING_CAPITAL per auto-bought trade
MIN_ALERT_STRATEGIES = 2                   # Require at least 2 converging strategies for Entry alerts
MIN_AUTO_BUY_SCORE = 75.0                  # Only Auto-Buy MEDIUM (75-85) or HIGH (>85) setups
REQUIRE_SPY_ABOVE_200EMA = True            # Pause new entries when SPY is below its daily 200 EMA
BREAKEVEN_R_TRIGGER = 1.0                  # Raise stop to breakeven once unrealized gain >= 1.0R

TAG_ENTRY = "Entry \U0001f6aa"
TAG_STOPLOSS = "Stop Loss \U0001f9f1\U0001f697"
TAG_TARGET = "Target \U0001f3c5"
TAG_SELL = "Sell \U0001f4e4"

ALL_STRATEGIES = [
    "4H Boroden Fib Cluster",
    "20d Breakout",
    "ICT Bullish FVG",
    "ICT Liquidity Sweep MSS",
    "IPO VWAP + Volume Reversal",
    "Bullish Butterfly Harmonic",
]

TOP_SPY_HOLDINGS = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "LLY", "BRK-B"]

SECTOR_ETFS: dict[str, tuple[str, str]] = {
    "XLK": ("Technology", "OFFENSIVE"),
    "XLY": ("Consumer Disc.", "OFFENSIVE"),
    "XLC": ("Comm. Services", "OFFENSIVE"),
    "XLF": ("Financials", "CYCLICAL"),
    "XLI": ("Industrials", "CYCLICAL"),
    "XLE": ("Energy", "CYCLICAL"),
    "XLB": ("Materials", "CYCLICAL"),
    "XLV": ("Health Care", "DEFENSIVE"),
    "XLP": ("Consumer Staples", "DEFENSIVE"),
    "XLU": ("Utilities", "DEFENSIVE"),
    "XLRE": ("Real Estate", "DEFENSIVE"),
}

TICKER_REGEX = re.compile(r"^[\$/]?([A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?)$")
RESERVED_COMMANDS = {
    "START", "HELP", "SCAN", "BORODEN", "AUTOBUY",
    "REGIME", "SECTOR", "OPEN", "CLOSE", "CROSSOVER",
    "PATTERN", "REPORT", "BUY", "SELL", "LEARN",
}

_STATE_LOCK = threading.RLock()
_SCAN_LOCK = asyncio.Lock()


def _load_env_files() -> None:
    """Load KEY=VALUE pairs from .env in the script folder or current working directory."""
    candidates = [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError as exc:
            logger.warning("Could not read %s: %s", env_path, exc)


def _save_token_to_env(token: str) -> Path:
    """Persist TELEGRAM_BOT_TOKEN into .env next to this script."""
    env_path = Path(__file__).resolve().parent / ".env"
    existing_lines: list[str] = []
    if env_path.is_file():
        existing_lines = [
            ln for ln in env_path.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("TELEGRAM_BOT_TOKEN=")
        ]
    existing_lines.append(f"TELEGRAM_BOT_TOKEN={token}")
    env_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
    return env_path


def resolve_telegram_token(cli_token: Optional[str] = None) -> str:
    """Resolve Telegram bot token from CLI flag, .env/environment, fallback constant, or prompt."""
    _load_env_files()
    token = (cli_token or os.environ.get("TELEGRAM_BOT_TOKEN") or TELEGRAM_BOT_TOKEN_FALLBACK or "").strip()
    if token:
        return token

    if sys.stdin and sys.stdin.isatty():
        print("\nTELEGRAM_BOT_TOKEN is not set.")
        entered = input("Paste your Telegram Bot Token (from @BotFather): ").strip()
        if entered:
            os.environ["TELEGRAM_BOT_TOKEN"] = entered
            try:
                saved_path = _save_token_to_env(entered)
                print(f"Saved token to {saved_path} so you won't be asked again.")
            except OSError as exc:
                logger.warning("Could not save token to .env: %s", exc)
            return entered

    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not set. Create a .env file next to bot_3.py with "
        "TELEGRAM_BOT_TOKEN=your_token, pass --token YOUR_TOKEN, or set the environment variable."
    )


def _resolve_alert_chat_id(app: Application) -> Optional[int | str]:
    """Return active chat ID for automated 1H/4H alerts."""
    cid = app.bot_data.get("alert_chat_id")
    if cid:
        return cid
    env_cid = (os.environ.get("TELEGRAM_CHAT_ID") or TELEGRAM_CHAT_ID_FALLBACK or "").strip()
    if env_cid:
        return int(env_cid) if env_cid.lstrip("-").isdigit() else env_cid
    return None


def _is_us_equity_session_open(now_dt: Optional[datetime] = None) -> bool:
    """Return True during regular US market hours (Mon-Fri 09:30-16:15 ET) so weekends don't re-fire stale bars."""
    now_et = (now_dt or datetime.now(ET_TZ)).astimezone(ET_TZ)
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return time(9, 30) <= t <= time(16, 15)


def _current_4h_session_progress(now_dt: Optional[datetime] = None) -> float:
    """Return elapsed fraction [0.30, 1.0] of the active US session 4H bucket (09:30-13:30 ET or 13:30-16:00 ET)."""
    now_et = (now_dt or datetime.now(ET_TZ)).astimezone(ET_TZ)
    if not _is_us_equity_session_open(now_et):
        return 1.0
    mins_since_midnight = now_et.hour * 60 + now_et.minute
    morning_open = 9 * 60 + 30    # 09:30 ET (570)
    afternoon_open = 13 * 60 + 30  # 13:30 ET (810)
    market_close = 16 * 60         # 16:00 ET (960)
    if mins_since_midnight < afternoon_open:
        elapsed = max(1, mins_since_midnight - morning_open)
        return max(0.30, min(1.0, elapsed / 240.0))
    elapsed = max(1, mins_since_midnight - afternoon_open)
    return max(0.30, min(1.0, elapsed / float(market_close - afternoon_open)))


def _to_naive_et_timestamp(ts_val: Any) -> Optional[pd.Timestamp]:
    """Normalize any timestamp or date string into a tz-naive Eastern Time (market time) Timestamp."""
    if ts_val is None or ts_val == "":
        return None
    try:
        ts = pd.to_datetime(ts_val)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            return ts.tz_convert(ET_TZ).tz_localize(None)
        return ts
    except Exception:
        return None


def _to_naive_et_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize a DatetimeIndex into tz-naive Eastern Time (market time)."""
    dt_idx = pd.to_datetime(index)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_convert(ET_TZ).tz_localize(None)
    return dt_idx.astype("datetime64[ns]")


@dataclass
class Boroden4HConfig:
    pivot_len: int = 5
    cluster_tol_pct: float = 0.35
    min_cluster: int = 2
    stop_buffer_pct: float = 0.0
    swing_1h_left: int = 2
    swing_1h_right: int = 2
    swing_1h_fallback_lookback: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 1.25
    breakeven_after_tp1: bool = True
    max_hold_bars: int = 24
    rsi_period: int = 14
    rsi_reclaim_level: float = 42.0
    vol_mult: float = 1.20
    tp1_ratio: float = 1.272
    tp2_ratio: float = 1.618
    tp3_ratio: float = 2.618
    initial_capital: float = 10_000.0
    position_pct: float = 0.10


# ==============================================================================
# 2. THREAD-SAFE PERSISTENCE, RESTART-SAFE DEDUP & PER-TICKER AUTO-LEARNER
# ==============================================================================
def _atomic_write_json(path: Path, data: Any) -> None:
    with _STATE_LOCK:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)


def load_alert_state() -> dict[str, dict[str, str]]:
    """Load persisted alert deduplication maps and prune entries older than 7 days."""
    with _STATE_LOCK:
        state: dict[str, dict[str, str]] = {"entry_alerts": {}, "heads_up_alerts": {}}
        if ALERT_STATE_FILE.exists():
            try:
                with ALERT_STATE_FILE.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    state["entry_alerts"] = dict(raw.get("entry_alerts", {}))
                    state["heads_up_alerts"] = dict(raw.get("heads_up_alerts", {}))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", ALERT_STATE_FILE, exc)

        cutoff = datetime.now(ET_TZ) - timedelta(days=7)
        for bucket in ("entry_alerts", "heads_up_alerts"):
            cleaned: dict[str, str] = {}
            for k, ts_str in state[bucket].items():
                try:
                    dt_val = datetime.fromisoformat(ts_str)
                    if dt_val.tzinfo is None:
                        dt_val = dt_val.replace(tzinfo=ET_TZ)
                    if dt_val >= cutoff:
                        cleaned[k] = ts_str
                except ValueError:
                    continue
            state[bucket] = cleaned
        return state


def save_alert_state(state: dict[str, dict[str, str]]) -> None:
    _atomic_write_json(ALERT_STATE_FILE, state)


def load_journal() -> list[dict[str, Any]]:
    with _STATE_LOCK:
        if JOURNAL_FILE.exists():
            try:
                with JOURNAL_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", JOURNAL_FILE, exc)
        return []


def save_journal(entries: list[dict[str, Any]]) -> None:
    _atomic_write_json(JOURNAL_FILE, entries)


def load_holdings() -> dict[str, dict[str, Any]]:
    """Load held positions: { 'AAPL': {'entry': 180.0, 'stop': 172.0, 'target': 195.0, 'shares': 10} }."""
    with _STATE_LOCK:
        if HOLDINGS_FILE.exists():
            try:
                with HOLDINGS_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", HOLDINGS_FILE, exc)
        return {}


def save_holdings(holdings: dict[str, dict[str, Any]]) -> None:
    _atomic_write_json(HOLDINGS_FILE, holdings)


def _default_ticker_profile(sym: str, base_cfg: Optional[Boroden4HConfig] = None) -> dict[str, Any]:
    """Create a fresh, isolated learning profile for a single ticker."""
    cfg = base_cfg or Boroden4HConfig()
    now_iso = datetime.now(ET_TZ).isoformat()
    return {
        "ticker": sym,
        "upgrade_version": 1,
        "created_at": now_iso,
        "last_upgraded_at": now_iso,
        "last_bar_seen": None,
        "volatility_regime": "STANDARD",
        "avg_atr_pct": None,
        "atr_stop_mult": round(cfg.atr_stop_mult, 2),
        "min_cluster": int(cfg.min_cluster),
        "min_rsi_reclaim": round(cfg.rsi_reclaim_level, 1),
        "vol_mult": round(cfg.vol_mult, 2),
        "tp_scale_mult": 1.00,
        "strategy_weights": {s: 1.00 for s in ALL_STRATEGIES},
        "strategy_stats": {s: {"wins": 0, "losses": 0} for s in ALL_STRATEGIES},
        "trade_events": [],
        "learned_trade_keys": [],
        "pending_signals": [],
        "wins": 0,
        "losses": 0,
        "upgrade_log": [f"{now_iso[:16]} v1: Initialized isolated DNA profile for {sym}"],
    }


def _ensure_profile_schema(sym: str, prof: dict[str, Any], base_cfg: Optional[Boroden4HConfig] = None) -> bool:
    """Backfill missing schema keys on an existing ticker profile; returns True if modified."""
    default_p = _default_ticker_profile(sym, base_cfg)
    changed = False
    for k, v in default_p.items():
        if k not in prof:
            prof[k] = v
            changed = True
    for s in ALL_STRATEGIES:
        if s not in prof["strategy_weights"]:
            prof["strategy_weights"][s] = 1.00
            changed = True
        if s not in prof["strategy_stats"]:
            prof["strategy_stats"][s] = {"wins": 0, "losses": 0}
            changed = True
    return changed


def load_ticker_profiles() -> dict[str, dict[str, Any]]:
    """Load per-ticker learning profiles, migrating legacy mistake_learner.json if present."""
    with _STATE_LOCK:
        if TICKER_LEARNER_FILE.exists():
            try:
                with TICKER_LEARNER_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load %s: %s", TICKER_LEARNER_FILE, exc)

        if LEGACY_MISTAKES_FILE.exists():
            try:
                with LEGACY_MISTAKES_FILE.open("r", encoding="utf-8") as f:
                    legacy = json.load(f)
                if isinstance(legacy, dict) and legacy:
                    migrated: dict[str, dict[str, Any]] = {}
                    cfg = Boroden4HConfig()
                    for raw_sym, old_p in legacy.items():
                        sym = raw_sym.strip().upper().replace(".", "-")
                        prof = _default_ticker_profile(sym, cfg)
                        prof["atr_stop_mult"] = round(cfg.atr_stop_mult + float(old_p.get("extra_atr_mult", 0.0)), 2)
                        prof["min_cluster"] = int(cfg.min_cluster + int(old_p.get("extra_cluster_req", 0)))
                        prof["min_rsi_reclaim"] = round(float(old_p.get("min_rsi_reclaim", cfg.rsi_reclaim_level)), 1)
                        old_mistakes = old_p.get("mistakes", [])
                        prof["losses"] = len(old_mistakes)
                        for m in old_mistakes:
                            prof["trade_events"].append({
                                "outcome": "LOSS",
                                "exit_time": str(m.get("date", "")),
                                "entry": float(m.get("entry", 0.0)),
                                "exit": float(m.get("exit", 0.0)),
                                "stop": float(m.get("stop", 0.0)),
                                "target": float(m.get("entry", 0.0)) * 1.03,
                                "pnl": float(m.get("pnl", 0.0)),
                                "reason": str(m.get("reason", "MIGRATED_LOSS")),
                                "strategies": [],
                                "post_exit_observed": bool(m.get("observed", False)),
                                "post_exit_behavior": str(m.get("behavior", "pending")),
                            })
                        migrated[sym] = prof
                    save_ticker_profiles(migrated)
                    return migrated
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to migrate %s: %s", LEGACY_MISTAKES_FILE, exc)
        return {}


def save_ticker_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    _atomic_write_json(TICKER_LEARNER_FILE, profiles)


def _merge_and_save_scanned_profiles(
    scanned_profiles: dict[str, dict[str, Any]],
    updated_syms: set[str],
) -> None:
    """Atomically merge scanned per-ticker DNA updates without clobbering concurrent trade outcomes."""
    if not updated_syms:
        return
    with _STATE_LOCK:
        disk_profiles = load_ticker_profiles()
        for sym in updated_syms:
            if sym not in scanned_profiles:
                continue
            scan_p = scanned_profiles[sym]
            disk_p = disk_profiles.get(sym)
            if disk_p is None:
                disk_profiles[sym] = scan_p
                continue

            disk_keys = set(disk_p.get("learned_trade_keys", []))
            scan_keys = set(scan_p.get("learned_trade_keys", []))
            new_disk_keys = disk_keys - scan_keys
            if new_disk_keys:
                for k in ("volatility_regime", "avg_atr_pct", "last_bar_seen", "pending_signals"):
                    disk_p[k] = scan_p.get(k)
                disk_p["atr_stop_mult"] = max(
                    float(disk_p.get("atr_stop_mult", 1.25)),
                    float(scan_p.get("atr_stop_mult", 1.25)),
                )
                disk_p["min_cluster"] = max(
                    int(disk_p.get("min_cluster", 2)),
                    int(scan_p.get("min_cluster", 2)),
                )
                disk_p["min_rsi_reclaim"] = max(
                    float(disk_p.get("min_rsi_reclaim", 42.0)),
                    float(scan_p.get("min_rsi_reclaim", 42.0)),
                )
                disk_p["vol_mult"] = max(
                    float(disk_p.get("vol_mult", 1.20)),
                    float(scan_p.get("vol_mult", 1.20)),
                )
                disk_profiles[sym] = disk_p
            else:
                disk_profiles[sym] = scan_p
        save_ticker_profiles(disk_profiles)


def get_ticker_learning_profile(
    ticker: str,
    base_cfg: Optional[Boroden4HConfig] = None,
    profiles_cache: Optional[dict[str, dict[str, Any]]] = None,
    defer_save: bool = False,
) -> dict[str, Any]:
    """Return the isolated learning profile for a single ticker (creating it if new)."""
    sym = ticker.strip().upper().replace(".", "-")
    with _STATE_LOCK:
        profiles = profiles_cache if profiles_cache is not None else load_ticker_profiles()
        changed = False
        if sym not in profiles:
            profiles[sym] = _default_ticker_profile(sym, base_cfg)
            changed = True
        else:
            changed = _ensure_profile_schema(sym, profiles[sym], base_cfg)
        if changed and not defer_save:
            save_ticker_profiles(profiles)
        return profiles[sym]


def _append_upgrade_note(profile: dict[str, Any], note: str) -> None:
    now_iso = datetime.now(ET_TZ).isoformat()
    profile["upgrade_version"] = int(profile.get("upgrade_version", 1)) + 1
    profile["last_upgraded_at"] = now_iso
    entry = f"{now_iso[:16]} v{profile['upgrade_version']}: {note}"
    log_list = profile.setdefault("upgrade_log", [])
    log_list.append(entry)
    profile["upgrade_log"] = log_list[-25:]


def _extract_matched_strategies_from_label(strategy_label: str) -> list[str]:
    if not strategy_label:
        return []
    return [s for s in ALL_STRATEGIES if s.lower() in strategy_label.lower()]


def record_ticker_trade_outcome(
    ticker: str,
    entry_price: float,
    exit_price: float,
    stop_price: float,
    target_price: float,
    pnl: float,
    exit_time_str: str,
    reason: str = "EXIT",
    strategy_label: str = "",
    trade_key: Optional[str] = None,
    profiles_cache: Optional[dict[str, dict[str, Any]]] = None,
    defer_save: bool = False,
) -> dict[str, Any]:
    """
    Upgrade a single ticker's profile after ANY closed trade (WIN or LOSS).
    Deduplicates by `trade_key` so repeated backtests never double-count historical trades.
    """
    sym = ticker.strip().upper().replace(".", "-")
    is_win = pnl > 0
    outcome = "WIN" if is_win else ("LOSS" if pnl < 0 else "BREAKEVEN")
    matched = _extract_matched_strategies_from_label(strategy_label)

    with _STATE_LOCK:
        profiles = profiles_cache if profiles_cache is not None else load_ticker_profiles()
        prof = profiles.get(sym) or _default_ticker_profile(sym)
        _ensure_profile_schema(sym, prof)

        learned_keys: list[str] = prof.setdefault("learned_trade_keys", [])
        if trade_key:
            if trade_key in learned_keys:
                profiles[sym] = prof
                return prof
            learned_keys.append(trade_key)
            prof["learned_trade_keys"] = learned_keys[-250:]

        if is_win:
            prof["wins"] = int(prof.get("wins", 0)) + 1
            prof["tp_scale_mult"] = round(min(1.35, float(prof.get("tp_scale_mult", 1.0)) + 0.03), 2)
            prof["atr_stop_mult"] = round(max(1.05, float(prof.get("atr_stop_mult", 1.25)) - 0.03), 2)
            for s in matched:
                st = prof["strategy_stats"].setdefault(s, {"wins": 0, "losses": 0})
                st["wins"] = int(st.get("wins", 0)) + 1
                w = float(prof["strategy_weights"].get(s, 1.0))
                prof["strategy_weights"][s] = round(min(1.45, w + 0.06), 2)
        elif pnl < 0:
            prof["losses"] = int(prof.get("losses", 0)) + 1
            for s in matched:
                st = prof["strategy_stats"].setdefault(s, {"wins": 0, "losses": 0})
                st["losses"] = int(st.get("losses", 0)) + 1
                w = float(prof["strategy_weights"].get(s, 1.0))
                prof["strategy_weights"][s] = round(max(0.55, w - 0.07), 2)

        prof.setdefault("trade_events", []).append({
            "outcome": outcome,
            "exit_time": str(exit_time_str),
            "entry": round(float(entry_price), 4),
            "exit": round(float(exit_price), 4),
            "stop": round(float(stop_price), 4),
            "target": round(float(target_price), 4),
            "pnl": round(float(pnl), 2),
            "reason": reason,
            "strategies": matched,
            "post_exit_observed": False,
            "post_exit_behavior": "pending",
        })
        prof["trade_events"] = prof["trade_events"][-40:]

        strat_note = f" ({', '.join(matched)})" if matched else ""
        _append_upgrade_note(
            prof,
            f"Trade {outcome} (${pnl:+.2f}, {reason}){strat_note} -> ATRx={prof['atr_stop_mult']:.2f}, TPx={prof['tp_scale_mult']:.2f}",
        )
        profiles[sym] = prof
        if not defer_save:
            save_ticker_profiles(profiles)
        return prof


def register_ticker_shadow_signal(
    ticker: str,
    signal_time: Any,
    entry_px: float,
    stop_px: float,
    tp1_px: float,
    matched_strategies: list[str],
    timeframe: str = "4H",
    profiles_cache: Optional[dict[str, dict[str, Any]]] = None,
    defer_save: bool = False,
) -> None:
    """Record an active signal checkpoint on `ticker` tagged with its native timeframe (`4H` or `1D`)."""
    sym = ticker.strip().upper().replace(".", "-")
    if not matched_strategies or entry_px <= 0:
        return
    norm_ts = _to_naive_et_timestamp(signal_time)
    if norm_ts is None:
        return
    sig_ts_str = norm_ts.isoformat()

    with _STATE_LOCK:
        profiles = profiles_cache if profiles_cache is not None else load_ticker_profiles()
        prof = profiles.get(sym) or _default_ticker_profile(sym)
        _ensure_profile_schema(sym, prof)
        pending = prof.setdefault("pending_signals", [])
        if any(p.get("signal_time") == sig_ts_str and p.get("tf", "4H") == timeframe for p in pending):
            profiles[sym] = prof
            return
        pending.append({
            "signal_time": sig_ts_str,
            "tf": timeframe,
            "entry": round(float(entry_px), 4),
            "stop": round(float(stop_px), 4),
            "tp1": round(float(tp1_px), 4),
            "strategies": list(matched_strategies),
        })
        prof["pending_signals"] = pending[-15:]
        profiles[sym] = prof
        if not defer_save:
            save_ticker_profiles(profiles)


def upgrade_ticker_learning_profile(
    ticker: str,
    recent_df: pd.DataFrame,
    base_cfg: Optional[Boroden4HConfig] = None,
    daily_df: Optional[pd.DataFrame] = None,
    profiles_cache: Optional[dict[str, dict[str, Any]]] = None,
    defer_save: bool = False,
) -> dict[str, Any]:
    """
    Continuously upgrades ONLY `ticker`'s isolated profile using:
      1. Ticker-specific volatility DNA (ATR% calibration).
      2. Timestamp-accurate post-exit observation on closed trades (strictly bars AFTER exit_time in ET).
      3. Timeframe-matched forward shadow-signal verification (`4H` signals on 4H bars, `1D` signals on 1D bars).
    """
    cfg = base_cfg or Boroden4HConfig()
    sym = ticker.strip().upper().replace(".", "-")

    with _STATE_LOCK:
        profiles = profiles_cache if profiles_cache is not None else load_ticker_profiles()
        prof = profiles.get(sym) or _default_ticker_profile(sym, cfg)
        _ensure_profile_schema(sym, prof, cfg)

        if recent_df is None or recent_df.empty or len(recent_df) < 10:
            profiles[sym] = prof
            if not defer_save:
                save_ticker_profiles(profiles)
            return prof

        changes: list[str] = []
        dt_index = _to_naive_et_index(recent_df.index)
        daily_dt_index = _to_naive_et_index(daily_df.index) if (daily_df is not None and not daily_df.empty) else None
        latest_bar_str = str(dt_index[-1])

        atr_series = compute_atr(recent_df, cfg.atr_period)
        atr_pct = float((atr_series.iloc[-1] / max(float(recent_df["close"].iloc[-1]), 1e-6)) * 100.0)
        prev_atr_pct = prof.get("avg_atr_pct")
        blended_atr_pct = atr_pct if prev_atr_pct is None else (0.7 * float(prev_atr_pct) + 0.3 * atr_pct)
        prof["avg_atr_pct"] = round(blended_atr_pct, 3)

        if blended_atr_pct >= 2.20:
            vol_regime = "HIGH_VOL"
            floor_atr_mult = 1.45
        elif blended_atr_pct <= 0.90:
            vol_regime = "LOW_VOL"
            floor_atr_mult = 1.15
        else:
            vol_regime = "STANDARD"
            floor_atr_mult = cfg.atr_stop_mult

        if prof.get("volatility_regime") != vol_regime:
            prof["volatility_regime"] = vol_regime
            changes.append(f"VolDNA->{vol_regime} ({blended_atr_pct:.2f}% ATR)")

        if float(prof.get("atr_stop_mult", cfg.atr_stop_mult)) < floor_atr_mult:
            prof["atr_stop_mult"] = round(floor_atr_mult, 2)
            changes.append(f"ATRx->{prof['atr_stop_mult']:.2f}")

        for ev in prof.get("trade_events", []):
            if ev.get("post_exit_observed"):
                continue
            exit_ts = _to_naive_et_timestamp(ev.get("exit_time"))
            if exit_ts is None:
                continue

            post_mask = dt_index > np.datetime64(exit_ts)
            post_bars = recent_df.loc[post_mask]
            if len(post_bars) < 2:
                continue

            post_high = float(post_bars["high"].head(12).max())
            ev_entry = float(ev.get("entry", 0.0))
            ev_exit = float(ev.get("exit", 0.0))
            ev_target = float(ev.get("target", ev_entry * 1.03))
            ev_strats = ev.get("strategies", [])

            if ev.get("outcome") == "LOSS":
                if post_high >= max(ev_entry, ev_exit * 1.015):
                    ev["post_exit_behavior"] = "stop_hunt_rebound"
                    prof["atr_stop_mult"] = round(min(2.35, float(prof["atr_stop_mult"]) + 0.15), 2)
                    prof["min_rsi_reclaim"] = round(min(56.0, float(prof["min_rsi_reclaim"]) + 1.5), 1)
                    changes.append(f"StopHunt->ATRx {prof['atr_stop_mult']:.2f}, RSI>={prof['min_rsi_reclaim']:.0f}")
                else:
                    ev["post_exit_behavior"] = "true_breakdown"
                    prof["min_cluster"] = min(4, int(prof["min_cluster"]) + 1)
                    prof["vol_mult"] = round(min(1.80, float(prof["vol_mult"]) + 0.08), 2)
                    for s in ev_strats:
                        prof["strategy_weights"][s] = round(max(0.55, float(prof["strategy_weights"].get(s, 1.0)) - 0.04), 2)
                    changes.append(f"Breakdown->Cluster>={prof['min_cluster']}, Vol>={prof['vol_mult']:.2f}x")
                ev["post_exit_observed"] = True

            elif ev.get("outcome") == "WIN":
                if post_high >= ev_target * 1.02:
                    ev["post_exit_behavior"] = "runner_continuation"
                    prof["tp_scale_mult"] = round(min(1.40, float(prof.get("tp_scale_mult", 1.0)) + 0.05), 2)
                    changes.append(f"Runner->TPx {prof['tp_scale_mult']:.2f}")
                else:
                    ev["post_exit_behavior"] = "target_fade"
                ev["post_exit_observed"] = True

        still_pending: list[dict[str, Any]] = []
        for sig in prof.get("pending_signals", []):
            sig_ts = _to_naive_et_timestamp(sig.get("signal_time"))
            if sig_ts is None:
                continue

            sig_tf = str(sig.get("tf", "4H")).upper()
            if sig_tf == "1D":
                if daily_df is None or daily_dt_index is None:
                    still_pending.append(sig)
                    continue
                eval_df = daily_df
                eval_idx = daily_dt_index
            else:
                eval_df = recent_df
                eval_idx = dt_index

            fwd_bars = eval_df.loc[eval_idx > np.datetime64(sig_ts)].head(16)
            if fwd_bars.empty:
                still_pending.append(sig)
                continue

            s_stop = float(sig.get("stop", 0.0))
            s_tp1 = float(sig.get("tp1", 0.0))
            s_strats = sig.get("strategies", [])
            resolved = False

            for _, bar in fwd_bars.iterrows():
                b_low = float(bar["low"])
                b_high = float(bar["high"])
                if s_stop > 0 and b_low <= s_stop:
                    for s in s_strats:
                        st = prof["strategy_stats"].setdefault(s, {"wins": 0, "losses": 0})
                        st["losses"] = int(st.get("losses", 0)) + 1
                        prof["strategy_weights"][s] = round(max(0.55, float(prof["strategy_weights"].get(s, 1.0)) - 0.03), 2)
                    changes.append(f"ShadowStop[{sig_tf}]({','.join(s_strats[:2])})")
                    resolved = True
                    break
                if s_tp1 > 0 and b_high >= s_tp1:
                    for s in s_strats:
                        st = prof["strategy_stats"].setdefault(s, {"wins": 0, "losses": 0})
                        st["wins"] = int(st.get("wins", 0)) + 1
                        prof["strategy_weights"][s] = round(min(1.45, float(prof["strategy_weights"].get(s, 1.0)) + 0.03), 2)
                    changes.append(f"ShadowTP1[{sig_tf}]({','.join(s_strats[:2])})")
                    resolved = True
                    break

            if not resolved and len(fwd_bars) < 16:
                still_pending.append(sig)

        prof["pending_signals"] = still_pending[-15:]

        if changes or prof.get("last_bar_seen") != latest_bar_str:
            prof["last_bar_seen"] = latest_bar_str
            summary_note = "; ".join(changes) if changes else f"Recalibrated on bar {latest_bar_str[:16]} ({vol_regime})"
            _append_upgrade_note(prof, summary_note)

        profiles[sym] = prof
        if not defer_save:
            save_ticker_profiles(profiles)
        return prof


def format_ticker_learning_summary(prof: dict[str, Any]) -> str:
    """Compact one-line summary of a ticker's isolated learned DNA."""
    ver = prof.get("upgrade_version", 1)
    vol_r = prof.get("volatility_regime", "STANDARD")
    atrx = prof.get("atr_stop_mult", 1.25)
    clust = prof.get("min_cluster", 2)
    rsi = prof.get("min_rsi_reclaim", 42.0)
    volm = prof.get("vol_mult", 1.20)
    wins = int(prof.get("wins", 0))
    losses = int(prof.get("losses", 0))
    weights = prof.get("strategy_weights", {})
    top_s = max(weights.items(), key=lambda kv: kv[1]) if weights else ("None", 1.0)
    return (
        f"v{ver} [{vol_r}] | {wins}W/{losses}L | ATRx {atrx:.2f}, Cluster>={clust}, "
        f"RSI>={rsi:.0f}, Vol>={volm:.2f}x | Top Strat: {top_s[0]} ({top_s[1]:.2f}x)"
    )


def record_position_buy(
    ticker: str,
    entry_px: float,
    stop_px: float,
    target_px: float,
    shares: float,
    strategy: str = "Boroden 4H / Auto",
    bought_date: Optional[str] = None,
    stop_date: Optional[str] = None,
) -> None:
    """Persist a bought position into holdings.json and trade_journal.json without creating duplicate open rows."""
    sym = ticker.strip().upper().replace(".", "-")
    now_str = bought_date or datetime.now(ET_TZ).isoformat()

    with _STATE_LOCK:
        holdings = load_holdings()
        already_open = sym in holdings
        holdings[sym] = {
            "entry": round(float(entry_px), 4),
            "stop": round(float(stop_px), 4),
            "initial_risk": round(max(float(entry_px) - float(stop_px), 0.0), 4),
            "target": round(float(target_px), 4),
            "shares": round(float(shares), 4),
            "strategy": strategy,
            "bought_date": now_str,
            "stop_date": stop_date,
        }
        save_holdings(holdings)

        journal = load_journal()
        if already_open:
            for item in reversed(journal):
                if str(item.get("ticker", "")).upper() == sym and (item.get("exit") is None or item.get("exit") == ""):
                    item["entry"] = round(float(entry_px), 4)
                    item["stop"] = round(float(stop_px), 4)
                    item["target"] = round(float(target_px), 4)
                    item["shares"] = round(float(shares), 4)
                    item["strategy"] = strategy
                    item["bought_date"] = now_str[:10]
                    break
        else:
            journal.append({
                "ticker": sym,
                "bought_date": now_str[:10],
                "strategy": strategy,
                "entry": round(float(entry_px), 4),
                "stop": round(float(stop_px), 4),
                "target": round(float(target_px), 4),
                "shares": round(float(shares), 4),
                "exit": None,
                "closed_date": None,
            })
        save_journal(journal)


def record_position_close(
    ticker: str,
    exit_px: float,
    reason: str = "EXIT",
) -> Optional[float]:
    """Close an open position in holdings.json and trade_journal.json, immediately upgrading that ticker's learner."""
    sym = ticker.strip().upper().replace(".", "-")
    now_iso = datetime.now(ET_TZ).isoformat()
    now_date = now_iso[:10]

    with _STATE_LOCK:
        holdings = load_holdings()
        removed = holdings.pop(sym, None)
        save_holdings(holdings)

        journal = load_journal()
        closed_any = False
        pnl_total = 0.0
        for item in reversed(journal):
            if str(item.get("ticker", "")).upper() == sym and (item.get("exit") is None or item.get("exit") == ""):
                item["exit"] = round(float(exit_px), 4)
                item["closed_date"] = now_date
                entry_px = float(item.get("entry", exit_px))
                stop_px = float(item.get("stop", entry_px))
                target_px = float(item.get("target", entry_px * 1.03))
                shares = float(item.get("shares", 0.0))
                strat_lbl = str(item.get("strategy", ""))
                pnl = (exit_px - entry_px) * shares
                pnl_total += pnl
                t_key = f"LIVE:{item.get('bought_date', '')}:{now_iso}:{reason}"
                record_ticker_trade_outcome(
                    sym, entry_px, exit_px, stop_px, target_px, pnl, now_iso, reason, strat_lbl, trade_key=t_key
                )
                item["learned_recorded"] = True
                closed_any = True
                break

        save_journal(journal)
        if closed_any:
            return pnl_total
        if removed is not None:
            entry_px = float(removed.get("entry", exit_px))
            stop_px = float(removed.get("stop", entry_px))
            target_px = float(removed.get("target", entry_px * 1.03))
            shares = float(removed.get("shares", 0.0))
            strat_lbl = str(removed.get("strategy", ""))
            pnl = (exit_px - entry_px) * shares
            t_key = f"LIVE:{removed.get('bought_date', '')}:{now_iso}:{reason}"
            record_ticker_trade_outcome(
                sym, entry_px, exit_px, stop_px, target_px, pnl, now_iso, reason, strat_lbl, trade_key=t_key
            )
            return pnl
        return None


# ==============================================================================
# 3. MARKET DATA FETCHERS, TECHNICAL INDICATORS, CROSSOVERS & PATTERNS
# ==============================================================================
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance DataFrame to lowercase open/high/low/close/volume."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out["close"] = out["adj_close"]
    required = ["open", "high", "low", "close", "volume"]
    if not all(col in out.columns for col in required):
        return pd.DataFrame()
    return out[required].dropna(subset=["open", "high", "low", "close"]).sort_index()


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV DataFrame aligned to US market open (09:30 ET: Morning 4H & Afternoon 4H)."""
    if df is None or df.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx_et = idx.tz_localize(ET_TZ)
    else:
        idx_et = idx.tz_convert(ET_TZ)
    df_et = df.copy()
    df_et.index = idx_et
    return (
        df_et.resample("4h", origin="start_day", offset="9h30min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["close"])
    )


def fetch_yf_ohlcv(ticker: str, period: str = "6y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance for a single ticker."""
    sym = ticker.strip().upper().replace(".", "-")
    if not sym:
        return pd.DataFrame()
    resample_4h = interval.lower() == "4h"
    dl_interval = "1h" if resample_4h else interval
    try:
        raw = yf.download(
            tickers=sym,
            period=period,
            interval=dl_interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        norm = _normalize_ohlcv(raw)
        return _resample_to_4h(norm) if resample_4h else norm
    except Exception as exc:
        logger.warning("Download error for %s (%s): %s", sym, interval, exc)
        return pd.DataFrame()


def fetch_yf_batch_ohlcv(tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV DataFrames from Yahoo Finance for multiple tickers."""
    cleaned = sorted({t.strip().upper().replace(".", "-") for t in tickers if t})
    result: dict[str, pd.DataFrame] = {}
    if not cleaned:
        return result
    resample_4h = interval.lower() == "4h"
    dl_interval = "1h" if resample_4h else interval
    for i in range(0, len(cleaned), BATCH_SIZE):
        chunk = cleaned[i : i + BATCH_SIZE]
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=dl_interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for sym in chunk:
                    if sym in level0:
                        norm = _normalize_ohlcv(raw[sym])
                        if resample_4h:
                            norm = _resample_to_4h(norm)
                        if not norm.empty:
                            result[sym] = norm
            elif len(chunk) == 1:
                norm = _normalize_ohlcv(raw)
                if resample_4h:
                    norm = _resample_to_4h(norm)
                if not norm.empty:
                    result[chunk[0]] = norm
        except Exception as exc:
            logger.warning("Batch download error for %s...: %s", chunk[:3], exc)
    return result


def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest market prices from Yahoo Finance for open positions."""
    prices: dict[str, float] = {}
    batch = fetch_yf_batch_ohlcv(tickers, period="5d", interval="1d")
    for sym, df in batch.items():
        if not df.empty and "close" in df.columns:
            s = df["close"].dropna()
            if not s.empty:
                prices[sym] = float(s.iloc[-1])
    return prices


def get_sp500_and_nasdaq100_tickers() -> list[str]:
    """Fetch S&P 500 and Nasdaq 100 tickers from Wikipedia, merged and deduplicated."""
    headers = {"User-Agent": "SandeepPickBot/1.0 (contact@local)"}
    tickers: set[str] = set()
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        sp500_tables = pd.read_html(io.StringIO(resp.text))
        sp500_col = "Symbol" if "Symbol" in sp500_tables[0].columns else sp500_tables[0].columns[0]
        for raw_t in sp500_tables[0][sp500_col].dropna():
            sym = str(raw_t).strip().upper().replace(".", "-")
            if re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", sym):
                tickers.add(sym)
    except Exception as exc:
        logger.warning("Could not fetch S&P 500 list dynamically: %s", exc)
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        ndx_tables = pd.read_html(io.StringIO(resp.text))
        for tbl in ndx_tables:
            for col in ("Ticker", "Symbol"):
                if col in tbl.columns:
                    for raw_t in tbl[col].dropna():
                        sym = str(raw_t).strip().upper().replace(".", "-")
                        if re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", sym):
                            tickers.add(sym)
                    break
    except Exception as exc:
        logger.warning("Could not fetch Nasdaq-100 list dynamically: %s", exc)
    if not tickers:
        tickers = {
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
            "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC",
            "JPM", "V", "UNH", "XOM", "MA", "PG", "HD", "LLY",
        }
    return sorted(tickers)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """Anchored VWAP starting from anchor_idx."""
    vwap = pd.Series(np.nan, index=df.index, dtype=float)
    if df.empty or anchor_idx < 0 or anchor_idx >= len(df):
        return vwap
    sub = df.iloc[anchor_idx:]
    typical_price = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    cum_pv = (typical_price * sub["volume"]).cumsum()
    cum_vol = sub["volume"].cumsum().replace(0, np.nan)
    vwap.iloc[anchor_idx:] = (cum_pv / cum_vol).values
    return vwap


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Identify local pivot highs and pivot lows."""
    out = df.copy()
    out["is_swing_low"] = False
    out["is_swing_high"] = False
    lows = out["low"].values
    highs = out["high"].values
    n = len(out)
    for i in range(lookback, n - lookback):
        if lows[i] == np.min(lows[i - lookback : i + lookback + 1]):
            out.iat[i, out.columns.get_loc("is_swing_low")] = True
        if highs[i] == np.max(highs[i - lookback : i + lookback + 1]):
            out.iat[i, out.columns.get_loc("is_swing_high")] = True
    return out


def compute_1h_swing_low_series(
    df_1h: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    fallback_lookback: int = 20,
) -> pd.Series:
    """Compute the latest confirmed 1-hour swing low at each 1H bar (no lookahead bias)."""
    if df_1h is None or df_1h.empty or "low" not in df_1h.columns:
        return pd.Series(dtype=float)

    lows = df_1h["low"].to_numpy(dtype=float)
    n = len(lows)
    confirmed_sl = np.full(n, np.nan)

    for confirm_idx in range(left + right, n):
        center_idx = confirm_idx - right
        window = lows[center_idx - left : center_idx + right + 1]
        if lows[center_idx] == np.min(window):
            confirmed_sl[confirm_idx] = lows[center_idx]

    series = pd.Series(confirmed_sl, index=df_1h.index, dtype=float).ffill()
    rolling_fallback = df_1h["low"].rolling(fallback_lookback, min_periods=1).min()
    return series.fillna(rolling_fallback)


def find_latest_swing_low_before_entry(
    ohlcv_df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    left: int = 2,
    right: int = 1,
    lookback: int = 15,
) -> tuple[float, Any]:
    """Find the latest confirmed swing low at or prior to `entry_idx` below `entry_price`."""
    if ohlcv_df is None or ohlcv_df.empty:
        return round(entry_price * 0.97, 2), None

    end_pos = min(max(entry_idx, 0), len(ohlcv_df) - 1)
    start_pos = max(0, end_pos - lookback)
    sub = ohlcv_df.iloc[start_pos : end_pos + 1]
    lows = sub["low"].to_numpy(dtype=float)
    n = len(lows)

    for i in range(n - 1 - right, left - 1, -1):
        window = lows[i - left : i + right + 1]
        if lows[i] == np.min(window) and lows[i] < entry_price:
            return float(lows[i]), sub.index[i]

    valid_lows = [(lows[i], sub.index[i]) for i in range(n) if lows[i] < entry_price]
    if valid_lows:
        min_low, min_ts = min(valid_lows, key=lambda item: item[0])
        return float(min_low), min_ts

    return round(entry_price * 0.97, 2), ohlcv_df.index[end_pos]


def detect_butterfly_harmonic(df: pd.DataFrame, pivot_len: int = 3, tol: float = 0.08) -> Optional[str]:
    """Detect a bullish XABCD Butterfly harmonic pattern forming near Point D."""
    if df is None or len(df) < 35:
        return None

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    close_now = float(df["close"].iloc[-1])
    if np.isnan(close_now) or close_now <= 0:
        return None
    n = len(df)

    pivots: list[tuple[int, str, float]] = []
    for i in range(pivot_len, n - pivot_len):
        win_h = highs[i - pivot_len : i + pivot_len + 1]
        win_l = lows[i - pivot_len : i + pivot_len + 1]
        if np.isnan(win_h).any() or np.isnan(win_l).any():
            continue
        if highs[i] == np.max(win_h):
            pivots.append((i, "H", float(highs[i])))
        elif lows[i] == np.min(win_l):
            pivots.append((i, "L", float(lows[i])))

    cleaned: list[tuple[int, str, float]] = []
    for idx, ptype, val in pivots:
        if not cleaned or cleaned[-1][1] != ptype:
            cleaned.append((idx, ptype, val))
        else:
            _, _, prev_val = cleaned[-1]
            if (ptype == "H" and val > prev_val) or (ptype == "L" and val < prev_val):
                cleaned[-1] = (idx, ptype, val)

    if len(cleaned) < 4:
        return None

    x_p, a_p, b_p, c_p = cleaned[-4], cleaned[-3], cleaned[-2], cleaned[-1]
    if not (x_p[1] == "L" and a_p[1] == "H" and b_p[1] == "L" and c_p[1] == "H"):
        return None

    X, A, B, C = x_p[2], a_p[2], b_p[2], c_p[2]
    xa, ab, bc = A - X, A - B, C - B
    if xa <= 0 or ab <= 0 or bc <= 0:
        return None

    ab_xa = ab / xa
    bc_ab = bc / ab
    xd_xa = (A - close_now) / xa

    if (
        abs(ab_xa - 0.786) <= tol
        and (0.382 - tol) <= bc_ab <= (0.886 + tol)
        and (1.272 - tol) <= xd_xa <= (1.618 + tol)
    ):
        return f"🦋 Bullish Butterfly Harmonic near Point D ({xd_xa:.3f} XA extension)"
    return None


def detect_ema_crossovers(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect confirmed and imminent bullish EMA crossovers (9/21, 21/50, 50/200 Golden Cross)."""
    if df is None or len(df) < 55:
        return []

    close = df["close"]
    cur_close = float(close.iloc[-1])
    if np.isnan(cur_close) or cur_close <= 0:
        return []

    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)

    hits: list[dict[str, Any]] = []

    # 1. EMA 9 / 21 Bullish Crossover (Confirmed or Imminent)
    prev_gap_9_21 = float(ema21.iloc[-2] - ema9.iloc[-2])
    curr_gap_9_21 = float(ema21.iloc[-1] - ema9.iloc[-1])
    gap_pct_9_21 = (curr_gap_9_21 / cur_close) * 100.0
    if prev_gap_9_21 >= 0 and curr_gap_9_21 < 0:
        hits.append({
            "pair": "9/21",
            "status": "CONFIRMED",
            "label": "⚡ EMA 9/21 Bullish Crossover (Confirmed)",
        })
    elif 0 < gap_pct_9_21 <= 0.45 and curr_gap_9_21 < prev_gap_9_21 and cur_close > float(ema9.iloc[-1]):
        hits.append({
            "pair": "9/21",
            "status": "IMMINENT",
            "label": f"⏳ EMA 9/21 About to Cross (gap {gap_pct_9_21:.2f}%)",
        })

    # 2. EMA 21 / 50 Bullish Crossover (Confirmed or Imminent)
    prev_gap_21_50 = float(ema50.iloc[-2] - ema21.iloc[-2])
    curr_gap_21_50 = float(ema50.iloc[-1] - ema21.iloc[-1])
    gap_pct_21_50 = (curr_gap_21_50 / cur_close) * 100.0
    if prev_gap_21_50 >= 0 and curr_gap_21_50 < 0:
        hits.append({
            "pair": "21/50",
            "status": "CONFIRMED",
            "label": "⚡ EMA 21/50 Bullish Crossover (Confirmed)",
        })
    elif 0 < gap_pct_21_50 <= 0.60 and curr_gap_21_50 < prev_gap_21_50 and cur_close > float(ema21.iloc[-1]):
        hits.append({
            "pair": "21/50",
            "status": "IMMINENT",
            "label": f"⏳ EMA 21/50 About to Cross (gap {gap_pct_21_50:.2f}%)",
        })

    # 3. EMA 50 / 200 Golden Cross (Confirmed or Imminent)
    if len(df) >= 200:
        prev_gap_50_200 = float(ema200.iloc[-2] - ema50.iloc[-2])
        curr_gap_50_200 = float(ema200.iloc[-1] - ema50.iloc[-1])
        gap_pct_50_200 = (curr_gap_50_200 / cur_close) * 100.0
        if prev_gap_50_200 >= 0 and curr_gap_50_200 < 0:
            hits.append({
                "pair": "50/200",
                "status": "CONFIRMED",
                "label": "🌟 EMA 50/200 Golden Cross (Confirmed)",
            })
        elif 0 < gap_pct_50_200 <= 0.80 and curr_gap_50_200 < prev_gap_50_200 and cur_close > float(ema50.iloc[-1]):
            hits.append({
                "pair": "50/200",
                "status": "IMMINENT",
                "label": f"⏳ EMA 50/200 Golden Cross Approaching (gap {gap_pct_50_200:.2f}%)",
            })

    return hits


def detect_chart_patterns(df: pd.DataFrame) -> list[dict[str, str]]:
    """Detect active bullish chart & harmonic patterns (Bull Flag, Falling Wedge, Cup & Handle, Double Bottom, Butterfly)."""
    if df is None or len(df) < 50:
        return []

    patterns: list[dict[str, str]] = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    cur_close = float(close.iloc[-1])
    if np.isnan(cur_close) or cur_close <= 0:
        return []

    # 1. Bull Flag
    pole_base = float(close.iloc[-16])
    pole_top = float(close.iloc[-6])
    flag_high = float(high.iloc[-6:-1].max())
    if not np.isnan(pole_base) and not np.isnan(flag_high) and pole_base > 0 and flag_high > 0:
        pole_return = (pole_top - pole_base) / pole_base
        flag_pullback = (flag_high - cur_close) / flag_high
        vol_recent = float(vol.iloc[-5:].mean())
        vol_pole = float(vol.iloc[-15:-5].mean())
        if pole_return >= 0.06 and 0.01 <= flag_pullback <= 0.045 and vol_pole > 0 and vol_recent < vol_pole * 0.85:
            patterns.append({
                "key": "FLAG",
                "name": "Bull Flag",
                "label": f"🚩 Bull Flag (pole +{pole_return*100:.1f}%, tight pullback -{flag_pullback*100:.1f}%)",
            })

    # 2. Falling Wedge
    h_tail = high.iloc[-20:].to_numpy(dtype=float)
    l_tail = low.iloc[-20:].to_numpy(dtype=float)
    if not np.isnan(h_tail).any() and not np.isnan(l_tail).any():
        idx = np.arange(20)
        high_slope = float(np.polyfit(idx, h_tail, 1)[0])
        low_slope = float(np.polyfit(idx, l_tail, 1)[0])
        if high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope) * 1.25:
            spread_start = float(h_tail[0] - l_tail[0])
            spread_now = float(h_tail[-1] - l_tail[-1])
            if spread_start > 0 and spread_now < spread_start * 0.65:
                patterns.append({
                    "key": "WEDGE",
                    "name": "Falling Wedge",
                    "label": "📐 Falling Wedge (converging downward channel compression)",
                })

    # 3. Cup & Handle
    window = close.iloc[-50:]
    left_rim = float(window.iloc[:10].max())
    cup_bottom = float(window.iloc[12:38].min())
    right_rim = float(window.iloc[38:45].max())
    handle_low = float(window.iloc[45:].min())
    if not any(np.isnan(v) for v in (left_rim, cup_bottom, right_rim, handle_low)) and left_rim > 0 and right_rim > 0:
        cup_depth = (left_rim - cup_bottom) / left_rim
        rim_diff = abs(left_rim - right_rim) / left_rim
        handle_drop = (right_rim - handle_low) / right_rim
        if (
            0.08 <= cup_depth <= 0.30
            and rim_diff <= 0.03
            and 0.01 <= handle_drop <= 0.08
            and cur_close > handle_low
            and cur_close >= right_rim * 0.94
        ):
            patterns.append({
                "key": "CUP",
                "name": "Cup & Handle",
                "label": f"☕ Cup & Handle (cup -{cup_depth*100:.1f}%, handle -{handle_drop*100:.1f}%, rim ${right_rim:.2f})",
            })

    # 4. Double Bottom (W-Pattern)
    lows_35 = low.iloc[-35:].to_numpy(dtype=float)
    highs_35 = high.iloc[-35:].to_numpy(dtype=float)
    if not np.isnan(lows_35).any() and not np.isnan(highs_35).any():
        first_half_idx = int(np.argmin(lows_35[:16]))
        second_half_idx = 18 + int(np.argmin(lows_35[18:33]))
        low1 = float(lows_35[first_half_idx])
        low2 = float(lows_35[second_half_idx])
        if low1 > 0 and abs(low1 - low2) / low1 <= 0.018:
            neckline = float(np.max(highs_35[first_half_idx : second_half_idx + 1]))
            mid_bounce = (neckline - min(low1, low2)) / min(low1, low2)
            if mid_bounce >= 0.04 and cur_close >= low2 * 1.01 and cur_close <= neckline * 1.02:
                patterns.append({
                    "key": "DOUBLE",
                    "name": "Double Bottom",
                    "label": f"🇼 Double Bottom (support ${min(low1, low2):.2f}, neckline ${neckline:.2f})",
                })

    # 5. Bullish Butterfly Harmonic
    butterfly_note = detect_butterfly_harmonic(df)
    if butterfly_note:
        patterns.append({
            "key": "BUTTERFLY",
            "name": "Bullish Butterfly Harmonic",
            "label": butterfly_note,
        })

    return patterns


def detect_heads_up_setups(df: pd.DataFrame) -> list[str]:
    """Low-priority 'Heads-Up Watch' detector combining EMA crossovers and chart patterns."""
    if df is None or len(df) < 55:
        return []
    notes: list[str] = []
    for cx in detect_ema_crossovers(df):
        notes.append(cx["label"])
    for pat in detect_chart_patterns(df):
        notes.append(pat["label"])
    return notes


# ==============================================================================
# 4. REAL CANDLESTICK CHART IMAGE GENERATOR
# ==============================================================================
def generate_trade_candlestick_chart(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    tp1: float,
    tp2: Optional[float] = None,
    tp3: Optional[float] = None,
    current_price: Optional[float] = None,
    timeframe_label: str = "4H",
    decision_label: str = "TRADE PLAN",
    max_bars: int = 55,
    entry_time: Any = None,
    stop_time: Any = None,
) -> Optional[io.BytesIO]:
    """Render a dark-themed real OHLC candlestick chart with Entry, Swing-Low Stop, and Exit boxes."""
    if ohlcv_df is None or ohlcv_df.empty:
        return None

    plot_df = ohlcv_df.tail(max_bars).copy()
    if len(plot_df) < 5:
        return None

    plot_df["ema21"] = compute_ema(plot_df["close"], 21)
    n = len(plot_df)
    x = np.arange(n)

    opens = plot_df["open"].to_numpy(dtype=float)
    highs = plot_df["high"].to_numpy(dtype=float)
    lows = plot_df["low"].to_numpy(dtype=float)
    closes = plot_df["close"].to_numpy(dtype=float)
    vols = plot_df["volume"].to_numpy(dtype=float)
    ema21 = plot_df["ema21"].to_numpy(dtype=float)

    cur_px = float(current_price) if (current_price is not None and current_price > 0) else float(closes[-1])
    entry_px = float(entry_price) if entry_price > 0 else cur_px
    stop_px = float(stop_loss) if stop_loss > 0 else round(entry_px * 0.97, 2)
    tp1_px = float(tp1) if tp1 > 0 else round(entry_px * 1.03, 2)

    dt_index = _to_naive_et_index(plot_df.index)
    date_fmt = "%b %d %H:%M" if timeframe_label.upper() in ("1H", "4H") else "%b %d"

    entry_idx = n - 1
    if entry_time is not None:
        aligned_entry = _to_naive_et_timestamp(entry_time)
        if aligned_entry is not None:
            entry_idx = int(np.argmin(np.abs(dt_index - np.datetime64(aligned_entry))))

    stop_idx = None
    if stop_time is not None:
        aligned_stop = _to_naive_et_timestamp(stop_time)
        if aligned_stop is not None:
            stop_idx = int(np.argmin(np.abs(dt_index - np.datetime64(aligned_stop))))
    if stop_idx is None:
        _, auto_stop_ts = find_latest_swing_low_before_entry(plot_df, entry_idx, entry_px)
        aligned_auto = _to_naive_et_timestamp(auto_stop_ts)
        if aligned_auto is not None:
            stop_idx = int(np.argmin(np.abs(dt_index - np.datetime64(aligned_auto))))

    fig = Figure(figsize=(11.5, 6.8))
    FigureCanvasAgg(fig)
    ax_price, ax_vol = fig.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [4.2, 1.0]},
        sharex=True,
    )
    bg_color = "#0B1220"
    panel_color = "#0F172A"
    grid_color = "#1E293B"
    text_color = "#E2E8F0"

    fig.patch.set_facecolor(bg_color)
    for ax in (ax_price, ax_vol):
        ax.set_facecolor(panel_color)
        ax.tick_params(colors=text_color, labelsize=8.5)
        ax.grid(True, color=grid_color, linestyle="--", linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    candle_width = 0.62
    bull_color = "#22C55E"
    bear_color = "#EF4444"

    for i in range(n):
        is_bull = closes[i] >= opens[i]
        c_col = bull_color if is_bull else bear_color
        ax_price.plot([x[i], x[i]], [lows[i], highs[i]], color=c_col, linewidth=1.15, zorder=3)
        body_low = min(opens[i], closes[i])
        body_h = max(abs(closes[i] - opens[i]), (highs[i] - lows[i]) * 0.02 + 1e-4)
        ax_price.add_patch(
            Rectangle(
                (x[i] - candle_width / 2.0, body_low),
                candle_width,
                body_h,
                facecolor=c_col,
                edgecolor=c_col,
                linewidth=0.8,
                zorder=4,
            )
        )

    ax_price.plot(x, ema21, color="#38BDF8", linewidth=1.25, alpha=0.85, label="EMA 21", zorder=5)

    x_min, x_max = -0.8, n + max(11.0, n * 0.28)
    zone_end = n + 0.35
    if stop_px < entry_px:
        ax_price.fill_between([entry_idx, zone_end], stop_px, entry_px, color="#EF4444", alpha=0.16, zorder=1)
    if tp1_px > entry_px:
        ax_price.fill_between([entry_idx, zone_end], entry_px, tp1_px, color="#22C55E", alpha=0.14, zorder=1)

    risk_pct = ((entry_px - stop_px) / entry_px * 100.0) if entry_px > 0 else 0.0
    tp1_pct = ((tp1_px - entry_px) / entry_px * 100.0) if entry_px > 0 else 0.0
    rr_ratio = (tp1_px - entry_px) / max(entry_px - stop_px, 1e-6) if (entry_px > stop_px and tp1_px > entry_px) else 0.0

    def _draw_level(
        price_val: float,
        label_txt: str,
        line_col: str,
        linestyle: str = "--",
        lw: float = 1.6,
        start_x: int = entry_idx,
    ) -> None:
        ax_price.plot([start_x, zone_end], [price_val, price_val], color=line_col, linestyle=linestyle, linewidth=lw, zorder=6)
        ax_price.text(
            zone_end + 0.15,
            price_val,
            label_txt,
            color="#FFFFFF",
            fontsize=8.0,
            fontweight="bold",
            va="center",
            ha="left",
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=line_col, edgecolor="none", alpha=0.92),
            zorder=8,
        )

    stop_start_x = stop_idx if (stop_idx is not None and stop_idx <= entry_idx) else entry_idx
    stop_dt_note = f" ({pd.Timestamp(dt_index[stop_idx]).strftime('%b %d')} Low)" if stop_idx is not None else ""

    _draw_level(entry_px, f"ENTRY ${entry_px:.2f}", "#2563EB", linestyle="-", lw=1.8, start_x=entry_idx)
    _draw_level(stop_px, f"STOP LOSS ${stop_px:.2f}{stop_dt_note} (-{risk_pct:.1f}%)", "#DC2626", linestyle="--", lw=1.7, start_x=stop_start_x)
    _draw_level(tp1_px, f"EXIT TP1 ${tp1_px:.2f} (+{tp1_pct:.1f}%)", "#16A34A", linestyle="--", lw=1.7, start_x=entry_idx)

    if tp2 is not None and not np.isnan(tp2) and tp2 > tp1_px:
        tp2_pct = (tp2 - entry_px) / entry_px * 100.0
        _draw_level(float(tp2), f"EXIT TP2 ${tp2:.2f} (+{tp2_pct:.1f}%)", "#0D9488", linestyle=":", lw=1.4, start_x=entry_idx)

    if tp3 is not None and not np.isnan(tp3) and (tp2 is None or tp3 > tp2):
        tp3_pct = (tp3 - entry_px) / entry_px * 100.0
        _draw_level(float(tp3), f"EXIT TP3 ${tp3:.2f} (+{tp3_pct:.1f}%)", "#D97706", linestyle=":", lw=1.3, start_x=entry_idx)

    entry_dt_str = pd.Timestamp(dt_index[entry_idx]).strftime("%b %d")
    ax_price.axvline(entry_idx, color="#60A5FA", linestyle=":", linewidth=1.3, alpha=0.9, zorder=2)
    ax_price.scatter(
        [entry_idx],
        [entry_px],
        marker="^",
        color="#22C55E",
        edgecolor="#FFFFFF",
        linewidth=1.2,
        s=130,
        zorder=10,
        label=f"Entry ({entry_dt_str} @ ${entry_px:.2f})",
    )
    if stop_idx is not None:
        ax_price.scatter(
            [stop_idx],
            [stop_px],
            marker="v",
            color="#EF4444",
            edgecolor="#FFFFFF",
            linewidth=1.1,
            s=110,
            zorder=10,
            label=f"Swing Low Stop ({pd.Timestamp(dt_index[stop_idx]).strftime('%b %d')} @ ${stop_px:.2f})",
        )
    if entry_idx != n - 1:
        ax_price.scatter(
            [n - 1],
            [cur_px],
            color="#FACC15",
            edgecolor="#0F172A",
            s=65,
            zorder=10,
            label=f"Current Price (${cur_px:.2f})",
        )

    vol_colors = [bull_color if closes[i] >= opens[i] else bear_color for i in range(n)]
    ax_vol.bar(x, vols, width=0.6, color=vol_colors, alpha=0.65, zorder=3)
    ax_vol.axvline(entry_idx, color="#60A5FA", linestyle=":", linewidth=1.3, alpha=0.9, zorder=2)
    ax_vol.set_ylabel("Volume", color=text_color, fontsize=8.5)
    ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    tick_step = max(1, n // 7)
    extra_ticks = [entry_idx, n - 1] + ([stop_idx] if stop_idx is not None else [])
    tick_positions = sorted(set(list(range(0, n, tick_step)) + extra_ticks))
    ax_vol.set_xticks(tick_positions)
    ax_vol.set_xticklabels([pd.Timestamp(dt_index[idx]).strftime(date_fmt) for idx in tick_positions], rotation=20, ha="right", color=text_color, fontsize=8)

    visible_levels = [float(np.nanmin(lows)), float(np.nanmax(highs)), entry_px, stop_px, tp1_px, cur_px]
    if tp2 is not None and not np.isnan(tp2) and tp2 > 0:
        visible_levels.append(float(tp2))
    visible_levels = [v for v in visible_levels if not np.isnan(v)]
    y_min = min(visible_levels)
    y_max = max(visible_levels)
    y_pad = max((y_max - y_min) * 0.09, cur_px * 0.01)
    ax_price.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_price.set_xlim(x_min, x_max)
    ax_price.set_ylabel("Price ($)", color=text_color, fontsize=9)

    rr_str = f" | R:R (TP1): {rr_ratio:.2f}x" if rr_ratio > 0 else ""
    ax_price.set_title(
        f"{ticker.upper()} ({timeframe_label} Candles) — {decision_label}{rr_str}",
        color="#F8FAFC",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax_price.legend(loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor=text_color, fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ==============================================================================
# 5. BORODEN 4H FIBONACCI ENGINE, SPY CYCLES, SECTOR MONEY FLOW & SCORING
# ==============================================================================
def score_strategy_confluence(
    matched_strategies: list[str],
    quality_bonus: float = 0.0,
    ticker_profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any] | None:
    """
    Rates a ticker by the number of matched strategies, modulated by THAT ticker's
    learned strategy weights:
      - > 4 strategies : HIGH   (> 85: 86.0 - 99.0)
      - 3-4 strategies : MEDIUM (75 - 85: 75.0 - 85.0)
      - 1-2 strategies : LOW    (60 - 75: 60.0 - 74.9)
      - 0 strategies   : None   (Ignored)
    """
    unique_strats = list(dict.fromkeys(s.strip() for s in matched_strategies if s))
    count = len(unique_strats)

    if count == 0:
        return None

    bonus = max(0.0, min(1.0, float(quality_bonus)))
    avg_weight = 1.0
    if ticker_profile and ticker_profile.get("strategy_weights"):
        w_map = ticker_profile["strategy_weights"]
        weights = [float(w_map.get(s, 1.0)) for s in unique_strats]
        if weights:
            avg_weight = sum(weights) / len(weights)
    weight_factor = max(0.0, min(1.0, (avg_weight - 0.65) / 0.70))
    effective_bonus = max(0.0, min(1.0, 0.55 * bonus + 0.45 * weight_factor))

    if count > 4:
        rating = "HIGH"
        extra_strats = min(count - 5, 2)
        score = round(86.0 + (extra_strats * 3.5) + (effective_bonus * 6.0), 1)
        score = min(score, 99.0)
    elif count in (3, 4):
        rating = "MEDIUM"
        if count == 3:
            score = round(75.0 + (effective_bonus * 4.9), 1)
        else:
            score = round(80.0 + (effective_bonus * 5.0), 1)
    else:
        rating = "LOW"
        if count == 1:
            score = round(60.0 + (effective_bonus * 7.4), 1)
        else:
            score = round(67.5 + (effective_bonus * 7.4), 1)

    badge = {"HIGH": "🔥 HIGH", "MEDIUM": "⚡ MEDIUM", "LOW": "🔹 LOW"}[rating]
    return {
        "rating": rating,
        "badge": badge,
        "score": score,
        "matched_count": count,
        "matched_strategies": unique_strats,
        "avg_strategy_weight": round(avg_weight, 2),
    }


def detect_confirmed_pivots(df: pd.DataFrame, pivot_len: int) -> pd.DataFrame:
    """Detect confirmed pivot highs and pivot lows without lookahead bias."""
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)

    confirmed_ph = np.full(n, np.nan)
    confirmed_pl = np.full(n, np.nan)

    window = 2 * pivot_len + 1
    for confirm_idx in range(window - 1, n):
        center_idx = confirm_idx - pivot_len
        left = center_idx - pivot_len
        right = center_idx + pivot_len

        if highs[center_idx] == np.max(highs[left : right + 1]):
            confirmed_ph[confirm_idx] = highs[center_idx]

        if lows[center_idx] == np.min(lows[left : right + 1]):
            confirmed_pl[confirm_idx] = lows[center_idx]

    out = df.copy()
    out["confirmed_ph"] = confirmed_ph
    out["confirmed_pl"] = confirmed_pl
    return out


def count_fib_cluster(ref_price: float, levels: list[Optional[float]], tol_pct: float) -> int:
    """Count how many Fibonacci support/symmetry levels fall within tol_pct of ref_price."""
    if np.isnan(ref_price) or ref_price <= 0:
        return 0
    tol = ref_price * (tol_pct / 100.0)
    count = 0
    for lvl in levels:
        if lvl is not None and not np.isnan(lvl):
            if abs(lvl - ref_price) <= tol:
                count += 1
    return count


def _align_daily_trend_to_4h(df_4h: pd.DataFrame, daily_df: Optional[pd.DataFrame]) -> pd.Series:
    """Return boolean Series aligned to df_4h: True when the latest completed Daily close >= Daily EMA50."""
    if daily_df is None or daily_df.empty or len(daily_df) < 20:
        return pd.Series(True, index=df_4h.index)
    d = daily_df.copy().sort_index()
    d["daily_ema50"] = compute_ema(d["close"], 50)
    d["daily_bull"] = d["close"] >= d["daily_ema50"]

    idx_4h = _to_naive_et_index(df_4h.index)
    # Daily bars from yfinance are stamped at 00:00; shift to 16:00 ET market close to avoid same-day lookahead
    idx_d = _to_naive_et_index(d.index).normalize() + pd.Timedelta(hours=16)
    d_aligned = pd.DataFrame({"daily_bull": d["daily_bull"].values}, index=idx_d).sort_index()
    h4_frame = pd.DataFrame(index=idx_4h).sort_index()
    merged = pd.merge_asof(h4_frame, d_aligned, left_index=True, right_index=True, direction="backward")
    return pd.Series(merged["daily_bull"].fillna(True).astype(bool).values, index=df_4h.index)


def _align_1h_swing_low_to_4h(
    df_4h: pd.DataFrame,
    df_1h: Optional[pd.DataFrame],
    cfg: Boroden4HConfig,
) -> pd.Series:
    """Align the latest confirmed 1-hour swing low to each 4H bar."""
    if df_1h is None or df_1h.empty:
        return pd.Series(np.nan, index=df_4h.index, dtype=float)
    sl_1h = compute_1h_swing_low_series(
        df_1h,
        left=cfg.swing_1h_left,
        right=cfg.swing_1h_right,
        fallback_lookback=cfg.swing_1h_fallback_lookback,
    )
    if sl_1h.empty:
        return pd.Series(np.nan, index=df_4h.index, dtype=float)

    idx_4h = _to_naive_et_index(df_4h.index)
    idx_1h = _to_naive_et_index(sl_1h.index)
    h4_close_idx = idx_4h + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
    s1h_frame = pd.DataFrame({"swing_low_1h": sl_1h.values}, index=idx_1h).sort_index()
    h4_frame = pd.DataFrame(index=h4_close_idx).sort_index()
    merged = pd.merge_asof(h4_frame, s1h_frame, left_index=True, right_index=True, direction="backward")
    return pd.Series(merged["swing_low_1h"].to_numpy(dtype=float), index=df_4h.index)


def evaluate_boroden_4h_signals(
    df: pd.DataFrame,
    cfg: Optional[Boroden4HConfig] = None,
    daily_df: Optional[pd.DataFrame] = None,
    ticker: Optional[str] = None,
    df_1h: Optional[pd.DataFrame] = None,
    ticker_profile: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate bar-by-bar 4H Boroden Fibonacci cluster, symmetry, and ticker-specific learned thresholds."""
    if df is None or df.empty:
        return pd.DataFrame()
    cfg = cfg or Boroden4HConfig()

    data = detect_confirmed_pivots(df, cfg.pivot_len)
    data["rsi_14"] = compute_rsi(data["close"], cfg.rsi_period)
    data["ema_21_4h"] = compute_ema(data["close"], 21)
    data["atr_14"] = compute_atr(data, cfg.atr_period)
    data["vol_sma20"] = data["volume"].rolling(20, min_periods=1).mean()
    data["daily_trend_ok"] = _align_daily_trend_to_4h(data, daily_df)
    data["swing_low_1h"] = _align_1h_swing_low_to_4h(data, df_1h, cfg)

    learned = ticker_profile if ticker_profile is not None else (get_ticker_learning_profile(ticker, cfg) if ticker else None)

    eff_min_cluster = int(learned.get("min_cluster", cfg.min_cluster)) if learned else cfg.min_cluster
    eff_atr_mult = float(learned.get("atr_stop_mult", cfg.atr_stop_mult)) if learned else cfg.atr_stop_mult
    eff_rsi_level = float(learned.get("min_rsi_reclaim", cfg.rsi_reclaim_level)) if learned else cfg.rsi_reclaim_level
    eff_vol_mult = float(learned.get("vol_mult", cfg.vol_mult)) if learned else cfg.vol_mult
    eff_tp_scale = float(learned.get("tp_scale_mult", 1.0)) if learned else 1.0

    last_high: Optional[float] = None
    prev_high: Optional[float] = None
    last_low: Optional[float] = None
    prev_low: Optional[float] = None
    prev_corr_drop: Optional[float] = None

    n = len(data)
    signals = np.zeros(n, dtype=bool)
    cluster_counts = np.zeros(n, dtype=int)
    trend_allowed_arr = np.zeros(n, dtype=bool)
    is_downtrend_arr = np.zeros(n, dtype=bool)
    has_higher_low_arr = np.zeros(n, dtype=bool)
    valid_swing_arr = np.zeros(n, dtype=bool)
    in_fib_zone_arr = np.zeros(n, dtype=bool)
    bullish_candle_arr = np.zeros(n, dtype=bool)
    candle_body_ok_arr = np.zeros(n, dtype=bool)
    rsi_reclaim_arr = np.zeros(n, dtype=bool)
    vol_ok_arr = np.zeros(n, dtype=bool)

    last_high_arr = np.full(n, np.nan)
    last_low_arr = np.full(n, np.nan)
    prev_corr_arr = np.full(n, np.nan)
    stop_arr = np.full(n, np.nan)
    tp1_arr = np.full(n, np.nan)
    tp2_arr = np.full(n, np.nan)
    tp3_arr = np.full(n, np.nan)

    rsi_arr = data["rsi_14"].to_numpy(dtype=float)
    ema21_arr = data["ema_21_4h"].to_numpy(dtype=float)
    atr_arr = data["atr_14"].to_numpy(dtype=float)
    vol_arr = data["volume"].to_numpy(dtype=float)
    vol_sma_arr = data["vol_sma20"].to_numpy(dtype=float)
    daily_ok_arr = data["daily_trend_ok"].to_numpy(dtype=bool)
    sl_1h_arr = data["swing_low_1h"].to_numpy(dtype=float)
    live_bar_progress = _current_4h_session_progress()

    for i in range(n):
        row = data.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        ph_val = row["confirmed_ph"]
        pl_val = row["confirmed_pl"]

        if not np.isnan(ph_val):
            prev_high = last_high
            last_high = float(ph_val)
            if prev_high is not None and last_low is not None and prev_high > last_low:
                prev_corr_drop = prev_high - last_low

        if not np.isnan(pl_val):
            prev_low = last_low
            last_low = float(pl_val)
            if last_high is not None and last_high > last_low:
                prev_corr_drop = last_high - last_low

        if last_high is not None:
            last_high_arr[i] = last_high
        if last_low is not None:
            last_low_arr[i] = last_low
        if prev_corr_drop is not None:
            prev_corr_arr[i] = prev_corr_drop

        has_higher_high = (last_high is not None and prev_high is not None and last_high > prev_high)
        has_higher_low = (last_low is not None and prev_low is not None and last_low > prev_low)
        has_lower_high = (last_high is not None and prev_high is not None and last_high < prev_high)
        has_lower_low = (last_low is not None and prev_low is not None and last_low < prev_low)

        is_uptrend = has_higher_high and has_higher_low
        is_downtrend = has_lower_high and has_lower_low and not has_higher_low
        trend_allows_long = is_uptrend or has_higher_low

        is_downtrend_arr[i] = is_downtrend
        has_higher_low_arr[i] = has_higher_low
        trend_allowed_arr[i] = trend_allows_long

        valid_bull_swing = (last_low is not None and last_high is not None and last_high > last_low)
        valid_swing_arr[i] = valid_bull_swing
        swing_range = (last_high - last_low) if valid_bull_swing else np.nan

        fib_levels: list[Optional[float]] = []
        s100 = np.nan
        res_tp1 = res_tp2 = res_tp3 = np.nan

        if valid_bull_swing:
            r382 = last_high - swing_range * 0.382
            r500 = last_high - swing_range * 0.500
            r618 = last_high - swing_range * 0.618
            r786 = last_high - swing_range * 0.786
            e1272 = last_high - swing_range * 1.272
            e1618 = last_high - swing_range * 1.618
            s100 = (last_high - prev_corr_drop) if prev_corr_drop is not None else np.nan
            a1618 = (last_high - prev_corr_drop * 1.618) if prev_corr_drop is not None else np.nan
            fib_levels = [r382, r500, r618, r786, e1272, e1618, s100, a1618]

            res_tp1 = last_low + swing_range * (cfg.tp1_ratio * eff_tp_scale)
            res_tp2 = last_low + swing_range * (cfg.tp2_ratio * eff_tp_scale)
            res_tp3 = last_low + swing_range * (cfg.tp3_ratio * eff_tp_scale)

        cnt = count_fib_cluster(c, fib_levels, cfg.cluster_tol_pct)
        cluster_counts[i] = cnt

        daily_ok = bool(daily_ok_arr[i])
        prev_rsi = rsi_arr[i - 1] if i > 0 else rsi_arr[i]
        rsi_reclaim = (rsi_arr[i] >= eff_rsi_level) and (
            (prev_rsi < eff_rsi_level) or (c >= ema21_arr[i])
        )
        candle_body_ok = (c > o) and (c > l + (h - l) * 0.4)
        bullish_candle = candle_body_ok and rsi_reclaim

        bar_pace = live_bar_progress if (i == n - 1) else 1.0
        vol_ok = vol_arr[i] >= (vol_sma_arr[i] * eff_vol_mult * bar_pace)

        in_fib_support_zone = valid_bull_swing and (cnt >= eff_min_cluster) and (c < last_high)
        long_sig = bool(daily_ok and trend_allows_long and in_fib_support_zone and bullish_candle and vol_ok)

        in_fib_zone_arr[i] = in_fib_support_zone
        candle_body_ok_arr[i] = candle_body_ok
        rsi_reclaim_arr[i] = rsi_reclaim
        bullish_candle_arr[i] = bullish_candle
        vol_ok_arr[i] = vol_ok
        signals[i] = long_sig

        sl_1h = sl_1h_arr[i]
        sw_4h, _ = find_latest_swing_low_before_entry(data, i, c, left=2, right=1, lookback=15)
        atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else (c * 0.01)
        atr_floor_stop = c - (eff_atr_mult * atr_val)

        if not np.isnan(sl_1h) and 0 < sl_1h < c:
            raw_stop = sl_1h * (1.0 - cfg.stop_buffer_pct / 100.0)
            stop_arr[i] = min(raw_stop, atr_floor_stop) if (learned and learned.get("losses", 0) > 0) else raw_stop
        elif 0 < sw_4h < c:
            raw_stop = sw_4h * (1.0 - cfg.stop_buffer_pct / 100.0)
            stop_arr[i] = min(raw_stop, atr_floor_stop) if (learned and learned.get("losses", 0) > 0) else raw_stop
        else:
            anchor_support = min(c, s100) if not np.isnan(s100) else c
            pct_stop = anchor_support * (1.0 - max(cfg.stop_buffer_pct, 0.20) / 100.0)
            stop_arr[i] = min(pct_stop, atr_floor_stop)

        t1 = res_tp1 if (not np.isnan(res_tp1) and res_tp1 > c) else c * 1.02
        t2 = res_tp2 if (not np.isnan(res_tp2) and res_tp2 > t1) else t1 * 1.02
        t3 = res_tp3 if (not np.isnan(res_tp3) and res_tp3 > t2) else t2 * 1.02
        tp1_arr[i], tp2_arr[i], tp3_arr[i] = t1, t2, t3

    data["fib_cluster_count"] = cluster_counts
    data["eff_min_cluster_4h"] = eff_min_cluster
    data["eff_rsi_level_4h"] = eff_rsi_level
    data["eff_vol_mult_4h"] = eff_vol_mult
    data["is_downtrend_4h"] = is_downtrend_arr
    data["has_higher_low_4h"] = has_higher_low_arr
    data["trend_allows_long_4h"] = trend_allowed_arr
    data["valid_bull_swing_4h"] = valid_swing_arr
    data["in_fib_zone_4h"] = in_fib_zone_arr
    data["candle_body_ok_4h"] = candle_body_ok_arr
    data["rsi_reclaim_4h"] = rsi_reclaim_arr
    data["bullish_candle_4h"] = bullish_candle_arr
    data["vol_ok_4h"] = vol_ok_arr
    data["last_swing_high_4h"] = last_high_arr
    data["last_swing_low_4h"] = last_low_arr
    data["prev_corr_drop_4h"] = prev_corr_arr
    data["boroden_4h_stop"] = stop_arr
    data["boroden_4h_tp1"] = tp1_arr
    data["boroden_4h_tp2"] = tp2_arr
    data["boroden_4h_tp3"] = tp3_arr
    data["boroden_4h_long_signal"] = signals
    return data


def run_boroden_4h_backtest(
    df: pd.DataFrame,
    cfg: Optional[Boroden4HConfig] = None,
    daily_df: Optional[pd.DataFrame] = None,
    ticker: Optional[str] = None,
    learn_mistakes: bool = True,
    df_1h: Optional[pd.DataFrame] = None,
    profiles_cache: Optional[dict[str, dict[str, Any]]] = None,
    defer_save: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a bar-by-bar long-only backtest on 4H OHLCV data, deduplicating historical trades per ticker."""
    cfg = cfg or Boroden4HConfig()
    prof = get_ticker_learning_profile(ticker, cfg, profiles_cache=profiles_cache, defer_save=defer_save) if ticker else None
    data = evaluate_boroden_4h_signals(df, cfg, daily_df=daily_df, ticker=ticker, df_1h=df_1h, ticker_profile=prof)
    if data.empty:
        return data, pd.DataFrame()

    equity = cfg.initial_capital
    in_position = False
    entry_price = np.nan
    stop_loss = np.nan
    targets = [np.nan, np.nan, np.nan]
    initial_qty = 0.0
    remaining_qty = 0.0
    tp_hit = [False, False, False]
    entry_time = None
    bars_held = 0
    trades: list[dict[str, Any]] = []

    for idx, row in data.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

        if in_position and remaining_qty > 0:
            bars_held += 1
            if l <= stop_loss:
                exit_px = min(o, stop_loss) if o < stop_loss else stop_loss
                pnl = (exit_px - entry_price) * remaining_qty
                equity += pnl
                reason = "BREAKEVEN_STOP" if (tp_hit[0] and abs(stop_loss - entry_price) < 1e-6) else "STOP_SWING_LOW"
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": idx,
                    "exit_reason": reason,
                    "entry_price": entry_price,
                    "exit_price": exit_px,
                    "qty": remaining_qty,
                    "pnl": pnl,
                })
                if learn_mistakes and ticker and pnl != 0 and reason != "BREAKEVEN_STOP":
                    bt_key = f"BT:{entry_time}:{idx}:{reason}"
                    record_ticker_trade_outcome(
                        ticker,
                        entry_price,
                        exit_px,
                        stop_loss,
                        targets[0],
                        pnl,
                        str(idx),
                        reason,
                        "4H Boroden Fib Cluster",
                        trade_key=bt_key,
                        profiles_cache=profiles_cache,
                        defer_save=defer_save,
                    )
                in_position = False
                remaining_qty = 0.0
                bars_held = 0
            else:
                scales = [(0, 0.40, "TP1"), (1, 0.35, "TP2"), (2, 0.25, "TP3")]
                for t_idx, alloc_share, label in scales:
                    if not tp_hit[t_idx] and not np.isnan(targets[t_idx]) and h >= targets[t_idx]:
                        exit_px = max(o, targets[t_idx])
                        scale_qty = min(remaining_qty, initial_qty * alloc_share)
                        pnl = (exit_px - entry_price) * scale_qty
                        equity += pnl
                        remaining_qty -= scale_qty
                        tp_hit[t_idx] = True
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": idx,
                            "exit_reason": label,
                            "entry_price": entry_price,
                            "exit_price": exit_px,
                            "qty": scale_qty,
                            "pnl": pnl,
                        })
                        if learn_mistakes and ticker and t_idx == 0:
                            bt_key = f"BT:{entry_time}:{idx}:{label}"
                            record_ticker_trade_outcome(
                                ticker,
                                entry_price,
                                exit_px,
                                stop_loss,
                                targets[0],
                                pnl,
                                str(idx),
                                label,
                                "4H Boroden Fib Cluster",
                                trade_key=bt_key,
                                profiles_cache=profiles_cache,
                                defer_save=defer_save,
                            )
                        if t_idx == 0 and cfg.breakeven_after_tp1 and remaining_qty > 0:
                            stop_loss = max(stop_loss, entry_price)

                if remaining_qty <= 1e-9:
                    in_position = False
                    remaining_qty = 0.0
                    bars_held = 0
                elif bars_held >= cfg.max_hold_bars:
                    pnl = (c - entry_price) * remaining_qty
                    equity += pnl
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": idx,
                        "exit_reason": "TIME_STOP",
                        "entry_price": entry_price,
                        "exit_price": c,
                        "qty": remaining_qty,
                        "pnl": pnl,
                    })
                    if learn_mistakes and ticker and pnl != 0:
                        bt_key = f"BT:{entry_time}:{idx}:TIME_STOP"
                        record_ticker_trade_outcome(
                            ticker,
                            entry_price,
                            c,
                            stop_loss,
                            targets[0],
                            pnl,
                            str(idx),
                            "TIME_STOP",
                            "4H Boroden Fib Cluster",
                            trade_key=bt_key,
                            profiles_cache=profiles_cache,
                            defer_save=defer_save,
                        )
                    in_position = False
                    remaining_qty = 0.0
                    bars_held = 0

        if (not in_position) and bool(row["boroden_4h_long_signal"]):
            alloc_cash = equity * cfg.position_pct
            initial_qty = alloc_cash / c if c > 0 else 0.0
            if initial_qty > 0:
                entry_price = c
                stop_loss = float(row["boroden_4h_stop"])
                targets = [
                    float(row["boroden_4h_tp1"]),
                    float(row["boroden_4h_tp2"]),
                    float(row["boroden_4h_tp3"]),
                ]
                remaining_qty = initial_qty
                entry_time = idx
                bars_held = 0
                tp_hit = [False, False, False]
                in_position = True

    if ticker:
        upgrade_ticker_learning_profile(
            ticker, data, cfg, daily_df=daily_df, profiles_cache=profiles_cache, defer_save=defer_save
        )

    return data, pd.DataFrame(trades)


def analyze_spy_200ema_cycles(spy_df: pd.DataFrame, lookback_years: int = 5) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Track SPY 200 EMA breakdowns over lookback_years and average trading days to trough."""
    if spy_df.empty:
        return pd.DataFrame(), {"cycles_found": 0, "avg_days_to_bottom": None, "below_200_ema": False}
    df = spy_df.copy().sort_index()
    df["ema_9"] = compute_ema(df["close"], 9)
    df["ema_21"] = compute_ema(df["close"], 21)
    df["ema_50"] = compute_ema(df["close"], 50)
    df["ema_200"] = compute_ema(df["close"], 200)

    cutoff_rows = min(len(df), lookback_years * 252)
    window = df.iloc[-cutoff_rows:].copy()

    below_2d = (window["close"] < window["ema_200"]) & (window["close"].shift(1) < window["ema_200"].shift(1))
    breakdown_mask = below_2d & (~below_2d.shift(1).fillna(False))
    breakdown_positions = np.flatnonzero(breakdown_mask.values)

    cycles: list[dict[str, Any]] = []
    close_arr = window["close"].values
    low_arr = window["low"].values
    dates = window.index

    for idx_num, pos in enumerate(breakdown_positions, start=1):
        next_pos = breakdown_positions[idx_num] if idx_num < len(breakdown_positions) else len(window)
        if pos + 1 >= next_pos:
            continue
        sub_lows = low_arr[pos + 1 : next_pos]
        trough_rel_idx = int(np.argmin(sub_lows))
        trough_pos = pos + 1 + trough_rel_idx
        days_to_bottom = int(trough_pos - pos)
        breakdown_price = float(close_arr[pos])
        trough_price = float(low_arr[trough_pos])
        drawdown_pct = ((trough_price - breakdown_price) / breakdown_price) * 100.0
        cycles.append({
            "cycle": idx_num,
            "breakdown_date": dates[pos].strftime("%Y-%m-%d"),
            "breakdown_close": round(breakdown_price, 2),
            "bottom_date": dates[trough_pos].strftime("%Y-%m-%d"),
            "bottom_low": round(trough_price, 2),
            "days_to_bottom": days_to_bottom,
            "drawdown_from_cross_pct": round(drawdown_pct, 2),
        })

    cycles_df = pd.DataFrame(cycles)
    last_close = float(df["close"].iloc[-1])
    last_200 = float(df["ema_200"].iloc[-1])
    summary: dict[str, Any] = {
        "cycles_found": int(len(cycles_df)),
        "avg_days_to_bottom": round(float(cycles_df["days_to_bottom"].mean()), 1) if not cycles_df.empty else None,
        "median_days_to_bottom": round(float(cycles_df["days_to_bottom"].median()), 1) if not cycles_df.empty else None,
        "below_200_ema": bool(last_close < last_200),
        "spy_close": round(last_close, 2),
        "spy_ema_200": round(last_200, 2),
    }
    return cycles_df, summary


def rank_bounce_and_divergence_leaders(
    spy_df: pd.DataFrame,
    holdings_dict: dict[str, pd.DataFrame],
    window_days: int = 20,
) -> pd.DataFrame:
    """Rank top SPY holdings by relative strength, bullish divergence, and 9/21 EMA trend."""
    if spy_df.empty or len(spy_df) < window_days:
        return pd.DataFrame()
    spy_sub = spy_df.iloc[-window_days:]
    spy_ret = ((float(spy_sub["close"].iloc[-1]) / float(spy_sub["close"].iloc[0])) - 1.0) * 100.0
    spy_first_half_low = float(spy_sub["low"].iloc[: window_days // 2].min())
    spy_second_half_low = float(spy_sub["low"].iloc[window_days // 2 :].min())
    spy_lower_low = spy_second_half_low < spy_first_half_low

    rows: list[dict[str, Any]] = []
    for ticker, h_df in holdings_dict.items():
        if h_df is None or len(h_df) < window_days:
            continue
        sub = h_df.iloc[-window_days:].copy()
        ema9 = float(compute_ema(h_df["close"], 9).iloc[-1])
        ema21 = float(compute_ema(h_df["close"], 21).iloc[-1])
        last_close = float(sub["close"].iloc[-1])
        ret_pct = ((last_close / float(sub["close"].iloc[0])) - 1.0) * 100.0
        rel_strength_pct = ret_pct - spy_ret
        h1_low = float(sub["low"].iloc[: window_days // 2].min())
        h2_low = float(sub["low"].iloc[window_days // 2 :].min())
        bullish_divergence = bool(spy_lower_low and (h2_low > h1_low))
        rows.append({
            "ticker": ticker,
            "period_return_pct": round(ret_pct, 2),
            "rel_vs_spy_pct": round(rel_strength_pct, 2),
            "bullish_divergence": bullish_divergence,
            "above_9_21_ema": bool((last_close > ema9) and (last_close > ema21)),
        })
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows).sort_values(by=["bullish_divergence", "rel_vs_spy_pct"], ascending=[False, False])
    return ranked.reset_index(drop=True)


def analyze_sector_money_flow() -> dict[str, Any]:
    """Rank all 11 S&P 500 Sector ETFs by 20D Chaikin Money Flow (CMF), Relative Strength vs SPY, and EMA trend."""
    tickers = ["SPY"] + list(SECTOR_ETFS.keys())
    data_map = fetch_yf_batch_ohlcv(tickers, period="3mo", interval="1d")
    spy_df = data_map.get("SPY")

    spy_5d = 0.0
    spy_20d = 0.0
    if spy_df is not None and len(spy_df) >= 21:
        spy_5d = (float(spy_df["close"].iloc[-1]) / float(spy_df["close"].iloc[-6]) - 1.0) * 100.0
        spy_20d = (float(spy_df["close"].iloc[-1]) / float(spy_df["close"].iloc[-21]) - 1.0) * 100.0

    rows: list[dict[str, Any]] = []
    for sym, (name, bucket) in SECTOR_ETFS.items():
        df = data_map.get(sym)
        if df is None or len(df) < 21:
            continue

        close = df["close"]
        cur_px = float(close.iloc[-1])
        ret_5d = (cur_px / float(close.iloc[-6]) - 1.0) * 100.0
        ret_20d = (cur_px / float(close.iloc[-21]) - 1.0) * 100.0
        rel_20d = ret_20d - spy_20d

        ema21 = float(compute_ema(close, 21).iloc[-1])
        ema50 = float(compute_ema(close, 50).iloc[-1])
        above_emas = cur_px > ema21 and cur_px > ema50

        # 20-day Chaikin Money Flow (CMF): > +0.05 = Institutional Inflow, < -0.05 = Outflow
        sub20 = df.iloc[-20:]
        hl_range = (sub20["high"] - sub20["low"]).replace(0.0, np.nan)
        mf_mult = ((sub20["close"] - sub20["low"]) - (sub20["high"] - sub20["close"])) / hl_range
        mf_vol = (mf_mult.fillna(0.0) * sub20["volume"]).sum()
        vol_sum = float(sub20["volume"].sum())
        cmf_20 = float(mf_vol / vol_sum) if vol_sum > 0 else 0.0

        vol_avg20 = float(sub20["volume"].mean())
        rvol = float(df["volume"].iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 1.0

        flow_score = (cmf_20 * 100.0) + (rel_20d * 1.2) + ((ret_5d - spy_5d) * 0.8)

        if cmf_20 >= 0.05 and rel_20d > 0 and above_emas:
            status = "🟢 STRONG INFLOW"
        elif cmf_20 > 0 or rel_20d > 0:
            status = "🟡 ROTATING IN"
        elif cmf_20 <= -0.05 and rel_20d < 0:
            status = "🔴 HEAVY OUTFLOW"
        else:
            status = "⚪ LAGGING"

        rows.append({
            "ticker": sym,
            "name": name,
            "bucket": bucket,
            "price": cur_px,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "rel_20d": rel_20d,
            "cmf_20": cmf_20,
            "rvol": rvol,
            "above_emas": above_emas,
            "status": status,
            "flow_score": flow_score,
        })

    rows.sort(key=lambda r: r["flow_score"], reverse=True)

    off_rows = [r for r in rows if r["bucket"] in ("OFFENSIVE", "CYCLICAL")]
    def_rows = [r for r in rows if r["bucket"] == "DEFENSIVE"]
    off_avg_rel = sum(r["rel_20d"] for r in off_rows) / len(off_rows) if off_rows else 0.0
    def_avg_rel = sum(r["rel_20d"] for r in def_rows) / len(def_rows) if def_rows else 0.0

    if off_avg_rel > def_avg_rel + 0.5:
        rotation_regime = "🚀 RISK-ON (Money flowing into Growth & Cyclicals)"
    elif def_avg_rel > off_avg_rel + 0.5:
        rotation_regime = "🛡️ RISK-OFF (Money rotating into Defensives)"
    else:
        rotation_regime = "⚖️ MIXED / SELECTIVE ROTATION"

    return {
        "sectors": rows,
        "spy_5d": spy_5d,
        "spy_20d": spy_20d,
        "rotation_regime": rotation_regime,
    }


def evaluate_setup_signals(
    df: pd.DataFrame,
    ipo_anchor_idx: Optional[int] = None,
    vol_mult: float = 1.5,
) -> pd.DataFrame:
    """Evaluate EMA no-man-land rule, structure, anchored VWAP, volume, and ICT confluence."""
    if df.empty:
        return pd.DataFrame()
    out = find_swing_points(df, lookback=2)
    out["ema_9"] = compute_ema(out["close"], 9)
    out["ema_21"] = compute_ema(out["close"], 21)
    out["ema_50"] = compute_ema(out["close"], 50)
    out["ema_200"] = compute_ema(out["close"], 200)
    resolved_anchor = max(0, len(out) - 120) if ipo_anchor_idx is None else ipo_anchor_idx
    out["ipo_vwap"] = compute_anchored_vwap(out, anchor_idx=resolved_anchor)
    out["vol_sma20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["unusual_volume"] = out["volume"] >= (out["vol_sma20"] * vol_mult)

    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)
    bullish_reversal_bar = (
        (out["close"] > out["open"])
        & (prev_close < prev_open)
        & (out["close"] >= ((prev_open + prev_close) / 2.0))
    )
    recent_msl = out["is_swing_low"].rolling(3, min_periods=1).max().astype(bool)
    out["structure_forming"] = recent_msl | bullish_reversal_bar

    c2_high = out["high"].shift(2)
    c0_low = out["low"]
    out["bullish_fvg"] = (c2_high < c0_low) & (out["close"] > out["open"])

    prior_low = out["low"].shift(1)
    prior_high = out["high"].shift(1)
    out["liq_sweep_mss"] = (out["low"] < prior_low) & (out["close"] > prior_high)

    between_21_50 = (out["close"].between(out["ema_21"], out["ema_50"], inclusive="both")) | (
        out["close"].between(out["ema_50"], out["ema_21"], inclusive="both")
    )
    between_50_200 = (out["close"].between(out["ema_50"], out["ema_200"], inclusive="both")) | (
        out["close"].between(out["ema_200"], out["ema_50"], inclusive="both")
    )
    in_between_emas = between_21_50 | between_50_200

    out["ema_zone_actionable"] = np.where(in_between_emas, out["structure_forming"], True)
    out["above_ipo_vwap"] = out["close"] >= out["ipo_vwap"]
    out["sma20"] = out["close"].rolling(20, min_periods=1).mean()
    out["high_20_prev"] = out["high"].rolling(20, min_periods=1).max().shift(1)
    out["breakout_20d"] = (out["close"] > out["high_20_prev"]) & (out["close"] > out["sma20"])
    out["sell_sma20_cross"] = (out["close"] < out["sma20"]) & (out["close"].shift(1) >= out["sma20"].shift(1))

    out["high_confluence_long"] = (
        out["ema_zone_actionable"]
        & out["above_ipo_vwap"]
        & out["unusual_volume"]
        & (out["bullish_fvg"] | out["liq_sweep_mss"] | out["structure_forming"])
    )
    return out


def build_live_market_scan(lookback_years: int = 5, window_days: int = 20) -> dict[str, Any]:
    """Pull live Yahoo Finance data for SPY and top holdings to compute regime and leaders."""
    spy_df = fetch_yf_ohlcv("SPY", period="6y", interval="1d")
    _, spy_summary = analyze_spy_200ema_cycles(spy_df, lookback_years=lookback_years)
    holdings_df_map = fetch_yf_batch_ohlcv(TOP_SPY_HOLDINGS, period="6mo", interval="1d")
    leaders_df = rank_bounce_and_divergence_leaders(spy_df, holdings_df_map, window_days=window_days)

    top_leader = leaders_df.iloc[0]["ticker"] if not leaders_df.empty else "N/A"
    top_rel = leaders_df.iloc[0]["rel_vs_spy_pct"] if not leaders_df.empty else 0.0
    avg_days = spy_summary.get("avg_days_to_bottom")
    below_200 = spy_summary.get("below_200_ema", False)

    profiles = load_ticker_profiles()
    total_upgrades = sum(int(p.get("upgrade_version", 1)) for p in profiles.values())

    regime_tag = "SPY < 200 EMA (Defensive/Bottom Watch)" if below_200 else "SPY >= 200 EMA (Trend Filter)"
    avg_days_str = f"{avg_days}d avg to trough" if avg_days is not None else "No 200 EMA breaks"
    note = (
        f"{regime_tag} | 5Y SPY 200EMA: {avg_days_str} | "
        f"Top Rel Leader ({window_days}d): {top_leader} ({top_rel:+.2f}% vs SPY) | "
        f"Per-Ticker Continuous Learner: {len(profiles)} tickers isolated ({total_upgrades} total DNA upgrades)."
    )
    return {
        "spy_summary": spy_summary,
        "leaders": leaders_df,
        "adaptation_note": note,
        "filter_state": "Isolated Per-Ticker DNA + 1H Swing-Low Stop",
    }


# ==============================================================================
# 6. JOURNAL METRICS & WEEKLY PDF REPORT GENERATOR
# ==============================================================================
def _fmt_signed_money(val: float) -> str:
    if abs(val) < 0.005:
        return "$+0.00"
    return f"${val:+,.2f}"


def _fmt_short_pnl(val: float) -> str:
    if abs(val) < 0.005:
        return "+0"
    return f"{val:+,.2f}" if abs(val) < 100 else f"{val:+,.0f}"


def _truncate_telegram(text: str, max_len: int = TELEGRAM_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 30] + "\n<i>...(truncated)</i>"


def _split_telegram_chunks(text: str, max_len: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split long HTML messages cleanly on line breaks so nothing is lost."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    for line in text.splitlines():
        add_len = len(line) + 1
        if current_lines and (current_len + add_len) > max_len:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += add_len
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


async def _reply_long_html(update: Update, text: str) -> None:
    if not update.message:
        return
    for chunk in _split_telegram_chunks(text):
        await update.message.reply_text(chunk, parse_mode="HTML")


def _get_week_bounds(now_ct: datetime) -> tuple[datetime, datetime]:
    """Return Monday 00:00 CT and Friday 23:59 CT for the current trading week."""
    monday = (now_ct - timedelta(days=now_ct.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)
    return monday, friday


def calculate_journal_metrics(
    entries: list[dict[str, Any]],
    live_prices: dict[str, float],
    now_ct: datetime,
) -> dict[str, Any]:
    """Automatically compute all summary cards, row metrics, and rolling win rate."""
    week_start, week_end = _get_week_bounds(now_ct)
    total_long_setups = len(entries)
    open_count = 0
    closed_wins = 0
    closed_losses = 0
    realized_weekly_pnl = 0.0
    unrealized_open_pnl = 0.0
    total_realized_pnl_all_time = 0.0
    closed_outcomes_chronological: list[bool] = []
    formatted_rows: list[dict[str, str]] = []
    journal_updated = False

    with _STATE_LOCK:
        profiles_cache = load_ticker_profiles()
        profiles_dirty = False

        for item in entries:
            ticker = str(item.get("ticker", "")).upper()
            opened_raw = str(item.get("bought_date", now_ct.strftime("%Y-%m-%d")))
            try:
                opened_dt = datetime.strptime(opened_raw[:10], "%Y-%m-%d").replace(tzinfo=CT_TZ)
                opened_short = opened_dt.strftime("%m-%d")
            except ValueError:
                opened_dt = now_ct
                opened_short = opened_raw[:5]

            strategy = str(item.get("strategy", "Boroden Fib Cluster"))
            entry_px = float(item.get("entry", 0.0))
            stop_px = float(item.get("stop", 0.0))
            target_px = float(item.get("target", 0.0))
            shares = float(item.get("shares", 0.0))
            exit_px_raw = item.get("exit")

            is_closed = exit_px_raw is not None and exit_px_raw != ""
            if is_closed:
                exit_px = float(exit_px_raw)
                trade_pnl = (exit_px - entry_px) * shares
                total_realized_pnl_all_time += trade_pnl
                closed_outcomes_chronological.append(trade_pnl > 0)
                if trade_pnl > 0:
                    closed_wins += 1
                elif trade_pnl < 0:
                    closed_losses += 1
                if not item.get("learned_recorded"):
                    closed_str = str(item.get("closed_date", now_ct.strftime("%Y-%m-%d")))
                    j_key = f"JRN:{opened_raw[:10]}:{closed_str}:{entry_px:.4f}:{exit_px:.4f}"
                    record_ticker_trade_outcome(
                        ticker,
                        entry_px,
                        exit_px,
                        stop_px,
                        target_px,
                        trade_pnl,
                        closed_str,
                        "JOURNAL_SYNC",
                        strategy,
                        trade_key=j_key,
                        profiles_cache=profiles_cache,
                        defer_save=True,
                    )
                    item["learned_recorded"] = True
                    journal_updated = True
                    profiles_dirty = True
                closed_date_raw = str(item.get("closed_date", opened_raw))
                try:
                    closed_dt = datetime.strptime(closed_date_raw[:10], "%Y-%m-%d").replace(tzinfo=CT_TZ)
                except ValueError:
                    closed_dt = now_ct
                if (week_start <= closed_dt <= week_end) or (week_start <= opened_dt <= week_end):
                    realized_weekly_pnl += trade_pnl
                status_str = "CLOSED"
                cur_or_exit_px = exit_px
                pnl_for_cell = trade_pnl
            else:
                open_count += 1
                status_str = "OPEN"
                cur_or_exit_px = live_prices.get(ticker, entry_px)
                trade_pnl = (cur_or_exit_px - entry_px) * shares
                unrealized_open_pnl += trade_pnl
                pnl_for_cell = trade_pnl

            formatted_rows.append({
                "ticker": ticker,
                "opened": opened_short,
                "strategy": strategy,
                "entry": f"${entry_px:,.2f}",
                "stop": f"${stop_px:,.2f}",
                "target": f"${target_px:,.2f}",
                "status": status_str,
                "cur_exit_pnl": f"${cur_or_exit_px:,.2f} ({_fmt_short_pnl(pnl_for_cell)})",
            })

        if profiles_dirty:
            save_ticker_profiles(profiles_cache)
        if journal_updated:
            save_journal(entries)

    closed_total = closed_wins + closed_losses
    closed_win_rate = (closed_wins / closed_total * 100.0) if closed_total > 0 else 0.0
    last_10 = closed_outcomes_chronological[-10:]
    rolling_10_wr = (sum(1 for w in last_10 if w) / len(last_10) * 100.0) if last_10 else 0.0
    current_equity = STARTING_CAPITAL + total_realized_pnl_all_time + unrealized_open_pnl

    return {
        "week_start": week_start,
        "week_end": week_end,
        "total_long_setups": total_long_setups,
        "closed_win_rate_str": f"{closed_win_rate:.1f}% ({closed_wins}W/{closed_losses}L)",
        "open_positions_str": f"{open_count} Active",
        "realized_weekly_pnl_str": _fmt_signed_money(realized_weekly_pnl),
        "unrealized_open_pnl_str": _fmt_signed_money(unrealized_open_pnl),
        "rolling_10_wr": rolling_10_wr,
        "current_equity": current_equity,
        "rows": formatted_rows,
    }


def _draw_page_chrome(canvas, doc, bot_name: str = BOT_NAME) -> None:
    canvas.saveState()
    width, height = letter
    banner_height = 0.88 * inch
    canvas.setFillColor(colors.HexColor("#0F243E"))
    canvas.rect(0, height - banner_height, width, banner_height, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 13)
    canvas.drawString(0.45 * inch, height - 0.34 * inch, "WEEKLY SWING TRADING JOURNAL (LONG-ONLY)")
    canvas.setFillColor(colors.HexColor("#CBD5E1"))
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(0.45 * inch, height - 0.52 * inch, "Dual-Engine Strategy: ICT Grade A+ OTE, Boroden Fib & Per-Ticker Continuous Learner")
    now_ct = datetime.now(CT_TZ)
    week_start, week_end = _get_week_bounds(now_ct)
    gen_str = now_ct.strftime("Generated %I:%M %p CT")
    week_str = f"Reporting Week: {week_start.strftime('%b %d, %Y')} - {week_end.strftime('%b %d, %Y')} ({gen_str})"
    canvas.drawRightString(width - 0.45 * inch, height - 0.36 * inch, week_str)
    footer_y = 0.35 * inch
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.6)
    canvas.line(0.45 * inch, footer_y + 0.14 * inch, width - 0.45 * inch, footer_y + 0.14 * inch)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.45 * inch, footer_y, f"Generated by {bot_name}")
    canvas.drawRightString(width - 0.45 * inch, footer_y, f"Page {canvas._pageNumber}")
    canvas.restoreState()


def build_weekly_journal_pdf(
    output_path: str | Path | None = None,
    atr_buffer_label: str = "1H/4H Swing Low",
    filter_state: str | None = None,
    adaptation_note: str | None = None,
) -> Path:
    now_ct = datetime.now(CT_TZ)
    if output_path is None:
        output_path = Path(f"Weekly_Trading_Journal_{now_ct.strftime('%Y-%m-%d')}.pdf")
    output_path = Path(output_path)

    scan = build_live_market_scan(lookback_years=5, window_days=20)
    resolved_filter = html.escape(filter_state or scan["filter_state"])
    resolved_note = html.escape(adaptation_note or scan["adaptation_note"])
    escaped_atr_label = html.escape(atr_buffer_label)

    entries = load_journal()
    open_tickers = [str(e.get("ticker", "")) for e in entries if e.get("exit") is None or e.get("exit") == ""]
    live_prices = get_current_prices(open_tickers)
    metrics = calculate_journal_metrics(entries, live_prices, now_ct)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=1.05 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    card_lbl = ParagraphStyle("CardLbl", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748B"), leading=8.5)
    card_val = ParagraphStyle("CardVal", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#0F172A"), leading=13)
    sec_title = ParagraphStyle("SecTitle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#0F243E"), leading=11, spaceBefore=6, spaceAfter=4)
    cell_txt = ParagraphStyle("CellTxt", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#1E293B"), leading=9.5)
    tbl_hdr = ParagraphStyle("TblHdr", parent=styles["Normal"], fontSize=7.5, textColor=colors.HexColor("#475569"), leading=9)
    engine_txt = ParagraphStyle("EngTxt", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#1E3A8A"), leading=10)

    usable_w = letter[0] - 0.9 * inch
    col_3 = usable_w / 3.0

    kpi_row_1 = Table(
        [
            [Paragraph("<b>TOTAL LONG SETUPS</b>", card_lbl), Paragraph("<b>CLOSED WIN RATE</b>", card_lbl), Paragraph("<b>OPEN POSITIONS</b>", card_lbl)],
            [Paragraph(f"<b>{metrics['total_long_setups']}</b>", card_val), Paragraph(f"<b>{html.escape(metrics['closed_win_rate_str'])}</b>", card_val), Paragraph(f"<b>{html.escape(metrics['open_positions_str'])}</b>", card_val)],
        ],
        colWidths=[col_3, col_3, col_3],
    )
    kpi_row_1.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")), ("BOX", (0, 0), (0, -1), 0.75, colors.HexColor("#E2E8F0")), ("BOX", (1, 0), (1, -1), 0.75, colors.HexColor("#E2E8F0")), ("BOX", (2, 0), (2, -1), 0.75, colors.HexColor("#E2E8F0")), ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, 0), (-1, 0), 1), ("TOPPADDING", (0, 1), (-1, 1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))

    kpi_row_2 = Table(
        [
            [Paragraph("<b>REALIZED WEEKLY P&amp;L</b>", card_lbl), Paragraph("<b>UNREALIZED OPEN P&amp;L</b>", card_lbl), Paragraph("<b>STOP LOSS RULE</b>", card_lbl)],
            [Paragraph(f"<b>{html.escape(metrics['realized_weekly_pnl_str'])}</b>", card_val), Paragraph(f"<b>{html.escape(metrics['unrealized_open_pnl_str'])}</b>", card_val), Paragraph(f"<b>{escaped_atr_label}</b>", card_val)],
        ],
        colWidths=[col_3, col_3, col_3],
    )
    kpi_row_2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")), ("BOX", (0, 0), (0, -1), 0.75, colors.HexColor("#E2E8F0")), ("BOX", (1, 0), (1, -1), 0.75, colors.HexColor("#E2E8F0")), ("BOX", (2, 0), (2, -1), 0.75, colors.HexColor("#E2E8F0")), ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, 0), (-1, 0), 1), ("TOPPADDING", (0, 1), (-1, 1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))

    engine_box = Table(
        [
            [Paragraph(f"<b>Per-Ticker Continuous Learning Status</b> (Rolling 10-Trade Win Rate: <b>{metrics['rolling_10_wr']:.1f}%</b> | Filter: <b>{resolved_filter}</b>)", engine_txt)],
            [Paragraph(f"<font color='#475569'>Latest Adaptation: {resolved_note} (Capital Base: ${STARTING_CAPITAL:,.0f} | Est. Equity: ${metrics['current_equity']:,.2f})</font>", cell_txt)],
        ],
        colWidths=[usable_w],
    )
    engine_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#2563EB")), ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, 0), (-1, 0), 2), ("TOPPADDING", (0, 1), (-1, 1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))

    trade_headers = [
        Paragraph("<b>Ticker</b>", tbl_hdr), Paragraph("<b>Opened</b>", tbl_hdr), Paragraph("<b>Strategy/Engine</b>", tbl_hdr),
        Paragraph("<b>Entry</b>", tbl_hdr), Paragraph("<b>Stop</b>", tbl_hdr), Paragraph("<b>Target</b>", tbl_hdr),
        Paragraph("<b>Status</b>", tbl_hdr), Paragraph("<b>Cur/Exit P&amp;L</b>", tbl_hdr),
    ]
    trade_data = [trade_headers]
    for r in metrics["rows"]:
        trade_data.append([
            Paragraph(f"<b>{html.escape(r['ticker'])}</b>", cell_txt),
            Paragraph(html.escape(r["opened"]), cell_txt),
            Paragraph(html.escape(r["strategy"]), cell_txt),
            Paragraph(html.escape(r["entry"]), cell_txt),
            Paragraph(html.escape(r["stop"]), cell_txt),
            Paragraph(html.escape(r["target"]), cell_txt),
            Paragraph(f"<b>{html.escape(r['status'])}</b>", cell_txt),
            Paragraph(f"<b>{html.escape(r['cur_exit_pnl'])}</b>", cell_txt),
        ])

    col_widths = [0.55 * inch, 0.55 * inch, 1.75 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.65 * inch, 1.94 * inch]
    trade_table = Table(trade_data, colWidths=col_widths, repeatRows=1)
    trade_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")), ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#CBD5E1")), ("LINEBELOW", (0, 1), (-1, -1), 0.5, colors.HexColor("#E2E8F0")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))

    elements = [
        kpi_row_1, Spacer(1, 0.1 * inch), kpi_row_2, Spacer(1, 0.12 * inch),
        engine_box, Spacer(1, 0.1 * inch), Paragraph("<b>WEEKLY LONG TRADE LOG &amp; PERFORMANCE</b>", sec_title), trade_table,
    ]
    doc.build(elements, onFirstPage=lambda c, d: _draw_page_chrome(c, d, BOT_NAME), onLaterPages=lambda c, d: _draw_page_chrome(c, d, BOT_NAME))
    return output_path


# ==============================================================================
# 7. MARKET SCANNER & STRUCTURED ALERT EVALUATION (BATCHED PER-TICKER LEARNING)
# ==============================================================================
async def evaluate_market(
    universe: list[str],
    holdings: dict[str, dict[str, Any]],
    include_4h_boroden: bool = True,
) -> dict[str, Any]:
    """
    Evaluate universe + holdings for Entry, Sell, Stop Loss, Target, and Heads-Up Watch alerts.
    Loads ticker profiles once into memory, updates all tickers in-memory, and merges atomically at the end.
    """
    spy_below_200ema = False
    if universe and REQUIRE_SPY_ABOVE_200EMA:
        spy_df = await asyncio.to_thread(fetch_yf_ohlcv, "SPY", "1y", "1d")
        if not spy_df.empty and len(spy_df) >= 200:
            spy_ema200 = float(compute_ema(spy_df["close"], 200).iloc[-1])
            spy_close = float(spy_df["close"].iloc[-1])
            spy_below_200ema = spy_close < spy_ema200

    all_symbols = sorted(set(universe) | set(holdings.keys()))
    raw_map = await asyncio.to_thread(fetch_yf_batch_ohlcv, all_symbols, "6mo", "1d")

    h1_map = (
        await asyncio.to_thread(fetch_yf_batch_ohlcv, all_symbols, "60d", "1h")
        if all_symbols
        else {}
    )
    h4_map = (
        {sym: _resample_to_4h(df1h) for sym, df1h in h1_map.items() if not df1h.empty}
        if include_4h_boroden
        else {}
    )

    boroden_cfg = Boroden4HConfig()
    profiles_cache = load_ticker_profiles()
    updated_syms: set[str] = set()

    buy_alerts: list[str] = []
    buy_signals: list[dict[str, Any]] = []
    sell_alerts: list[str] = []
    stoploss_alerts: list[str] = []
    target_alerts: list[str] = []
    exit_signals: list[dict[str, Any]] = []
    heads_up_signals: list[dict[str, Any]] = []

    for sym in universe:
        df = raw_map.get(sym)
        df_1h = h1_map.get(sym)
        df_4h = h4_map.get(sym)

        has_4h = df_4h is not None and len(df_4h) >= 15
        chart_df = df_4h if has_4h else df
        tf_lbl = "4H" if has_4h else "1D"
        if chart_df is None or chart_df.empty:
            continue

        ticker_prof = upgrade_ticker_learning_profile(
            sym, chart_df, boroden_cfg, daily_df=df, profiles_cache=profiles_cache, defer_save=True
        )
        updated_syms.add(sym)

        matched_strategies: list[str] = []
        sig_df = pd.DataFrame()
        b4h_df = pd.DataFrame()
        vol_ratio = 1.0

        if df is not None and len(df) >= 20:
            sig_df = evaluate_setup_signals(df, vol_mult=max(1.35, float(ticker_prof.get("vol_mult", 1.20)) + 0.25))
            last_d = sig_df.iloc[-1]
            vol_avg_d = float(last_d.get("vol_sma20", 1.0))
            if vol_avg_d > 0:
                vol_ratio = float(last_d["volume"]) / vol_avg_d

            if bool(last_d.get("breakout_20d", False)):
                matched_strategies.append("20d Breakout")
            if bool(last_d.get("bullish_fvg", False)) and bool(last_d.get("ema_zone_actionable", False)) and bool(last_d.get("above_ipo_vwap", False)):
                matched_strategies.append("ICT Bullish FVG")
            if bool(last_d.get("liq_sweep_mss", False)) and bool(last_d.get("ema_zone_actionable", False)):
                matched_strategies.append("ICT Liquidity Sweep MSS")
            if (
                bool(last_d.get("above_ipo_vwap", False))
                and bool(last_d.get("unusual_volume", False))
                and bool(last_d.get("structure_forming", False))
                and bool(last_d.get("ema_zone_actionable", False))
            ):
                matched_strategies.append("IPO VWAP + Volume Reversal")

        boroden_fired = False
        if df_4h is not None and len(df_4h) >= (2 * boroden_cfg.pivot_len + 5):
            b4h_df = evaluate_boroden_4h_signals(
                df_4h, boroden_cfg, daily_df=df, ticker=sym, df_1h=df_1h, ticker_profile=ticker_prof
            )
            last_4h = b4h_df.iloc[-1]
            if bool(last_4h.get("boroden_4h_long_signal", False)):
                boroden_fired = True
                matched_strategies.append("4H Boroden Fib Cluster")

        if len(chart_df) >= 35 and detect_butterfly_harmonic(chart_df) is not None:
           matched_strategies.append("Bullish Butterfly Harmonic")

        trend_4h_ok = True
        if not b4h_df.empty and "trend_allows_long_4h" in b4h_df.columns:
            trend_4h_ok = bool(b4h_df.iloc[-1]["trend_allows_long_4h"])

        if (
            not spy_below_200ema
            and trend_4h_ok
            and len(set(matched_strategies)) >= MIN_ALERT_STRATEGIES
        ):
            if boroden_fired and not b4h_df.empty:
                last_4h = b4h_df.iloc[-1]
                entry_px = float(last_4h["close"])
                entry_ts = b4h_df.index[-1]
                stop_px = float(last_4h["boroden_4h_stop"])
                _, stop_ts = find_latest_swing_low_before_entry(b4h_df, len(b4h_df) - 1, entry_px)
                tp1_px = float(last_4h["boroden_4h_tp1"])
                tp2_px = float(last_4h["boroden_4h_tp2"])
                tp3_px = float(last_4h["boroden_4h_tp3"])
                sig_tf = "4H"
                sig_chart_df = b4h_df
            else:
                ref_df = sig_df if not sig_df.empty else chart_df
                entry_px = float(ref_df["close"].iloc[-1])
                entry_ts = ref_df.index[-1]
                stop_px, stop_ts = find_latest_swing_low_before_entry(ref_df, len(ref_df) - 1, entry_px)
                tp_mult = float(ticker_prof.get("tp_scale_mult", 1.0))
                risk = max(entry_px - stop_px, entry_px * 0.01)
                tp1_px = round(entry_px + (1.5 * tp_mult) * risk, 2)
                tp2_px = round(entry_px + (2.5 * tp_mult) * risk, 2)
                tp3_px = round(entry_px + (3.5 * tp_mult) * risk, 2)
                sig_tf = "1D" if not sig_df.empty else tf_lbl
                sig_chart_df = ref_df

            rr_ratio = (tp1_px - entry_px) / max(entry_px - stop_px, 1e-6) if (entry_px > stop_px and tp1_px > entry_px) else 1.5
            rr_bonus = min(1.0, max(0.0, (rr_ratio - 1.2) / 2.3))
            vol_bonus = min(1.0, max(0.0, (vol_ratio - 1.0) / 1.5))
            quality_bonus = round(0.65 * rr_bonus + 0.35 * vol_bonus, 4)

            scored = score_strategy_confluence(matched_strategies, quality_bonus, ticker_prof)
            if scored is not None:
                register_ticker_shadow_signal(
                    sym,
                    entry_ts,
                    entry_px,
                    stop_px,
                    tp1_px,
                    scored["matched_strategies"],
                    timeframe=sig_tf,
                    profiles_cache=profiles_cache,
                    defer_save=True,
                )

                strat_summary = " + ".join(scored["matched_strategies"])
                rated_strategy_label = f"[{scored['rating']} {scored['score']:.1f}] {strat_summary}"
                dna_ver = ticker_prof.get("upgrade_version", 1)
                norm_entry_ts = _to_naive_et_timestamp(entry_ts)
                bar_id_str = norm_entry_ts.isoformat() if norm_entry_ts is not None else str(entry_ts)
                msg = (
                    f"{TAG_ENTRY} {scored['badge']} (<code>{scored['score']:.1f}/100</code> | "
                    f"<code>{scored['matched_count']}</code> Strategies | DNA <code>v{dna_ver}</code>) — <b>{sym}</b> | "
                    f"Entry: <code>${entry_px:.2f}</code> | Stop: <code>${stop_px:.2f}</code> | "
                    f"Exit TP1: <code>${tp1_px:.2f}</code>\n"
                    f"<i>Matched: {html.escape(strat_summary)}</i>"
                )
                buy_signals.append({
                    "ticker": sym,
                    "rating": scored["rating"],
                    "score": scored["score"],
                    "matched_count": scored["matched_count"],
                    "matched_strategies": scored["matched_strategies"],
                    "strategy": rated_strategy_label,
                    "entry": entry_px,
                    "entry_time": entry_ts,
                    "bar_id": bar_id_str,
                    "stop": stop_px,
                    "stop_time": stop_ts,
                    "tp1": tp1_px,
                    "tp2": tp2_px,
                    "tp3": tp3_px,
                    "chart_df": sig_chart_df,
                    "tf_label": sig_tf,
                    "message": msg,
                })

        if len(chart_df) >= 55:
            hu_notes = detect_heads_up_setups(chart_df)
            if hu_notes:
                heads_up_signals.append({
                    "ticker": sym,
                    "price": float(chart_df["close"].iloc[-1]),
                    "tf_label": tf_lbl,
                    "notes": hu_notes,
                })

    if updated_syms:
        _merge_and_save_scanned_profiles(profiles_cache, updated_syms)

    buy_signals.sort(key=lambda s: (s["score"], s["matched_count"]), reverse=True)
    buy_alerts = [s["message"] for s in buy_signals]

    for sym, pos in holdings.items():
        df = raw_map.get(sym)
        df_1h = h1_map.get(sym)
        df_4h = h4_map.get(sym)
        if df is None or df.empty:
            continue
        sig_df = evaluate_setup_signals(df, vol_mult=1.5)
        last = sig_df.iloc[-1]
        close_px = float(last["close"])
        sma20 = float(last.get("sma20", close_px))
        entry_px = float(pos.get("entry", close_px))
        entry_dt = pos.get("bought_date")
        stop_dt = pos.get("stop_date")
        stop_price = pos.get("stop")
        target_price = pos.get("target")
        chart_df = df_4h if (df_4h is not None and len(df_4h) >= 15) else df
        tf_lbl = "4H" if (df_4h is not None and len(df_4h) >= 15) else "1D"

        intraday_src = df_1h if (df_1h is not None and not df_1h.empty) else chart_df
        intraday_low = close_px
        intraday_high = close_px
        hit_stop_first = False
        hit_target_first = False
        if intraday_src is not None and not intraday_src.empty:
            src_idx = _to_naive_et_index(intraday_src.index)
            bought_ts = _to_naive_et_timestamp(entry_dt)
            if bought_ts is not None:
                # If date-only YYYY-MM-DD was provided, start monitoring after that session's close (16:00 ET)
                if len(str(entry_dt).strip()) <= 10:
                    bought_ts = bought_ts.normalize() + pd.Timedelta(hours=16)
                post_entry = intraday_src.loc[src_idx > np.datetime64(bought_ts)]
            else:
                post_entry = intraday_src.iloc[-1:]

            trigger_fill_px: Optional[float] = None
            if not post_entry.empty:
                intraday_low = float(post_entry["low"].min())
                intraday_high = float(post_entry["high"].max())
                close_px = float(post_entry["close"].iloc[-1])
                for _, bar in post_entry.iterrows():
                    b_open = float(bar["open"])
                    b_low = float(bar["low"])
                    b_high = float(bar["high"])
                    if stop_price is not None and b_low <= float(stop_price):
                        hit_stop_first = True
                        intraday_low = b_low
                        trigger_fill_px = min(b_open, float(stop_price)) if b_open < float(stop_price) else float(stop_price)
                        break
                    if target_price is not None and b_high >= float(target_price):
                        hit_target_first = True
                        intraday_high = b_high
                        trigger_fill_px = max(b_open, float(target_price)) if b_open > float(target_price) else float(target_price)
                        break

        if stop_price is not None and (hit_stop_first or (not hit_target_first and close_px <= float(stop_price))):
            fill_px = trigger_fill_px if trigger_fill_px is not None else min(close_px, float(stop_price))
            msg = f"{TAG_STOPLOSS} — <b>{sym}</b> | Price: <code>${fill_px:.2f}</code> (Low <code>${intraday_low:.2f}</code>) &le; Stop Loss: <code>${float(stop_price):.2f}</code> (Entry: <code>${entry_px:.2f}</code>)"
            stoploss_alerts.append(msg)
            exit_signals.append({
                "ticker": sym,
                "exit_type": "STOP_LOSS",
                "current_price": fill_px,
                "entry": entry_px,
                "entry_time": entry_dt,
                "stop": float(stop_price),
                "stop_time": stop_dt,
                "target": float(target_price) if target_price else round(entry_px * 1.05, 2),
                "chart_df": chart_df,
                "tf_label": tf_lbl,
                "message": msg,
            })
        elif target_price is not None and (hit_target_first or close_px >= float(target_price)):
            fill_px = trigger_fill_px if trigger_fill_px is not None else max(close_px, float(target_price))
            msg = f"{TAG_TARGET} — <b>{sym}</b> | Price: <code>${fill_px:.2f}</code> (High <code>${intraday_high:.2f}</code>) &ge; Exit Target: <code>${float(target_price):.2f}</code> (Entry: <code>${entry_px:.2f}</code>)"
            target_alerts.append(msg)
            exit_signals.append({
                "ticker": sym,
                "exit_type": "TARGET_EXIT",
                "current_price": fill_px,
                "entry": entry_px,
                "entry_time": entry_dt,
                "stop": float(stop_price) if stop_price else round(entry_px * 0.96, 2),
                "stop_time": stop_dt,
                "target": float(target_price),
                "chart_df": chart_df,
                "tf_label": tf_lbl,
                "message": msg,
            })

    return {
        "buy": buy_alerts,
        "buy_signals": buy_signals,
        "sell": sell_alerts,
        "stoploss": stoploss_alerts,
        "target": target_alerts,
        "exit_signals": exit_signals,
        "heads_up": heads_up_signals,
    }


def _format_section(title: str, items: list[str], empty_msg: str) -> str:
    if not items:
        return f"<b>{title}</b>\n{empty_msg}"
    shown = items[:25]
    numbered = [f"{i}. {item}" for i, item in enumerate(shown, start=1)]
    extra = f"\n<i>...and {len(items) - 25} more</i>" if len(items) > 25 else ""
    return f"<b>{title} ({len(items)})</b>\n" + "\n".join(numbered) + extra


def _describe_4h_fib_proximity(
    price: float,
    swing_high: float,
    swing_low: float,
    prev_corr: float,
    tol_pct: float,
) -> tuple[list[str], str]:
    """Return matching Fibonacci cluster labels and the nearest Fib support level."""
    if np.isnan(swing_high) or np.isnan(swing_low) or swing_high <= swing_low or price <= 0:
        return [], "N/A (no confirmed 4H bull swing yet)"

    rng = swing_high - swing_low
    named_levels: list[tuple[str, float]] = [
        ("38.2% Ret", swing_high - rng * 0.382),
        ("50.0% Ret", swing_high - rng * 0.500),
        ("61.8% Ret", swing_high - rng * 0.618),
        ("78.6% Ret", swing_high - rng * 0.786),
        ("127.2% Ext", swing_high - rng * 1.272),
        ("161.8% Ext", swing_high - rng * 1.618),
    ]
    if not np.isnan(prev_corr) and prev_corr > 0:
        named_levels.append(("100% Sym", swing_high - prev_corr))
        named_levels.append(("161.8% Sym", swing_high - prev_corr * 1.618))

    tol = price * (tol_pct / 100.0)
    matched = [f"{lbl} (${lvl:.2f})" for lbl, lvl in named_levels if abs(lvl - price) <= tol]

    nearest_lbl, nearest_val = min(named_levels, key=lambda item: abs(item[1] - price))
    dist_pct = ((price - nearest_val) / nearest_val) * 100.0
    nearest_str = f"{nearest_lbl} at ${nearest_val:.2f} ({dist_pct:+.2f}% away)"
    return matched, nearest_str


async def analyze_single_ticker_with_chart(ticker: str) -> tuple[str, Optional[io.BytesIO], str]:
    """Run full single-ticker analysis, upgrading ONLY that ticker's isolated DNA profile without repeat-backtest inflation."""
    sym = ticker.strip().upper().lstrip("/$").replace(".", "-")
    cfg = Boroden4HConfig()

    daily_df, df_1h = await asyncio.gather(
        asyncio.to_thread(fetch_yf_ohlcv, sym, "1y", "1d"),
        asyncio.to_thread(fetch_yf_ohlcv, sym, "60d", "1h"),
    )
    if daily_df.empty and df_1h.empty:
        return (
            f"Could not find market data for <b>{html.escape(sym)}</b>. Check the ticker symbol and try again.",
            None,
            "",
        )

    df_4h = _resample_to_4h(df_1h) if not df_1h.empty else pd.DataFrame()
    chart_source_df = df_4h if (not df_4h.empty and len(df_4h) >= 15) else daily_df
    tf_label = "4H" if (not df_4h.empty and len(df_4h) >= 15) else "1D"

    ticker_prof = upgrade_ticker_learning_profile(sym, chart_source_df, cfg, daily_df=daily_df)

    passed_reasons: list[str] = []
    failed_reasons: list[str] = []
    triggered_engines: list[str] = []
    matched_strategies: list[str] = []

    cur_px = 0.0
    ema21 = ema50 = ema200 = sma20 = high_20_prev = 0.0
    daily_breakout = False
    daily_confluence = False
    vol_ratio = 1.0
    sig_df = pd.DataFrame()

    if not daily_df.empty:
        sig_df = evaluate_setup_signals(daily_df, vol_mult=max(1.35, float(ticker_prof.get("vol_mult", 1.20)) + 0.25))
        last_d = sig_df.iloc[-1]
        cur_px = float(last_d["close"])
        ema21 = float(last_d["ema_21"])
        ema50 = float(last_d["ema_50"])
        ema200 = float(last_d["ema_200"])
        sma20 = float(last_d["sma20"])
        high_20_prev = float(last_d["high_20_prev"]) if not np.isnan(last_d["high_20_prev"]) else cur_px
        ipo_vwap = float(last_d["ipo_vwap"]) if not np.isnan(last_d["ipo_vwap"]) else cur_px
        d_vol = float(last_d["volume"])
        d_vol_avg = float(last_d["vol_sma20"])
        if d_vol_avg > 0:
            vol_ratio = d_vol / d_vol_avg

        daily_breakout = bool(last_d["breakout_20d"])
        daily_confluence = bool(last_d["high_confluence_long"])

        if daily_breakout:
            matched_strategies.append("20d Breakout")
            triggered_engines.append("Daily 20d Breakout")
            passed_reasons.append(
                f"Daily 20d Breakout: Close (<code>${cur_px:.2f}</code>) broke above 20d High (<code>${high_20_prev:.2f}</code>) &amp; SMA20 (<code>${sma20:.2f}</code>)"
            )
        else:
            failed_reasons.append(
                f"Daily 20d Breakout: Close (<code>${cur_px:.2f}</code>) has not cleared 20d High (<code>${high_20_prev:.2f}</code>)"
            )

        if bool(last_d["bullish_fvg"]) and bool(last_d["ema_zone_actionable"]) and bool(last_d["above_ipo_vwap"]):
            matched_strategies.append("ICT Bullish FVG")
        if bool(last_d["liq_sweep_mss"]) and bool(last_d["ema_zone_actionable"]):
            matched_strategies.append("ICT Liquidity Sweep MSS")
        if (
            bool(last_d["above_ipo_vwap"])
            and bool(last_d["unusual_volume"])
            and bool(last_d["structure_forming"])
            and bool(last_d["ema_zone_actionable"])
        ):
            matched_strategies.append("IPO VWAP + Volume Reversal")

        if daily_confluence:
            triggered_engines.append("Daily High-Confluence (IPO VWAP/ICT)")
            passed_reasons.append(
                f"Daily High-Confluence: Above IPO VWAP (<code>${ipo_vwap:.2f}</code>), unusual volume, and bullish ICT/structure trigger"
            )
        else:
            missing_d_parts: list[str] = []
            if not bool(last_d["ema_zone_actionable"]):
                missing_d_parts.append("stuck in EMA no-man's-land without reversal structure")
            if not bool(last_d["above_ipo_vwap"]):
                missing_d_parts.append(f"below IPO VWAP (${ipo_vwap:.2f})")
            if not bool(last_d["unusual_volume"]):
                missing_d_parts.append(f"daily vol ({d_vol:,.0f}) &lt; threshold")
            if not (bool(last_d["bullish_fvg"]) or bool(last_d["liq_sweep_mss"]) or bool(last_d["structure_forming"])):
                missing_d_parts.append("no bullish FVG / liquidity sweep / swing low")
            failed_reasons.append(
                f"Daily High-Confluence: {', '.join(missing_d_parts) if missing_d_parts else 'criteria not met'}"
            )

    boroden_active = False
    b4h_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    tp1 = tp2 = tp3 = 0.0
    boroden_stop_px = 0.0

    if not df_4h.empty and len(df_4h) >= (2 * cfg.pivot_len + 5):
        b4h_df, trades_df = await asyncio.to_thread(
            run_boroden_4h_backtest, df_4h, cfg, daily_df, sym, True, df_1h
        )
        ticker_prof = get_ticker_learning_profile(sym, cfg)
        last_4h = b4h_df.iloc[-1]
        if cur_px <= 0:
            cur_px = float(last_4h["close"])

        boroden_active = bool(last_4h["boroden_4h_long_signal"])
        if boroden_active:
            matched_strategies.append("4H Boroden Fib Cluster")
            triggered_engines.append("4H Boroden Fib Cluster")

        boroden_stop_px = float(last_4h["boroden_4h_stop"])
        tp1 = float(last_4h["boroden_4h_tp1"])
        tp2 = float(last_4h["boroden_4h_tp2"])
        tp3 = float(last_4h["boroden_4h_tp3"])

        d_ok = bool(last_4h["daily_trend_ok"]) and (ema50 <= 0 or cur_px >= ema50)
        if d_ok:
            passed_reasons.append(f"Daily Trend Filter: Price (<code>${cur_px:.2f}</code>) &ge; Daily EMA50 (<code>${ema50:.2f}</code>)")
        else:
            failed_reasons.append(f"Daily Trend Filter: Price (<code>${cur_px:.2f}</code>) is below Daily EMA50 (<code>${ema50:.2f}</code>)")

        trend_4h_ok = bool(last_4h["trend_allows_long_4h"])
        has_hl = bool(last_4h["has_higher_low_4h"])
        is_dt = bool(last_4h["is_downtrend_4h"])
        if trend_4h_ok:
            struct_desc = "4H Higher Low confirmed" if has_hl else "4H Uptrend (HH + HL)"
            passed_reasons.append(f"4H Structure: {struct_desc}")
        else:
            dt_note = "4H Downtrend (LH + LL)" if is_dt else "no confirmed 4H Higher Low yet"
            failed_reasons.append(f"4H Structure: {dt_note}")

        cluster_cnt = int(last_4h["fib_cluster_count"])
        req_cluster = int(last_4h["eff_min_cluster_4h"])
        valid_swing = bool(last_4h["valid_bull_swing_4h"])
        in_fib = bool(last_4h["in_fib_zone_4h"])
        sh_4h = float(last_4h["last_swing_high_4h"])
        sl_4h = float(last_4h["last_swing_low_4h"])
        corr_4h = float(last_4h["prev_corr_drop_4h"])
        matched_fibs, nearest_fib = _describe_4h_fib_proximity(cur_px, sh_4h, sl_4h, corr_4h, cfg.cluster_tol_pct)

        if in_fib:
            fib_detail = ", ".join(matched_fibs) if matched_fibs else f"{cluster_cnt} levels"
            passed_reasons.append(
                f"4H Fib Support Cluster ({sym} req &ge;{req_cluster}): <code>{cluster_cnt}/{req_cluster}</code> levels matched ({html.escape(fib_detail)})"
            )
        else:
            if not valid_swing:
                failed_reasons.append("4H Fib Cluster: No valid confirmed 4H bull swing (High &gt; Low)")
            elif not np.isnan(sh_4h) and cur_px >= sh_4h:
                failed_reasons.append(
                    f"4H Fib Cluster: Price (<code>${cur_px:.2f}</code>) is at/above 4H Swing High (<code>${sh_4h:.2f}</code>) — waiting for pullback"
                )
            else:
                failed_reasons.append(
                    f"4H Fib Cluster ({sym} req &ge;{req_cluster}): Only <code>{cluster_cnt}/{req_cluster}</code> levels within {cfg.cluster_tol_pct}% (nearest: {html.escape(nearest_fib)})"
                )

        candle_ok = bool(last_4h["candle_body_ok_4h"])
        rsi_ok = bool(last_4h["rsi_reclaim_4h"])
        rsi_val = float(last_4h["rsi_14"])
        req_rsi = float(last_4h["eff_rsi_level_4h"])
        ema21_4h = float(last_4h["ema_21_4h"])
        if candle_ok and rsi_ok:
            passed_reasons.append(
                f"4H Momentum &amp; Candle: Bullish close + RSI (<code>{rsi_val:.1f}</code> &ge; <code>{req_rsi:.0f}</code>) / EMA21 (<code>${ema21_4h:.2f}</code>) reclaim"
            )
        else:
            c_reasons: list[str] = []
            if not candle_ok:
                c_reasons.append("4H candle is red or closed in lower 40% of range")
            if not rsi_ok:
                c_reasons.append(f"4H RSI ({rsi_val:.1f}) &lt; {req_rsi:.0f} ({sym} learned) or no EMA21 ({ema21_4h:.2f}) reclaim")
            failed_reasons.append(f"4H Momentum &amp; Candle: {'; '.join(c_reasons)}")

        vol_4h_ok = bool(last_4h["vol_ok_4h"])
        v_cur = float(last_4h["volume"])
        eff_vm = float(last_4h["eff_vol_mult_4h"])
        v_req = float(last_4h["vol_sma20"]) * eff_vm
        if vol_4h_ok:
            passed_reasons.append(f"4H Volume: <code>{v_cur:,.0f}</code> &ge; <code>{v_req:,.0f}</code> ({eff_vm:.2f}x avg)")
        else:
            failed_reasons.append(f"4H Volume: <code>{v_cur:,.0f}</code> is below <code>{v_req:,.0f}</code> ({eff_vm:.2f}x 20-bar avg)")
    else:
        failed_reasons.append("4H Boroden Engine: Not enough intraday 4H bars available")

    if chart_source_df is not None and len(chart_source_df) >= 35 and detect_butterfly_harmonic(chart_source_df) is not None:
        matched_strategies.append("Bullish Butterfly Harmonic")

    holdings = load_holdings()
    active_pos = holdings.get(sym)

    use_daily_plan = (not boroden_active) and (not active_pos) and (not sig_df.empty) and bool(matched_strategies)
    plan_df = sig_df if use_daily_plan else chart_source_df
    plan_tf = "1D" if use_daily_plan else tf_label

    entry_px = float(plan_df["close"].iloc[-1]) if not plan_df.empty else cur_px
    entry_ts = plan_df.index[-1] if not plan_df.empty else None
    entry_bar_idx = len(plan_df) - 1
    stop_px = boroden_stop_px if (boroden_active and boroden_stop_px > 0 and boroden_stop_px < entry_px) else 0.0
    stop_ts = None

    if active_pos:
        entry_px = float(active_pos.get("entry", cur_px))
        entry_ts = active_pos.get("bought_date") or entry_ts
        dt_series = _to_naive_et_index(plan_df.index)
        t_ts = _to_naive_et_timestamp(entry_ts)
        if t_ts is not None:
            entry_bar_idx = int(np.argmin(np.abs(dt_series - np.datetime64(t_ts))))
        stop_px = float(active_pos.get("stop", stop_px))
        stop_ts = active_pos.get("stop_date")
        tp1 = float(active_pos.get("target", tp1))
        triggered_engines.append("Active Open Position")

    if stop_px <= 0:
        stop_px, stop_ts = find_latest_swing_low_before_entry(
            plan_df, entry_bar_idx, entry_px, left=2, right=1, lookback=15
        )
    elif stop_ts is None:
        _, stop_ts = find_latest_swing_low_before_entry(
            plan_df, entry_bar_idx, entry_px, left=2, right=1, lookback=15
        )

    if use_daily_plan or tp1 <= entry_px:
        tp_mult = float(ticker_prof.get("tp_scale_mult", 1.0))
        risk = max(entry_px - stop_px, entry_px * 0.01)
        tp1 = round(entry_px + (1.5 * tp_mult) * risk, 2)
        tp2 = round(entry_px + (2.5 * tp_mult) * risk, 2)
        tp3 = round(entry_px + (3.5 * tp_mult) * risk, 2)

    rr_ratio = (tp1 - entry_px) / max(entry_px - stop_px, 1e-6) if (entry_px > stop_px and tp1 > entry_px) else 1.5
    rr_bonus = min(1.0, max(0.0, (rr_ratio - 1.2) / 2.3))
    vol_bonus = min(1.0, max(0.0, (vol_ratio - 1.0) / 1.5))
    quality_bonus = round(0.65 * rr_bonus + 0.35 * vol_bonus, 4)
    scored = score_strategy_confluence(matched_strategies, quality_bonus, ticker_prof)

    if scored is not None:
        register_ticker_shadow_signal(
            sym, entry_ts, entry_px, stop_px, tp1, scored["matched_strategies"], timeframe=plan_tf
        )
        score_line = (
            f"<b>Alert Rating:</b> {scored['badge']} — Score: <code>{scored['score']:.1f}/100</code> "
            f"(<code>{scored['matched_count']}/6</code> Strategies | {sym} Strat Wt: <code>{scored['avg_strategy_weight']:.2f}x</code>)"
        )
    else:
        score_line = f"<b>Alert Rating:</b> ⚪ IGNORED — Score: <code>0/100</code> (<code>0/6</code> Strategies matched on {sym})"

    take_position = (scored is not None) or (active_pos is not None)
    entry_date_label = pd.to_datetime(entry_ts).strftime("%b %d") if entry_ts is not None else "Now"
    stop_date_label = pd.to_datetime(stop_ts).strftime("%b %d") if stop_ts is not None else "Swing Low"

    if take_position:
        decision_header = f"🟢 <b>POSITION DECISION: TAKE LONG — {html.escape(sym)}</b> (Cur: <code>${cur_px:.2f}</code>)"
        rating_tag = f" | {scored['rating']} {scored['score']:.1f}" if scored else ""
        chart_decision = f"LONG @ ${entry_px:.2f} ({entry_date_label}{rating_tag})"
        all_labels = list(dict.fromkeys(triggered_engines + (scored["matched_strategies"] if scored else [])))
        engines_str = html.escape(" + ".join(all_labels))
        why_section = (
            f"{score_line}\n\n"
            f"<b>Why Taking Position ({engines_str}):</b>\n"
            + "\n".join(f"✅ {r}" for r in passed_reasons)
        )
        if failed_reasons:
            why_section += "\n\n<b>Other Checks:</b>\n" + "\n".join(f"• {r}" for r in failed_reasons)
    else:
        decision_header = f"🔴 <b>POSITION DECISION: DO NOT TAKE POSITION — {html.escape(sym)}</b> (<code>${cur_px:.2f}</code>)"
        chart_decision = "DO NOT TAKE (IGNORED - 0 STRATEGIES)"
        why_section = (
            f"{score_line}\n\n"
            "<b>Why Not Taking Position (Failed Conditions):</b>\n"
            + "\n".join(f"❌ {r}" for r in failed_reasons)
        )
        if passed_reasons:
            why_section += "\n\n<b>Conditions Already Met:</b>\n" + "\n".join(f"✅ {r}" for r in passed_reasons)

    heads_up_notes = detect_heads_up_setups(chart_source_df)
    if heads_up_notes:
        why_section += (
            f"\n\n<b>👀 Early Heads-Up Watch ({tf_label} EMA / Patterns):</b>\n"
            + "\n".join(f"• {html.escape(n)}" for n in heads_up_notes)
        )

    risk_dollars = max(entry_px - stop_px, 0.01) if stop_px > 0 else 0.0
    risk_pct = (risk_dollars / entry_px * 100.0) if (entry_px > 0 and stop_px > 0) else 0.0

    levels_lines = [
        f"<b>{'Trade Plan' if take_position else 'Reference Levels'}:</b>",
        f"• <b>Entry ({entry_date_label}):</b> <code>${entry_px:.2f}</code> | <b>Current Price:</b> <code>${cur_px:.2f}</code>",
        f"• <b>Stop Loss ({stop_date_label} Swing Low):</b> <code>${stop_px:.2f}</code> (-{risk_pct:.2f}%)",
        f"• <b>Exit Targets:</b> TP1 <code>${tp1:.2f}</code> | TP2 <code>${tp2:.2f}</code> | TP3 <code>${tp3:.2f}</code>",
        f"• <b>Daily EMAs:</b> EMA21 <code>${ema21:.2f}</code> | EMA50 <code>${ema50:.2f}</code> | EMA200 <code>${ema200:.2f}</code>",
        f"• <b>{html.escape(sym)} Learned DNA:</b> <code>{html.escape(format_ticker_learning_summary(ticker_prof))}</code>",
    ]
    if stop_px > 0 and tp1 > 0:
        levels_lines.append(
            f"• <b>Log Trade:</b> <code>/buy {html.escape(sym)} {entry_px:.2f} {stop_px:.2f} {tp1:.2f} 10</code>"
        )

    if not trades_df.empty:
        total_pnl = float(trades_df["pnl"].sum())
        win_rate = (trades_df["pnl"] > 0).mean() * 100.0
        levels_lines.append(
            f"• <b>60D 4H Backtest:</b> {len(trades_df)} exits | Win Rate: <code>{win_rate:.1f}%</code> | Net P&amp;L: <code>{_fmt_signed_money(total_pnl)}</code>"
        )

    full_msg = _truncate_telegram(f"{decision_header}\n\n{why_section}\n\n" + "\n".join(levels_lines))

    chart_buf = await asyncio.to_thread(
        generate_trade_candlestick_chart,
        sym,
        plan_df,
        entry_px,
        stop_px,
        tp1,
        tp2,
        tp3,
        cur_px,
        plan_tf,
        chart_decision,
        55,
        entry_ts,
        stop_ts,
    )

    short_caption = _truncate_telegram(
        (
            f"{decision_header}\n"
            f"{score_line}\n"
            f"• <b>Entry ({entry_date_label}):</b> <code>${entry_px:.2f}</code> | <b>Stop Loss ({stop_date_label} Low):</b> <code>${stop_px:.2f}</code> (-{risk_pct:.2f}%)\n"
            f"• <b>Exits:</b> TP1 <code>${tp1:.2f}</code> | TP2 <code>${tp2:.2f}</code> | TP3 <code>${tp3:.2f}</code>"
        ),
        TELEGRAM_CAPTION_MAX_LEN,
    )
    return full_msg, chart_buf, short_caption


# ==============================================================================
# 8. AUTOMATED 1-HOUR SCANNER & 4-HOUR EXIT / STOP-LOSS MONITOR
# ==============================================================================
async def run_hourly_entry_scan(app: Application, manual_chat_id: Optional[int | str] = None) -> None:
    """
    Runs every 1 hour during US market session (or immediately on /scan):
    1. Upgrades each ticker's isolated DNA profile in a single batched merge & scans for Entry alerts.
    2. Persists deduplication keys in `alert_state.json` so restarts & weekends never re-fire alerts.
    """
    chat_id = manual_chat_id or _resolve_alert_chat_id(app)
    if not chat_id:
        logger.warning("Skipping 1H auto-scan: no chat_id configured yet. Send /start to the bot.")
        return

    if manual_chat_id is None and not _is_us_equity_session_open():
        logger.info("Market closed (weekend/after-hours); skipping automated 1H entry scan.")
        return

    if _SCAN_LOCK.locked():
        if manual_chat_id is not None:
            await app.bot.send_message(
                chat_id=chat_id,
                text="⏳ A market scan is already in progress — please wait for it to finish.",
            )
        else:
            logger.info("Skipping scheduled 1H scan because another scan is already running.")
        return

    async with _SCAN_LOCK:
        universe = await asyncio.to_thread(get_sp500_and_nasdaq100_tickers)
        holdings = load_holdings()
        alerts = await evaluate_market(universe, holdings, include_4h_boroden=True)

        auto_buy = bool(app.bot_data.get("auto_buy_enabled", AUTO_BUY_DEFAULT))
        alert_state = load_alert_state()
        recent_alerted = alert_state["entry_alerts"]
        state_dirty = False

        new_buy_signals: list[dict[str, Any]] = []
        for sig in alerts["buy_signals"]:
            dedup_key = f"{sig['ticker']}:{sig['bar_id']}"
            if manual_chat_id is not None or dedup_key not in recent_alerted:
                new_buy_signals.append(sig)

        for sig in new_buy_signals[:10]:
            dedup_key = f"{sig['ticker']}:{sig['bar_id']}"
            recent_alerted[dedup_key] = datetime.now(ET_TZ).isoformat()
            state_dirty = True

            sym = sig["ticker"]
            entry_px = float(sig["entry"])
            entry_ts = sig.get("entry_time")
            stop_px = float(sig["stop"])
            stop_ts = sig.get("stop_time")
            tp1_px = float(sig["tp1"])
            tp2_px = float(sig["tp2"])
            tp3_px = float(sig["tp3"])

            auto_bought_note = ""
            current_holdings = load_holdings()
            if (
                auto_buy
                and float(sig.get("score", 0.0)) >= MIN_AUTO_BUY_SCORE
                and sym not in current_holdings
                and len(current_holdings) < MAX_AUTO_OPEN_POSITIONS
                and entry_px > 0
            ):
                alloc_dollars = STARTING_CAPITAL * AUTO_POSITION_PCT
                shares = round(alloc_dollars / entry_px, 2)
                if shares > 0:
                    norm_s = _to_naive_et_timestamp(stop_ts)
                    # Use actual execution time so intraday stop monitoring only inspects bars after entry
                    bought_dt_str = datetime.now(ET_TZ).isoformat()
                    stop_dt_str = norm_s.isoformat() if norm_s is not None else None
                    record_position_buy(
                        sym,
                        entry_px,
                        stop_px,
                        tp1_px,
                        shares,
                        strategy=sig["strategy"],
                        bought_date=bought_dt_str,
                        stop_date=stop_dt_str,
                    )
                    auto_bought_note = (
                        f"\n🤖 <b>Auto-Bought:</b> <code>{shares:g}</code> shares @ <code>${entry_px:.2f}</code> "
                        f"(4H Exit/Stop monitor active)"
                    )

            chart_title = f"ENTRY [{sig['rating']} {sig['score']:.1f}] ({sig['matched_count']} Strats)"
            chart_buf = await asyncio.to_thread(
                generate_trade_candlestick_chart,
                sym,
                sig["chart_df"],
                entry_px,
                stop_px,
                tp1_px,
                tp2_px,
                tp3_px,
                entry_px,
                sig["tf_label"],
                chart_title,
                55,
                entry_ts,
                stop_ts,
            )
            caption = _truncate_telegram(sig["message"] + auto_bought_note, TELEGRAM_CAPTION_MAX_LEN)
            if chart_buf is not None:
                await app.bot.send_photo(chat_id=chat_id, photo=chart_buf, caption=caption, parse_mode="HTML")
            else:
                await app.bot.send_message(chat_id=chat_id, text=caption, parse_mode="HTML")
            await asyncio.sleep(0.5)

        now_et = datetime.now(ET_TZ)
        heads_up_seen = alert_state["heads_up_alerts"]
        fresh_heads_up: list[str] = []

        for hu in alerts.get("heads_up", []):
            sym = hu["ticker"]
            last_sent_iso = heads_up_seen.get(sym)
            if manual_chat_id is None and last_sent_iso:
                try:
                    prev_dt = datetime.fromisoformat(last_sent_iso)
                    if prev_dt.tzinfo is None:
                        prev_dt = prev_dt.replace(tzinfo=ET_TZ)
                    if (now_et - prev_dt) < timedelta(hours=HEADS_UP_COOLDOWN_HOURS):
                        continue
                except ValueError:
                    pass
            heads_up_seen[sym] = now_et.isoformat()
            state_dirty = True
            joined_notes = "; ".join(html.escape(n) for n in hu["notes"])
            fresh_heads_up.append(
                f"• <b>{sym}</b> ({hu['tf_label']} @ <code>${hu['price']:.2f}</code>): {joined_notes}"
            )

        if state_dirty:
            save_alert_state(alert_state)

        if fresh_heads_up:
            header = (
                "👀 <b>HEADS-UP PATTERN &amp; EMA WATCH (Low-Priority Alert)</b>\n"
                "<i>Early setups forming (no confirmed entry trigger yet — watch for breakout/reclaim):</i>\n\n"
            )
            body = "\n".join(fresh_heads_up[:18])
            extra = f"\n<i>...and {len(fresh_heads_up) - 18} more on watch</i>" if len(fresh_heads_up) > 18 else ""
            await app.bot.send_message(
                chat_id=chat_id,
                text=_truncate_telegram(header + body + extra),
                parse_mode="HTML",
                disable_notification=True,
            )

        if manual_chat_id is not None and not new_buy_signals and not fresh_heads_up:
            sections = [
                _format_section("1. Entry Alerts", alerts["buy"], "No active entry signals."),
                _format_section("2. Stop Loss Alerts", alerts["stoploss"], "No stop losses hit."),
                _format_section("3. Target Alerts", alerts["target"], "No targets hit."),
                _format_section("4. Sell Alerts (SMA20 Cross)", alerts["sell"], "No sell signals."),
                _format_section("5. Heads-Up Watch (EMA / Patterns)", [], "No early pattern or EMA pre-cross setups."),
            ]
            await app.bot.send_message(
                chat_id=chat_id,
                text=_truncate_telegram("\n\n".join(sections)),
                parse_mode="HTML",
            )


async def run_4h_exit_and_stoploss_monitor(app: Application) -> None:
    """Runs every 4 hours during US market hours to monitor open holdings and close/upgrade on STOP_LOSS, TARGET_EXIT, or SMA20_EXIT."""
    chat_id = _resolve_alert_chat_id(app)
    holdings = load_holdings()
    if not chat_id or not holdings or not _is_us_equity_session_open():
        return

    async with _SCAN_LOCK:
        alerts = await evaluate_market([], holdings, include_4h_boroden=True)

    for sig in alerts["exit_signals"]:
        sym = sig["ticker"]
        exit_type = sig["exit_type"]
        cur_px = float(sig["current_price"])
        entry_px = float(sig["entry"])
        entry_ts = sig.get("entry_time")
        stop_px = float(sig["stop"])
        stop_ts = sig.get("stop_time")
        target_px = float(sig["target"])

        closed_note = ""
        if exit_type in ("STOP_LOSS", "TARGET_EXIT", "SMA20_EXIT"):
            pnl = record_position_close(sym, cur_px, reason=exit_type)
            if pnl is not None:
                prof = get_ticker_learning_profile(sym)
                closed_note = (
                    f"\n📒 <b>Position Closed:</b> Realized P&amp;L <code>{_fmt_signed_money(pnl)}</code> "
                    f"| <b>{sym} DNA upgraded to v{prof.get('upgrade_version', 1)}</b>"
                )

        chart_buf = await asyncio.to_thread(
            generate_trade_candlestick_chart,
            sym,
            sig["chart_df"],
            entry_px,
            stop_px,
            target_px,
            None,
            None,
            cur_px,
            sig["tf_label"],
            f"{exit_type.replace('_', ' ')} TRIGGERED",
            55,
            entry_ts,
            stop_ts,
        )
        caption = _truncate_telegram(sig["message"] + closed_note, TELEGRAM_CAPTION_MAX_LEN)
        if chart_buf is not None:
            await app.bot.send_photo(chat_id=chat_id, photo=chart_buf, caption=caption, parse_mode="HTML")
        else:
            await app.bot.send_message(chat_id=chat_id, text=caption, parse_mode="HTML")

    remaining = load_holdings()
    if remaining:
        live_prices = await asyncio.to_thread(get_current_prices, list(remaining.keys()))
        status_lines = ["⏱ <b>4-Hour Bought Position Monitor (Exit / Stop-Loss Check)</b>"]
        holdings_dirty = False
        be_updates: dict[str, float] = {}

        for sym, pos in remaining.items():
            entry_px = float(pos.get("entry", 0.0))
            stop_px = float(pos.get("stop", 0.0))
            target_px = float(pos.get("target", 0.0))
            shares = float(pos.get("shares", 0.0))
            cur_px = live_prices.get(sym, entry_px)

            initial_risk = float(pos.get("initial_risk", max(entry_px - stop_px, 0.0)))
            be_note = ""
            if (
                initial_risk > 0
                and stop_px < entry_px
                and cur_px >= entry_px + (BREAKEVEN_R_TRIGGER * initial_risk)
            ):
                stop_px = round(entry_px, 4)
                pos["stop"] = stop_px
                pos["initial_risk"] = round(initial_risk, 4)
                holdings_dirty = True
                be_updates[sym] = stop_px
                be_note = " 🔒 <b>[Stop &rarr; Breakeven]</b>"

            pnl = (cur_px - entry_px) * shares
            to_stop_pct = ((cur_px - stop_px) / cur_px * 100.0) if cur_px > 0 else 0.0
            to_target_pct = ((target_px - cur_px) / cur_px * 100.0) if cur_px > 0 else 0.0
            status_lines.append(
                f"• <b>{sym}</b>: Cur <code>${cur_px:.2f}</code> ({_fmt_signed_money(pnl)}) | "
                f"Stop <code>${stop_px:.2f}</code> ({to_stop_pct:.1f}% buffer) | "
                f"Exit Target <code>${target_px:.2f}</code> ({to_target_pct:+.1f}% away){be_note}"
            )

        if holdings_dirty:
            with _STATE_LOCK:
                latest_holdings = load_holdings()
                for sym, new_stop in be_updates.items():
                    if sym in latest_holdings:
                        latest_holdings[sym]["stop"] = new_stop
                        latest_holdings[sym]["initial_risk"] = remaining[sym]["initial_risk"]
                save_holdings(latest_holdings)

                journal = load_journal()
                for item in reversed(journal):
                    j_sym = str(item.get("ticker", "")).upper()
                    if j_sym in be_updates and (item.get("exit") is None or item.get("exit") == ""):
                        item["stop"] = be_updates[j_sym]
                save_journal(journal)
        await app.bot.send_message(
            chat_id=chat_id,
            text=_truncate_telegram("\n".join(status_lines)),
            parse_mode="HTML",
        )


async def _hourly_scan_loop(app: Application) -> None:
    await asyncio.sleep(15)
    while True:
        try:
            logger.info("Running scheduled 1-hour entry alert scan...")
            await run_hourly_entry_scan(app)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Error in 1-hour auto-scan loop: %s", exc)
        await asyncio.sleep(AUTO_SCAN_INTERVAL_SECONDS)


async def _four_hour_exit_loop(app: Application) -> None:
    await asyncio.sleep(45)
    while True:
        try:
            logger.info("Running scheduled 4-hour exit & stop-loss monitor...")
            await run_4h_exit_and_stoploss_monitor(app)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Error in 4-hour exit/stop-loss monitor loop: %s", exc)
        await asyncio.sleep(EXIT_MONITOR_INTERVAL_SECONDS)


async def _post_init_scheduler(app: Application) -> None:
    app.bot_data.setdefault("auto_buy_enabled", AUTO_BUY_DEFAULT)
    app.bot_data["hourly_task"] = asyncio.create_task(_hourly_scan_loop(app))
    app.bot_data["four_hour_task"] = asyncio.create_task(_four_hour_exit_loop(app))
    logger.info("Background schedulers active: 1H Entry Auto-Scan & 4H Exit/Stop-Loss Monitor.")


async def _post_shutdown_scheduler(app: Application) -> None:
    for key in ("hourly_task", "four_hour_task"):
        task = app.bot_data.get(key)
        if task and not task.done():
            task.cancel()


async def _telegram_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gracefully handle Telegram RetryAfter rate limits and unexpected handler errors."""
    err = context.error
    if isinstance(err, RetryAfter):
        logger.warning("Telegram flood control hit; sleeping %s seconds.", err.retry_after)
        await asyncio.sleep(float(err.retry_after) + 0.5)
        return
    if isinstance(err, TelegramError):
        logger.warning("Telegram API error: %s", err)
        return
    logger.error("Unhandled exception: %s\n%s", err, "".join(traceback.format_exception(None, err, err.__traceback__)))


# ==============================================================================
# 9. TELEGRAM BOT COMMAND & NATURAL-TEXT HANDLERS
# ==============================================================================
def _remember_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        context.bot_data["alert_chat_id"] = update.effective_chat.id


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    auto_state = "ON" if context.bot_data.get("auto_buy_enabled", AUTO_BUY_DEFAULT) else "OFF"
    text = (
        "<b>Stock Alert, Candlestick Chart &amp; Per-Ticker Learner Bot</b>\n\n"
        "<b>Active Automation:</b>\n"
        "• <b>1-Hour Auto-Scan:</b> Upgrades each ticker's isolated DNA profile &amp; sends ranked chart alerts (HIGH &gt;85, MEDIUM 75-85, LOW 60-75)\n"
        "• <b>4-Hour Exit / Stop Monitor:</b> Checks bought positions every 4h &amp; upgrades ticker DNA on every win/loss\n"
        f"• <b>Auto-Buy Tracking:</b> <code>{auto_state}</code> (toggle with <code>/autobuy on</code> or <code>/autobuy off</code>)\n\n"
        "<b>Commands &amp; Quick Keywords:</b>\n"
        "1. Type any ticker (e.g. <code>NVDA</code>, <code>$AAPL</code>, <code>/TSLA</code>) or <code>/boroden TICKER</code> — Candlestick chart + Confluence Score &amp; Ticker DNA\n"
        "2. <code>/sector</code> (or type <code>sector</code>) — See where institutional money is flowing across all 11 S&amp;P sectors\n"
        "3. <code>/open</code> (or type <code>open</code>) — Show all open positions with live unrealized P&amp;L, stops &amp; targets\n"
        "4. <code>/close</code> (or type <code>close</code>) — Show recent closed positions with realized P&amp;L &amp; win rate\n"
        "5. <code>/crossover</code> (or type <code>cross over</code>) — List stocks with confirmed/imminent EMA crossovers (9/21, 21/50, 50/200)\n"
        "6. <code>/pattern</code> (or type <code>pattern</code>) — List stocks forming Bull Flag, Cup &amp; Handle, Falling Wedge, Double Bottom, or Butterfly\n"
        "7. <code>/learn [TICKER]</code> — Inspect a ticker's isolated learned parameters &amp; strategy weights\n"
        "8. <code>/scan</code> — Run immediate scan with ranked chart alerts &amp; Heads-Up Watch\n"
        "9. <code>/regime</code> — SPY 5Y 200EMA bottom stats &amp; top holdings divergence\n"
        "10. <code>/report</code> — Generate Weekly Trading Journal PDF\n"
        "11. <code>/buy TICKER ENTRY STOP TARGET SHARES [YYYY-MM-DD]</code> — Log a bought position\n"
        "12. <code>/sell TICKER EXIT_PRICE</code> — Close a position &amp; upgrade that ticker's learner\n"
        "13. <code>/autobuy on|off</code> — Enable/disable auto-buying on 1H entry alerts\n"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_sector(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show where institutional money is flowing across all 11 S&P 500 sectors."""
    if not update.message:
        return
    _remember_chat(update, context)
    await update.message.reply_text("Analyzing 11 S&P 500 Sector ETFs for money inflow & rotation...")

    res = await asyncio.to_thread(analyze_sector_money_flow)
    sectors = res["sectors"]
    if not sectors:
        await update.message.reply_text("Could not fetch sector ETF data right now. Try again shortly.")
        return

    lines = [
        "💸 <b>SECTOR MONEY FLOW &amp; ROTATION</b>",
        f"<b>Regime:</b> {html.escape(res['rotation_regime'])}",
        f"<b>Benchmark (SPY):</b> 5D <code>{res['spy_5d']:+.2f}%</code> | 20D <code>{res['spy_20d']:+.2f}%</code>",
        "<i>Ranked by 20D Chaikin Money Flow (CMF) + Relative Strength vs SPY:</i>\n",
    ]

    for idx, s in enumerate(sectors, start=1):
        lines.append(
            f"{idx}. <b>{s['ticker']}</b> ({html.escape(s['name'])}) — {s['status']}\n"
            f"   5D: <code>{s['ret_5d']:+.1f}%</code> | 20D: <code>{s['ret_20d']:+.1f}%</code> "
            f"(<code>{s['rel_20d']:+.1f}%</code> vs SPY) | CMF: <code>{s['cmf_20']:+.2f}</code> | RVOL: <code>{s['rvol']:.2f}x</code>"
        )

    top_3 = ", ".join(f"{s['ticker']} ({s['name']})" for s in sectors[:3])
    bot_2 = ", ".join(f"{s['ticker']} ({s['name']})" for s in sectors[-2:])
    lines.append(f"\n🔥 <b>Top Inflow Leaders:</b> {html.escape(top_3)}")
    lines.append(f"❄️ <b>Outflow / Avoid:</b> {html.escape(bot_2)}")

    await _reply_long_html(update, "\n".join(lines))


async def cmd_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all currently open positions with live prices, stops, targets, and unrealized P&L."""
    if not update.message:
        return
    _remember_chat(update, context)

    holdings = load_holdings()
    # Also include any journal entry marked open if missing from holdings.json
    journal = load_journal()
    for item in journal:
        if item.get("exit") is None or item.get("exit") == "":
            sym = str(item.get("ticker", "")).strip().upper()
            if sym and sym not in holdings:
                holdings[sym] = {
                    "entry": float(item.get("entry", 0.0)),
                    "stop": float(item.get("stop", 0.0)),
                    "target": float(item.get("target", 0.0)),
                    "shares": float(item.get("shares", 0.0)),
                    "strategy": str(item.get("strategy", "Journal Open")),
                    "bought_date": str(item.get("bought_date", "")),
                }

    if not holdings:
        await update.message.reply_text(
            "📭 <b>No Open Positions</b>\n"
            "Log a trade with <code>/buy TICKER ENTRY STOP TARGET SHARES</code> or enable <code>/autobuy on</code>.",
            parse_mode="HTML",
        )
        return

    tickers = sorted(holdings.keys())
    live_prices = await asyncio.to_thread(get_current_prices, tickers)

    total_cost = 0.0
    total_value = 0.0
    lines: list[str] = []

    for idx, sym in enumerate(tickers, start=1):
        pos = holdings[sym]
        entry_px = float(pos.get("entry", 0.0))
        stop_px = float(pos.get("stop", 0.0))
        target_px = float(pos.get("target", 0.0))
        shares = float(pos.get("shares", 0.0))
        strat = str(pos.get("strategy", "Manual"))
        bought_dt = str(pos.get("bought_date", ""))[:10]

        cur_px = live_prices.get(sym, entry_px)
        cost_basis = entry_px * shares
        mkt_value = cur_px * shares
        pnl_dollars = mkt_value - cost_basis
        pnl_pct = ((cur_px - entry_px) / entry_px * 100.0) if entry_px > 0 else 0.0

        total_cost += cost_basis
        total_value += mkt_value

        stop_buf_pct = ((cur_px - stop_px) / cur_px * 100.0) if (cur_px > 0 and stop_px > 0) else 0.0
        tgt_rem_pct = ((target_px - cur_px) / cur_px * 100.0) if (cur_px > 0 and target_px > 0) else 0.0
        pnl_icon = "🟢" if pnl_dollars >= 0 else "🔴"

        lines.append(
            f"{idx}. {pnl_icon} <b>{html.escape(sym)}</b> ({shares:g} sh @ <code>${entry_px:.2f}</code>) — "
            f"Cur: <code>${cur_px:.2f}</code> | P&amp;L: <code>{_fmt_signed_money(pnl_dollars)}</code> (<code>{pnl_pct:+.2f}%</code>)\n"
            f"   Stop: <code>${stop_px:.2f}</code> ({stop_buf_pct:.1f}% buffer) | "
            f"Target: <code>${target_px:.2f}</code> ({tgt_rem_pct:+.1f}% away)\n"
            f"   <i>{html.escape(strat)}{f' | Opened {bought_dt}' if bought_dt else ''}</i>"
        )

    total_unrealized = total_value - total_cost
    total_unrealized_pct = (total_unrealized / total_cost * 100.0) if total_cost > 0 else 0.0

    header = (
        f"📂 <b>OPEN POSITIONS ({len(tickers)} Active)</b>\n"
        f"<b>Invested:</b> <code>${total_cost:,.2f}</code> | <b>Current Value:</b> <code>${total_value:,.2f}</code>\n"
        f"<b>Total Unrealized P&amp;L:</b> <code>{_fmt_signed_money(total_unrealized)}</code> (<code>{total_unrealized_pct:+.2f}%</code>)\n\n"
    )
    await _reply_long_html(update, header + "\n".join(lines))


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent closed positions with realized P&L and win rate."""
    if not update.message:
        return
    _remember_chat(update, context)

    limit = 15
    if context.args and context.args[0].isdigit():
        limit = max(1, min(50, int(context.args[0])))

    journal = load_journal()
    closed_items = [
        item for item in journal
        if item.get("exit") is not None and item.get("exit") != ""
    ]

    if not closed_items:
        await update.message.reply_text(
            "📭 <b>No Closed Positions Yet</b>\n"
            "Close an active trade with <code>/sell TICKER EXIT_PRICE</code>.",
            parse_mode="HTML",
        )
        return

    # Sort by closed_date (fallback bought_date), keeping newest first
    indexed_closed = list(enumerate(closed_items))
    indexed_closed.sort(
        key=lambda pair: (str(pair[1].get("closed_date") or pair[1].get("bought_date") or ""), pair[0]),
        reverse=True,
    )

    total_wins = 0
    total_losses = 0
    total_realized = 0.0
    for _, item in indexed_closed:
        pnl = (float(item.get("exit", 0.0)) - float(item.get("entry", 0.0))) * float(item.get("shares", 0.0))
        total_realized += pnl
        if pnl > 0:
            total_wins += 1
        elif pnl < 0:
            total_losses += 1

    total_decided = total_wins + total_losses
    win_rate = (total_wins / total_decided * 100.0) if total_decided > 0 else 0.0

    lines: list[str] = []
    for idx, (_, item) in enumerate(indexed_closed[:limit], start=1):
        sym = str(item.get("ticker", "")).upper()
        entry_px = float(item.get("entry", 0.0))
        exit_px = float(item.get("exit", 0.0))
        shares = float(item.get("shares", 0.0))
        strat = str(item.get("strategy", "Manual"))
        opened_dt = str(item.get("bought_date", ""))[:10]
        closed_dt = str(item.get("closed_date", opened_dt))[:10]

        pnl = (exit_px - entry_px) * shares
        pnl_pct = ((exit_px - entry_px) / entry_px * 100.0) if entry_px > 0 else 0.0
        badge = "🟢 WIN" if pnl > 0 else ("🔴 LOSS" if pnl < 0 else "⚪ BE")

        lines.append(
            f"{idx}. {badge} <b>{html.escape(sym)}</b> ({shares:g} sh) — "
            f"<code>${entry_px:.2f}</code> &rarr; <code>${exit_px:.2f}</code> | "
            f"P&amp;L: <code>{_fmt_signed_money(pnl)}</code> (<code>{pnl_pct:+.2f}%</code>)\n"
            f"   <i>Closed: {closed_dt} (Opened: {opened_dt}) | {html.escape(strat)}</i>"
        )

    header = (
        f"📕 <b>RECENT CLOSED POSITIONS (Showing {min(limit, len(indexed_closed))} of {len(indexed_closed)})</b>\n"
        f"<b>Win Rate:</b> <code>{win_rate:.1f}%</code> ({total_wins}W/{total_losses}L) | "
        f"<b>Total Realized P&amp;L:</b> <code>{_fmt_signed_money(total_realized)}</code>\n\n"
    )
    await _reply_long_html(update, header + "\n".join(lines))


async def cmd_crossover(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan the market for stocks with confirmed or imminent EMA crossovers (9/21, 21/50, 50/200)."""
    if not update.message:
        return
    _remember_chat(update, context)

    pair_filter = context.args[0].strip() if context.args else ""
    await update.message.reply_text("Scanning S&P 500 + Nasdaq 100 for bullish EMA crossovers...")

    universe = await asyncio.to_thread(get_sp500_and_nasdaq100_tickers)
    raw_map = await asyncio.to_thread(fetch_yf_batch_ohlcv, universe, "1y", "1d")

    confirmed_hits: list[dict[str, Any]] = []
    imminent_hits: list[dict[str, Any]] = []

    for sym in universe:
        df = raw_map.get(sym)
        if df is None or len(df) < 55:
            continue
        crosses = detect_ema_crossovers(df)
        if not crosses:
            continue
        if pair_filter:
            crosses = [c for c in crosses if pair_filter in c["pair"] or pair_filter.lower() in c["label"].lower()]
            if not crosses:
                continue

        cur_px = float(df["close"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-20:].mean())
        rvol = float(df["volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
        labels = "; ".join(html.escape(c["label"]) for c in crosses)
        entry_row = {"ticker": sym, "price": cur_px, "rvol": rvol, "labels": labels}

        if any(c["status"] == "CONFIRMED" for c in crosses):
            confirmed_hits.append(entry_row)
        else:
            imminent_hits.append(entry_row)

    confirmed_hits.sort(key=lambda r: r["rvol"], reverse=True)
    imminent_hits.sort(key=lambda r: r["rvol"], reverse=True)

    if not confirmed_hits and not imminent_hits:
        await update.message.reply_text("No bullish EMA crossovers found across the universe right now.")
        return

    sections = [
        f"📈 <b>EMA CROSSOVER SCANNER</b> ({len(confirmed_hits)} Confirmed, {len(imminent_hits)} Imminent)"
    ]
    if confirmed_hits:
        c_lines = [
            f"{i}. <b>{r['ticker']}</b> (<code>${r['price']:.2f}</code> | RVOL <code>{r['rvol']:.2f}x</code>) — {r['labels']}"
            for i, r in enumerate(confirmed_hits[:30], start=1)
        ]
        if len(confirmed_hits) > 30:
            c_lines.append(f"<i>...and {len(confirmed_hits) - 30} more confirmed</i>")
        sections.append("<b>⚡ CONFIRMED BULLISH CROSSOVERS:</b>\n" + "\n".join(c_lines))

    if imminent_hits:
        i_lines = [
            f"{i}. <b>{r['ticker']}</b> (<code>${r['price']:.2f}</code> | RVOL <code>{r['rvol']:.2f}x</code>) — {r['labels']}"
            for i, r in enumerate(imminent_hits[:25], start=1)
        ]
        if len(imminent_hits) > 25:
            i_lines.append(f"<i>...and {len(imminent_hits) - 25} more approaching</i>")
        sections.append("<b>⏳ ABOUT TO CROSS (PRE-CROSS WATCH):</b>\n" + "\n".join(i_lines))

    sections.append("<i>Tip: Type any ticker symbol (e.g. NVDA) to see its full candlestick chart &amp; trade plan.</i>")
    await _reply_long_html(update, "\n\n".join(sections))


async def cmd_pattern(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan the market for stocks forming bullish chart patterns (Bull Flag, Cup & Handle, Falling Wedge, Double Bottom, Butterfly)."""
    if not update.message:
        return
    _remember_chat(update, context)

    pat_filter = context.args[0].strip().lower() if context.args else ""
    await update.message.reply_text("Scanning S&P 500 + Nasdaq 100 for bullish chart patterns...")

    universe = await asyncio.to_thread(get_sp500_and_nasdaq100_tickers)
    raw_map = await asyncio.to_thread(fetch_yf_batch_ohlcv, universe, "6mo", "1d")

    pattern_hits: list[dict[str, Any]] = []
    for sym in universe:
        df = raw_map.get(sym)
        if df is None or len(df) < 50:
            continue
        pats = detect_chart_patterns(df)
        if not pats:
            continue
        if pat_filter:
            pats = [
                p for p in pats
                if pat_filter in p["key"].lower() or pat_filter in p["name"].lower()
            ]
            if not pats:
                continue

        cur_px = float(df["close"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-20:].mean())
        rvol = float(df["volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
        pattern_hits.append({
            "ticker": sym,
            "price": cur_px,
            "rvol": rvol,
            "count": len(pats),
            "labels": "; ".join(html.escape(p["label"]) for p in pats),
        })

    pattern_hits.sort(key=lambda r: (r["count"], r["rvol"]), reverse=True)

    if not pattern_hits:
        await update.message.reply_text("No active chart patterns matched across the universe right now.")
        return

    lines = [
        f"📐 <b>STOCK CHART PATTERN SCANNER ({len(pattern_hits)} Stocks Found)</b>",
        "<i>Patterns tracked: Bull Flag, Cup &amp; Handle, Falling Wedge, Double Bottom, Bullish Butterfly:</i>\n",
    ]
    for i, r in enumerate(pattern_hits[:40], start=1):
        lines.append(
            f"{i}. <b>{r['ticker']}</b> (<code>${r['price']:.2f}</code> | RVOL <code>{r['rvol']:.2f}x</code>) — {r['labels']}"
        )
    if len(pattern_hits) > 40:
        lines.append(f"<i>...and {len(pattern_hits) - 40} more stocks with active patterns</i>")

    lines.append("\n<i>Tip: Type any ticker (e.g. AAPL) to generate its candlestick chart &amp; entry/stop levels.</i>")
    await _reply_long_html(update, "\n".join(lines))


async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inspect a specific ticker's isolated learning DNA or list the most upgraded tickers."""
    if not update.message:
        return
    _remember_chat(update, context)

    if context.args:
        sym = context.args[0].strip().upper().lstrip("/$").replace(".", "-")
        prof = get_ticker_learning_profile(sym)
        weights = prof.get("strategy_weights", {})
        stats = prof.get("strategy_stats", {})

        strat_lines: list[str] = []
        for s in ALL_STRATEGIES:
            w = float(weights.get(s, 1.0))
            st = stats.get(s, {"wins": 0, "losses": 0})
            strat_lines.append(
                f"• {html.escape(s)}: <code>{w:.2f}x</code> (<code>{st.get('wins', 0)}W/{st.get('losses', 0)}L</code>)"
            )

        recent_logs = prof.get("upgrade_log", [])[-5:]
        log_lines = [f"• <code>{html.escape(lg)}</code>" for lg in reversed(recent_logs)]

        msg = (
            f"🧠 <b>ISOLATED TICKER DNA — {html.escape(sym)}</b> (Upgrade <code>v{prof.get('upgrade_version', 1)}</code>)\n"
            f"<i>Learned exclusively from {html.escape(sym)}'s own price action &amp; trades:</i>\n\n"
            f"• <b>Volatility Regime:</b> <code>{prof.get('volatility_regime', 'STANDARD')}</code> "
            f"(Avg ATR: <code>{prof.get('avg_atr_pct') or 0:.2f}%</code>)\n"
            f"• <b>Record:</b> <code>{prof.get('wins', 0)}W / {prof.get('losses', 0)}L</code> | "
            f"<b>Shadow Pending:</b> <code>{len(prof.get('pending_signals', []))}</code>\n"
            f"• <b>Custom Thresholds:</b> ATR Stop <code>{prof.get('atr_stop_mult', 1.25):.2f}x</code> | "
            f"Min Fib Cluster <code>&ge;{prof.get('min_cluster', 2)}</code> | "
            f"RSI Reclaim <code>&ge;{prof.get('min_rsi_reclaim', 42.0):.0f}</code> | "
            f"Vol <code>&ge;{prof.get('vol_mult', 1.20):.2f}x</code> | "
            f"TP Scale <code>{prof.get('tp_scale_mult', 1.0):.2f}x</code>\n\n"
            f"<b>Strategy Weights on {html.escape(sym)}:</b>\n" + "\n".join(strat_lines) + "\n\n"
            f"<b>Recent Upgrades:</b>\n" + ("\n".join(log_lines) if log_lines else "• None yet")
        )
        await _reply_long_html(update, msg)
        return

    profiles = load_ticker_profiles()
    if not profiles:
        await update.message.reply_text("No ticker profiles learned yet. Run <code>/scan</code> or type any ticker like <code>NVDA</code>.", parse_mode="HTML")
        return

    ranked = sorted(profiles.values(), key=lambda p: (int(p.get("upgrade_version", 1)), int(p.get("wins", 0)) + int(p.get("losses", 0))), reverse=True)
    lines = [
        f"🧠 <b>PER-TICKER CONTINUOUS LEARNER OVERVIEW</b> ({len(profiles)} Isolated Tickers)",
        "<i>Use <code>/learn TICKER</code> (e.g. <code>/learn NVDA</code>) for full strategy weights &amp; upgrade log:</i>\n",
    ]
    for p in ranked[:15]:
        sym = html.escape(str(p.get("ticker", "")))
        lines.append(f"• <b>{sym}</b>: <code>{html.escape(format_ticker_learning_summary(p))}</code>")
    await _reply_long_html(update, "\n".join(lines))


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    await update.message.reply_text("Scanning S&P 500 + Nasdaq 100 universe, upgrading per-ticker DNA & generating ranked charts...")
    await run_hourly_entry_scan(context.application, manual_chat_id=update.effective_chat.id)


async def cmd_boroden(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    if not context.args:
        await update.message.reply_text(
            "Usage: <code>/boroden NVDA</code> (or simply type <code>NVDA</code> or <code>/NVDA</code>)",
            parse_mode="HTML",
        )
        return
    ticker = context.args[0]
    await update.message.reply_text(f"Analyzing <b>{html.escape(ticker.upper())}</b>, upgrading its DNA & drawing chart...", parse_mode="HTML")
    report, chart_buf, caption = await analyze_single_ticker_with_chart(ticker)
    if chart_buf is not None:
        await update.message.reply_photo(photo=chart_buf, caption=caption, parse_mode="HTML")
    await update.message.reply_text(report, parse_mode="HTML")


async def cmd_autobuy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    if not context.args or context.args[0].lower() not in ("on", "off"):
        cur = "ON" if context.bot_data.get("auto_buy_enabled", AUTO_BUY_DEFAULT) else "OFF"
        await update.message.reply_text(f"Auto-Buy on 1H Entry Alerts is currently <b>{cur}</b>.\nUse <code>/autobuy on</code> or <code>/autobuy off</code>.", parse_mode="HTML")
        return
    enabled = context.args[0].lower() == "on"
    context.bot_data["auto_buy_enabled"] = enabled
    await update.message.reply_text(
        f"Auto-Buy on 1H Entry Alerts is now <b>{'ON' if enabled else 'OFF'}</b>.",
        parse_mode="HTML",
    )


async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    await update.message.reply_text("Fetching SPY 200EMA regime & top holdings divergence...")
    scan = await asyncio.to_thread(build_live_market_scan, 5, 20)
    s = scan["spy_summary"]
    leaders_df = scan["leaders"]

    lines = [
        "<b>SPY Market Regime &amp; Divergence Leaders</b>",
        f"1. <b>SPY Close:</b> <code>${s.get('spy_close', 0):.2f}</code> vs 200 EMA: <code>${s.get('spy_ema_200', 0):.2f}</code>",
        f"2. <b>Status:</b> {'BELOW 200 EMA (Defensive)' if s.get('below_200_ema') else 'ABOVE 200 EMA (Bullish)'}",
        f"3. <b>5Y Breakdowns:</b> {s.get('cycles_found', 0)} | Avg Days to Bottom: <code>{s.get('avg_days_to_bottom')}</code>",
        "",
        "<b>Top Relative Strength Leaders (20d):</b>",
    ]
    if not leaders_df.empty:
        for idx, row in leaders_df.head(6).iterrows():
            div_tag = " [BULL DIV]" if row["bullish_divergence"] else ""
            lines.append(
                f"{idx + 1}. <b>{row['ticker']}</b>: <code>{row['rel_vs_spy_pct']:+.2f}%</code> vs SPY{div_tag}"
            )
    else:
        lines.append("No leader data available.")

    await _reply_long_html(update, "\n".join(lines))


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    await update.message.reply_text("Generating Weekly Trading Journal PDF...")
    pdf_path = await asyncio.to_thread(build_weekly_journal_pdf)
    with pdf_path.open("rb") as f:
        await update.message.reply_document(document=f, filename=pdf_path.name)


async def cmd_buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    if len(context.args) < 5:
        await update.message.reply_text(
            "Usage: <code>/buy TICKER ENTRY STOP TARGET SHARES [YYYY-MM-DD]</code>\n"
            "Example: <code>/buy NVDA 139.00 135.00 145.20 20 2026-10-26</code>",
            parse_mode="HTML",
        )
        return
    try:
        sym = context.args[0].strip().upper().replace(".", "-")
        entry_px = float(context.args[1])
        stop_px = float(context.args[2])
        target_px = float(context.args[3])
        shares = float(context.args[4])
        bought_date = context.args[5].strip() if len(context.args) >= 6 else None
        if bought_date is not None:
            datetime.strptime(bought_date, "%Y-%m-%d")
    except ValueError:
        await update.message.reply_text("Invalid format. Use: /buy NVDA 139.00 135.00 145.20 20 2026-10-26")
        return

    df_1h = await asyncio.to_thread(fetch_yf_ohlcv, sym, "60d", "1h")
    df_4h = _resample_to_4h(df_1h) if not df_1h.empty else await asyncio.to_thread(fetch_yf_ohlcv, sym, "6mo", "1d")
    tf_lbl = "4H" if (not df_1h.empty and len(df_4h) >= 15) else "1D"

    entry_ts = bought_date or (df_4h.index[-1] if not df_4h.empty else None)
    entry_idx = len(df_4h) - 1
    if entry_ts is not None and not df_4h.empty:
        dt_series = _to_naive_et_index(df_4h.index)
        t_ts = _to_naive_et_timestamp(entry_ts)
        if t_ts is not None:
            entry_idx = int(np.argmin(np.abs(dt_series - np.datetime64(t_ts))))

    _, stop_ts = find_latest_swing_low_before_entry(df_4h, entry_idx, entry_px)
    norm_stop = _to_naive_et_timestamp(stop_ts)
    stop_dt_str = norm_stop.isoformat() if norm_stop is not None else None
    record_position_buy(sym, entry_px, stop_px, target_px, shares, strategy="Manual / Buy", bought_date=bought_date, stop_date=stop_dt_str)

    cur_px = float(df_4h["close"].iloc[-1]) if not df_4h.empty else entry_px
    chart_buf = await asyncio.to_thread(
        generate_trade_candlestick_chart,
        sym,
        df_4h,
        entry_px,
        stop_px,
        target_px,
        None,
        None,
        cur_px,
        tf_lbl,
        "BOUGHT POSITION (4H EXIT MONITOR ACTIVE)",
        55,
        entry_ts,
        stop_ts,
    )
    msg = (
        f"{TAG_ENTRY} Logged <b>{sym}</b> | Entry: <code>${entry_px:.2f}</code> | "
        f"Stop Loss: <code>${stop_px:.2f}</code> | Exit Target: <code>${target_px:.2f}</code> | Shares: <code>{shares:g}</code>\n"
        "⏱ 4-Hour Exit &amp; Stop-Loss monitor is now tracking this trade."
    )
    if chart_buf is not None:
        await update.message.reply_photo(photo=chart_buf, caption=msg, parse_mode="HTML")
    else:
        await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_sell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    _remember_chat(update, context)
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: <code>/sell TICKER EXIT_PRICE</code>\nExample: <code>/sell NVDA 148.00</code>",
            parse_mode="HTML",
        )
        return
    try:
        sym = context.args[0].strip().upper().replace(".", "-")
        exit_px = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Invalid price. Use: /sell NVDA 148.00")
        return

    pnl_total = record_position_close(sym, exit_px, reason="MANUAL_SELL")
    if pnl_total is not None:
        prof = get_ticker_learning_profile(sym)
        await update.message.reply_text(
            f"{TAG_SELL} Closed <b>{sym}</b> at <code>${exit_px:.2f}</code> | Realized P&amp;L: <code>{_fmt_signed_money(pnl_total)}</code>\n"
            f"🧠 <b>{sym} DNA Upgraded:</b> <code>{html.escape(format_ticker_learning_summary(prof))}</code>",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(f"No open journal entry found for <b>{sym}</b>.", parse_mode="HTML")


async def handle_direct_ticker_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles natural-text keywords ('open', 'close', 'sector', 'cross over', 'pattern')
    as well as direct ticker inputs like 'NVDA', '$NVDA', '/NVDA', or '/BRK.B'.
    """
    if not update.message or not update.message.text:
        return
    _remember_chat(update, context)

    clean_phrase = re.sub(r"\s+", " ", update.message.text.strip()).lower()

    # 1. Natural-text keyword routing (works with or without slash)
    if clean_phrase in ("open", "open position", "open positions", "holdings"):
        await cmd_open(update, context)
        return
    if clean_phrase in ("close", "closed", "close position", "close positions", "closed positions"):
        await cmd_close(update, context)
        return
    if clean_phrase in ("sector", "sectors", "sector flow", "money flow"):
        await cmd_sector(update, context)
        return
    if clean_phrase in ("cross over", "crossover", "crossovers", "ema crossover", "ema cross over", "corssover"):
        await cmd_crossover(update, context)
        return
    if clean_phrase in ("pattern", "patterns", "stock pattern", "stocks pattern", "chart pattern"):
        await cmd_pattern(update, context)
        return

    # 2. Direct ticker lookup
    raw_text = update.message.text.strip().split()[0]
    if "@" in raw_text:
        raw_text = raw_text.split("@", 1)[0]

    match = TICKER_REGEX.match(raw_text)
    if not match:
        return

    ticker = match.group(1).upper()
    if ticker in RESERVED_COMMANDS:
        return
    await update.message.reply_text(f"Analyzing <b>{html.escape(ticker)}</b>, upgrading its DNA & generating chart...", parse_mode="HTML")
    report, chart_buf, caption = await analyze_single_ticker_with_chart(ticker)
    if chart_buf is not None:
        await update.message.reply_photo(photo=chart_buf, caption=caption, parse_mode="HTML")
    await update.message.reply_text(report, parse_mode="HTML")


def main() -> None:
    parser = argparse.ArgumentParser(description="SandeepPick Stock Alert, Candlestick Chart & Per-Ticker Learner Bot")
    parser.add_argument("--token", type=str, default=None, help="Telegram Bot Token")
    args = parser.parse_args()

    token = resolve_telegram_token(args.token)
    app = (
        ApplicationBuilder()
        .token(token)
        .post_init(_post_init_scheduler)
        .post_shutdown(_post_shutdown_scheduler)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("boroden", cmd_boroden))
    app.add_handler(CommandHandler("sector", cmd_sector))
    app.add_handler(CommandHandler("open", cmd_open))
    app.add_handler(CommandHandler("close", cmd_close))
    app.add_handler(CommandHandler("crossover", cmd_crossover))
    app.add_handler(CommandHandler("pattern", cmd_pattern))
    app.add_handler(CommandHandler("learn", cmd_learn))
    app.add_handler(CommandHandler("autobuy", cmd_autobuy))
    app.add_handler(CommandHandler("regime", cmd_regime))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("buy", cmd_buy))
    app.add_handler(CommandHandler("sell", cmd_sell))

    app.add_handler(MessageHandler(filters.COMMAND, handle_direct_ticker_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_direct_ticker_message))
    app.add_error_handler(_telegram_error_handler)

    logger.info("Bot started. 1H Auto-Scan, 4H Exit Monitor & Per-Ticker Continuous Learner active...")
    app.run_polling()


if __name__ == "__main__":
    main()
