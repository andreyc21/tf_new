#!/usr/bin/env python3
"""
Bybit Microstructure Scanner — BTCUSDT (configurable)
Author: ChatGPT
Language: English (per user's preference for scripts)
Description:
  - Connects to Bybit v5 public WebSocket (linear + spot) to track:
      * Trades (signed volume → cumulative delta)
      * Orderbook 50 levels (depth within ±0.5% of mid, imbalance)
      * Tickers (last, indexPrice → basis; markPrice)
      * Liquidations (aggregate USD)
  - Polls Open Interest via REST (every 60s) for linear perp
  - Builds 1‑minute bars, rolling stats, and emits alerts for:
      * Compression (flat price + rising OI)
      * Thin book (bid depth vs rolling median)
      * Ignition down (zscore(1m ret) < -3, delta<0, basis drop)
      * Cascade (liq spike + OI move classification)
  - Optional Telegram alerts via env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - Optional strategy hooks to place orders are stubbed (print-only).

DISCLAIMER: This is research tooling. Use at your own risk. No guarantee of accuracy.
"""

import asyncio
import json
import math
import os
import signal
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

import aiohttp

# ----------------------------- Config -----------------------------

DEFAULT_CONFIG = {
    "symbol": "BTCUSDT",
    "linear_ws_url": "wss://stream.bybit.com/v5/public/linear",
    "spot_ws_url": "wss://stream.bybit.com/v5/public/spot",
    "oi_rest_url": "https://api.bybit.com/v5/market/open-interest",
    "poll_oi_sec": 60,
    "orderbook_depth_levels": 50,
    "depth_band_pct": 0.005,         # ±0.5%
    "rolling_minutes": 60,           # for z-scores / medians
    "alerts": {
        "compression_sigma": 0.2,    # |ret_15m| < 0.2σ
        "compression_oi_sigma": 2.0, # OI Δ15m > +2σ (vs its 60m changes)
        "thin_book_ratio": 0.4,      # bid_depth / median_30m < 0.4
        "ignition_zscore": -3.0,     # zscore(ret_1m) < -3
        "cascade_liq_usd_5m": 5_000_000,  # tweak per venue
    },
    "levels": {
        # Optional keyed levels to watch (manual supports/resistances)
        # Example: "L1": 115000, "L2": 112000
    },
    "log_every_sec": 10,
    "telegram": {
        "enabled": True,
        "bot_token_env": "TELEGRAM_BOT_TOKEN",
        "chat_id_env": "TELEGRAM_CHAT_ID"
    }
}

# ----------------------------- Utils -----------------------------

def zscore(series: List[float]) -> float:
    if len(series) < 10:
        return 0.0
    mean = sum(series)/len(series)
    var = sum((x - mean)**2 for x in series)/len(series)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (series[-1] - mean) / std

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b

def now_ms() -> int:
    return int(time.time() * 1000)

def floor_minute(ts_ms: int) -> int:
    return (ts_ms // 60000) * 60000

async def telegram_send(session: aiohttp.ClientSession, text: str, cfg) -> None:
    if not cfg["telegram"]["enabled"]:
        return
    token = os.getenv(cfg["telegram"]["bot_token_env"], "")
    chat_id = os.getenv(cfg["telegram"]["chat_id_env"], "")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            await resp.text()
    except Exception:
        pass

# ----------------------------- Data Structures -----------------------------

@dataclass
class TickerState:
    last: Optional[float] = None
    index_price: Optional[float] = None
    mark_price: Optional[float] = None

class OrderBook:
    """Simple order book maintaining price→size dicts."""
    def __init__(self):
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

    def update_snapshot(self, bids: List[List[str]], asks: List[List[str]]):
        self.bids = {float(p): float(s) for p, s in bids if float(s) > 0}
        self.asks = {float(p): float(s) for p, s in asks if float(s) > 0}

    def update_delta(self, bids: List[List[str]], asks: List[List[str]]):
        for p, s in bids:
            p, s = float(p), float(s)
            if s == 0 or s == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = s
        for p, s in asks:
            p, s = float(p), float(s)
            if s == 0 or s == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = s

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        bb = max(self.bids.keys()) if self.bids else None
        ba = min(self.asks.keys()) if self.asks else None
        return bb, ba

    def depth_within(self, mid: float, band_pct: float) -> Tuple[float, float]:
        """Sum size within ±band_pct around mid; return (bid_depth, ask_depth)."""
        if not self.bids or not self.asks or not mid:
            return 0.0, 0.0
        min_bid_p = mid * (1 - band_pct)
        max_ask_p = mid * (1 + band_pct)
        bid_depth = sum(s for p, s in self.bids.items() if p >= min_bid_p)
        ask_depth = sum(s for p, s in self.asks.items() if p <= max_ask_p)
        return bid_depth, ask_depth

class MinuteBarBuilder:
    """Aggregates 1-min OHLCV and signed delta."""
    def __init__(self, window: int = 60):
        self.window = window
        self.bars = deque(maxlen=window)  # each = dict with keys
        self.current_ts = None
        self.current = None

    def on_trade(self, ts_ms: int, price: float, size: float, side: str):
        minute = floor_minute(ts_ms)
        if self.current_ts != minute:
            if self.current is not None:
                self.bars.append(self.current)
            self.current_ts = minute
            self.current = {
                "ts": minute,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "delta": 0.0,   # buy - sell
            }
        self.current["high"] = max(self.current["high"], price)
        self.current["low"] = min(self.current["low"], price)
        self.current["close"] = price
        self.current["volume"] += size
        if side.upper().startswith("B"):
            self.current["delta"] += size
        else:
            self.current["delta"] -= size

    def get_returns(self) -> List[float]:
        closes = [b["close"] for b in self.bars][-30:]  # last 30 mins
        if len(closes) < 2:
            return []
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i-1]
            curr = closes[i]
            if prev > 0:
                rets.append((curr - prev) / prev)
        return rets

    def get_last_close(self) -> Optional[float]:
        return self.bars[-1]["close"] if self.bars else None

    def get_last_delta(self) -> float:
        return self.bars[-1]["delta"] if self.bars else 0.0

# ----------------------------- Scanner -----------------------------

class Scanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.depth_band = cfg["depth_band_pct"]

        self.linear_ws_url = cfg["linear_ws_url"]
        self.spot_ws_url = cfg["spot_ws_url"]
        self.oi_rest_url = cfg["oi_rest_url"]

        self.ticker_linear = TickerState()
        self.ticker_spot = TickerState()
        self.ob_linear = OrderBook()

        self.mb = MinuteBarBuilder(window=self.cfg["rolling_minutes"])
        self.depth_bid_window = deque(maxlen=30)  # last 30 values for median
        self.rets_window = deque(maxlen=self.cfg["rolling_minutes"])
        self.oi_window = deque(maxlen=self.cfg["rolling_minutes"])  # absolute OI
        self.oi_delta_window = deque(maxlen=self.cfg["rolling_minutes"])  # ΔOI per min

        self.liq_5m_window = deque(maxlen=5)  # liquidation USD per minute

        self.last_log_ts = 0
        self.session: Optional[aiohttp.ClientSession] = None

    # --- Metrics helpers

    def current_mid(self) -> Optional[float]:
        bb, ba = self.ob_linear.best_bid_ask()
        if bb and ba:
            return (bb + ba) / 2.0
        return None

    def basis(self) -> Optional[float]:
        if self.ticker_linear.last is not None and self.ticker_linear.index_price is not None:
            return self.ticker_linear.last - self.ticker_linear.index_price
        return None

    def depth_stats(self) -> Tuple[float, float, float]:
        mid = self.current_mid()
        bid, ask = self.ob_linear.depth_within(mid or 0.0, self.depth_band)
        total = bid + ask if (bid + ask) > 0 else 1.0
        imb = (bid - ask) / total
        return bid, ask, imb

    def zscore_1m(self) -> float:
        rets = list(self.rets_window)
        if len(rets) < 10:
            return 0.0
        mu = sum(rets) / len(rets)
        var = sum((x - mu) ** 2 for x in rets) / len(rets)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0:
            return 0.0
        return (rets[-1] - mu) / sd

    # --- Alert logic

    async def maybe_alerts(self):
        cfgA = self.cfg["alerts"]

        # 1) Compression: |ret_15m| < 0.2σ AND OI_Δ15m > +2σ
        if len(self.rets_window) >= 15 and len(self.oi_delta_window) >= 15:
            ret_15m = (1 + sum(self.rets_window[-15:])) - 1  # simple approx
            # σ of returns over rolling window
            if len(self.rets_window) >= 30:
                mu = sum(self.rets_window) / len(self.rets_window)
                var = sum((x - mu) ** 2 for x in self.rets_window) / len(self.rets_window)
                sd = math.sqrt(var) if var > 0 else 0.0
            else:
                sd = 0.0
            oi_deltas = list(self.oi_delta_window)[-15:]
            if len(self.oi_delta_window) >= 30:
                mu_oi = sum(self.oi_delta_window) / len(self.oi_delta_window)
                var_oi = sum((x - mu_oi) ** 2 for x in self.oi_delta_window) / len(self.oi_delta_window)
                sd_oi = math.sqrt(var_oi) if var_oi > 0 else 0.0
            else:
                sd_oi = 0.0

            cond_compress = (sd > 0 and abs(ret_15m) < cfgA["compression_sigma"] * sd)
            cond_oi = (sd_oi > 0 and sum(oi_deltas) > cfgA["compression_oi_sigma"] * sd_oi)
            if cond_compress and cond_oi:
                await self.alert(f"Compression signal: |ret_15m|={ret_15m:.4f}, OI_Δ15m sum={sum(oi_deltas):.3f} (> {cfgA['compression_oi_sigma']}σ).")

        # 2) Thin book: bid_depth within band compared to 30-min median
        if len(self.depth_bid_window) >= 10:
            bid_depth, _, _ = self.depth_stats()
            med = median(self.depth_bid_window) if self.depth_bid_window else 1.0
            if med > 0 and bid_depth / med < self.cfg["alerts"]["thin_book_ratio"]:
                await self.alert(f"Thin book: bid_depth {bid_depth:.0f} < {self.cfg['alerts']['thin_book_ratio']*100:.0f}% of 30m median {med:.0f}.")

        # 3) Ignition down: zscore(1m) < -3 AND delta_1m << 0 AND basis drop
        if len(self.mb.bars) >= 2:
            z1 = self.zscore_1m()
            last_delta = self.mb.get_last_delta()
            if self.ticker_linear.index_price is not None:
                basis_now = self.basis() or 0.0
                # compute basis change over last 3 mins
                # store a small window
                if not hasattr(self, "_basis_hist"):
                    self._basis_hist = deque(maxlen=10)
                self._basis_hist.append(basis_now)
                basis_drop = len(self._basis_hist) >= 2 and (self._basis_hist[-1] < self._basis_hist[-2])
            else:
                basis_drop = False

            if (z1 < self.cfg["alerts"]["ignition_zscore"]) and (last_delta < 0) and basis_drop:
                await self.alert(f"Ignition down: z1m={z1:.2f}, delta_1m={last_delta:.0f}, basis↓.")

        # 4) Cascade: liquidation spike (5m) above threshold; classify by OI delta
        if len(self.liq_5m_window) >= 5 and len(self.oi_delta_window) >= 5:
            liq_5m = sum(self.liq_5m_window)
            if liq_5m > self.cfg["alerts"]["cascade_liq_usd_5m"]:
                oi_5m = sum(list(self.oi_delta_window)[-5:])
                kind = "capitulation (OI↓)" if oi_5m < 0 else "leveraging (OI↑)"
                await self.alert(f"Cascade: liquidations ≈ ${liq_5m:,.0f} over 5m, OI_Δ5m={oi_5m:.3f} → {kind}.")

    async def alert(self, text: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] ALERT {self.symbol}: {text}"
        print(line, flush=True)
        if self.session:
            await telegram_send(self.session, line, self.cfg)

    # --- Network

    async def run(self):
        print("Starting scanner…", flush=True)
        async with aiohttp.ClientSession() as session:
            self.session = session
            tasks = [
                asyncio.create_task(self.ws_linear()),
                asyncio.create_task(self.ws_spot()),
                asyncio.create_task(self.poll_open_interest()),
                asyncio.create_task(self.periodic_logger())
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()

    async def ws_linear(self):
        """Connect to linear public WS: orderbook, trades, tickers, liquidation."""
        subs = [
            f"orderbook.{self.cfg['orderbook_depth_levels']}.{self.symbol}",
            f"publicTrade.{self.symbol}",
            f"tickers.{self.symbol}",
            f"liquidation.{self.symbol}",
        ]
        await self._ws_consume(self.linear_ws_url, subs, is_linear=True)

    async def ws_spot(self):
        """Connect to spot public WS: tickers only for spot last (optional)."""
        subs = [f"tickers.{self.symbol}"]
        await self._ws_consume(self.spot_ws_url, subs, is_linear=False)

    async def _ws_consume(self, url: str, subs: List[str], is_linear: bool):
        while True:
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.ws_connect(url, heartbeat=20) as ws:
                        await ws.send_json({"op": "subscribe", "args": subs})
                        print(f"Subscribed to {subs} on {url}", flush=True)
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._on_ws_message(msg.json(), is_linear)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
            except Exception as e:
                print(f"WS error ({url}): {e}. Reconnecting in 3s…", flush=True)
                await asyncio.sleep(3)

    async def _on_ws_message(self, data: dict, is_linear: bool):
        if "topic" not in data:
            return
        topic = data["topic"]
        # Orderbook
        if topic.startswith("orderbook.") and is_linear:
            t = data.get("type", "")
            if t == "snapshot":
                bid = data["data"]["b"] if "data" in data else data["b"]
                ask = data["data"]["a"] if "data" in data else data["a"]
                self.ob_linear.update_snapshot(bid, ask)
            elif t == "delta":
                bid = data["data"]["b"]
                ask = data["data"]["a"]
                self.ob_linear.update_delta(bid, ask)

            mid = self.current_mid()
            if mid:
                bid_depth, _, _ = self.depth_stats()
                self.depth_bid_window.append(bid_depth)

        # Trades (for 1m bars + delta)
        elif topic.startswith("publicTrade.") and is_linear:
            for tr in data.get("data", []):
                price = float(tr["p"])
                size = float(tr.get("v", tr.get("q", 0)))
                side = tr.get("S", tr.get("s", "Buy"))
                ts_ms = int(tr.get("T", now_ms()))
                self.mb.on_trade(ts_ms, price, size, side)

                # Add return to rets window once per minute (when bar closes)
                # We approximate by updating every trade but only when new minute appended
                if len(self.mb.bars) >= 2 and self.mb.bars[-1]["ts"] != self.mb.bars[-2]["ts"]:
                    rets = self.mb.get_returns()
                    if rets:
                        self.rets_window.clear()
                        self.rets_window.extend(rets)

        # Tickers (last, mark, index)
        elif topic.startswith("tickers."):
            # Some fields are strings
            d = data.get("data", {})
            last = float(d.get("lastPrice", d.get("lastPrice", 0)) or 0)
            mark = float(d.get("markPrice", d.get("markPrice", 0)) or 0)
            indexp = float(d.get("indexPrice", d.get("indexPrice", 0)) or 0)
            if is_linear:
                self.ticker_linear.last = last or self.ticker_linear.last
                self.ticker_linear.mark_price = mark or self.ticker_linear.mark_price
                self.ticker_linear.index_price = indexp or self.ticker_linear.index_price
            else:
                self.ticker_spot.last = last or self.ticker_spot.last

        # Liquidations
        elif topic.startswith("liquidation.") and is_linear:
            # Aggregate notional per minute
            minute = floor_minute(now_ms())
            if not hasattr(self, "_liq_bucket_ts"):
                self._liq_bucket_ts = minute
                self._liq_bucket_usd = 0.0
            # sum abs qty * price
            for liq in data.get("data", []):
                price = float(liq.get("price", 0) or liq.get("p", 0) or 0.0)
                qty = float(liq.get("qty", 0) or liq.get("q", 0) or 0.0)
                self._liq_bucket_usd += abs(price * qty)
            if minute != self._liq_bucket_ts:
                # roll
                self.liq_5m_window.append(self._liq_bucket_usd)
                self._liq_bucket_usd = 0.0
                self._liq_bucket_ts = minute

        # Periodically evaluate alerts
        await self.maybe_alerts()

    async def poll_open_interest(self):
        """Poll OI every poll_oi_sec; uses v5 market open-interest (linear)."""
        params = {"category": "linear", "symbol": self.symbol, "intervalTime": "1"}
        # Note: Bybit API param names can vary by version; adjust if needed.
        last_oi = None
        while True:
            try:
                async with self.session.get(self.oi_rest_url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    js = await resp.json()
                    # Expecting data like {"result":{"list":[{"openInterest":"1234.5", "timestamp":"..."}]}}; get most recent
                    oi = None
                    try:
                        arr = js.get("result", {}).get("list", [])
                        if arr:
                            # take latest item
                            latest = sorted(arr, key=lambda x: int(x.get("timestamp", 0)))[-1]
                            oi = float(latest.get("openInterest", latest.get("open_interest", 0)))
                    except Exception:
                        pass
                    if oi is not None:
                        self.oi_window.append(oi)
                        if last_oi is not None:
                            self.oi_delta_window.append(oi - last_oi)
                        last_oi = oi
            except Exception as e:
                print(f"OI poll error: {e}", flush=True)
            await asyncio.sleep(self.cfg["poll_oi_sec"])

    async def periodic_logger(self):
        while True:
            await asyncio.sleep(self.cfg["log_every_sec"])
            ts = time.strftime("%H:%M:%S", time.localtime())
            last = self.ticker_linear.last
            idx = self.ticker_linear.index_price
            basis = (last - idx) if (last and idx) else None
            bid_depth, ask_depth, imb = self.depth_stats()
            z1 = self.zscore_1m()
            print(f"[{ts}] {self.symbol} last={last} idx={idx} basis={basis} "
                  f"bidDepth±0.5%={bid_depth:.0f} askDepth±0.5%={ask_depth:.0f} imb={imb:.2f} z1m={z1:.2f}", flush=True)

# ----------------------------- Entry -----------------------------

def load_config(path: str) -> dict:
    import yaml
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # shallow-merge with DEFAULT_CONFIG
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k, None), dict):
                    merge(a[k], v)
                else:
                    a.setdefault(k, v)
        merge(cfg, DEFAULT_CONFIG)
        return cfg
    return DEFAULT_CONFIG

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bybit microstructure scanner")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scanner = Scanner(cfg)

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            pass

    await scanner.run()

_shutdown_called = False
async def shutdown():
    global _shutdown_called
    if _shutdown_called:
        return
    _shutdown_called = True
    print("Shutting down…", flush=True)
    await asyncio.sleep(0.2)
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

