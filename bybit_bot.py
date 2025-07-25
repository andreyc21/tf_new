import os
import time
from datetime import datetime, timedelta, timezone
from pybit.unified_trading import WebSocket, HTTP
import numpy as np
import uuid
import re
import logging

# === ЛОГГЕР ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("rsi-bot")

# === НАСТРОЙКИ ===
API_KEY  = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOL = 'BTCUSDT'
POSITION_SIZE = 0.01  # фиксированный размер позиции (например, 0.01 BTC)
RSI_PERIOD = 14
RSI_BUY = 30
RSI_SELL = 70
CANDLE_MINUTES = 5
# Определяем testnet по переменной окружения
TESTNET = os.environ.get('TESTNET', '1') == '1'

class CandleBuilder:
    def __init__(self, minutes=5):
        self.minutes = minutes
        self.current_candle = None
        self.current_candle_time = None
        self.candles = []

    def add_tick(self, price, ts):
        dt = datetime.fromtimestamp(ts / 1000, timezone.utc)
        candle_time = dt - timedelta(minutes=dt.minute % self.minutes,
                                     seconds=dt.second,
                                     microseconds=dt.microsecond)
        if self.current_candle is None or candle_time != self.current_candle_time:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
            self.current_candle = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'start_time': candle_time
            }
            self.current_candle_time = candle_time
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price

    def get_candles(self):
        candles = self.candles.copy()
        if self.current_candle:
            candles.append(self.current_candle)
        return candles

    def preload_candles(self, candles):
        self.candles = candles.copy()
        if candles:
            self.current_candle = candles[-1]
            self.current_candle_time = candles[-1]['start_time']

# === RSI ===
def compute_rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period+1):])
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100. - 100. / (1. + rs)
    return rsi

def compute_bollinger_bands(prices, period=20, num_std=2):
    prices = np.array(prices)
    if len(prices) < period:
        return None, None, None
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

class RSIBot:
    def __init__(self, http, ws, symbol, position_size):
        self.http = http
        self.ws = ws
        self.symbol = symbol
        self.position_size = position_size
        self.candle_builder = CandleBuilder(CANDLE_MINUTES)
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.last_signal = 0
        self.last_order_time = None
        self.last_rsi_print_minute = None
        self.last_rsi = None

    def cancel_my_orders(self):
        try:
            open_orders = self.http.get_open_orders(
                category="linear",
                symbol=self.symbol
            )
            for order in open_orders['result']['list']:
                link_id = order.get('orderLinkId', '')
                if link_id.startswith('rsi-bot-'):
                    order_id = order['orderId']
                    self.http.cancel_order(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_id
                    )
                    logger.info(f"[CANCEL] Cancelled order {order_id} ({link_id})")
        except Exception as e:
            logger.warning(f"Не удалось отменить ордера: {e}")

    def on_tick(self, price, ts):
        self.candle_builder.add_tick(price, ts)
        candles = self.candle_builder.get_candles()
        closes = [c['close'] for c in candles]
        rsi = compute_rsi(closes, RSI_PERIOD)
        self.last_rsi = rsi
        ma, upper, lower = compute_bollinger_bands(closes)
        # --- Логика сигналов ---
        signal = self.position
        if rsi < RSI_BUY and self.position <= 0:
            signal = 1  # лонг
        elif rsi > RSI_SELL and self.position >= 0:
            signal = -1  # шорт
        # --- Торговля ---
        if signal != self.position and signal != self.last_signal:
            self.cancel_my_orders()
            self.trade(signal, price)
            self.last_signal = signal
        # --- Вывод RSI, цены и BB раз в минуту ---
        dt = datetime.fromtimestamp(ts / 1000, timezone.utc)
        current_minute = dt.replace(second=0, microsecond=0)
        if self.last_rsi_print_minute is None or current_minute > self.last_rsi_print_minute:
            last_close = closes[-1] if closes else price
            bb_str = ""
            if ma is not None:
                bb_str = f" BB: lower={lower:.2f} MA={ma:.2f} upper={upper:.2f}"
            logger.info(f"[RSI] {dt.strftime('%Y-%m-%d %H:%M:%S')} RSI({RSI_PERIOD}): {rsi:.2f} Last price: {last_close}{bb_str}")
            self.last_rsi_print_minute = current_minute

    def trade(self, signal, price):
        logger.info(f"[TRADE] {datetime.fromtimestamp(time.time(), timezone.utc)} Signal: {signal}, Price: {price}")
        try:
            order_link_id = f"rsi-bot-{uuid.uuid4()}"
            offset = 0.001  # 0.1%
            # Закрыть противоположную позицию лимитным ордером
            if self.position == 1 and signal == -1:
                limit_price = round(price * (1 + offset), 2)
                self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Sell",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    reduceOnly=True,
                    orderLinkId=order_link_id
                )
            elif self.position == -1 and signal == 1:
                limit_price = round(price * (1 - offset), 2)
                self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Buy",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    reduceOnly=True,
                    orderLinkId=order_link_id
                )
            # Открыть новую позицию лимитным ордером
            if signal == 1:
                limit_price = round(price * (1 - offset), 2)
                self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Buy",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    orderLinkId=order_link_id
                )
            elif signal == -1:
                limit_price = round(price * (1 + offset), 2)
                self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Sell",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    orderLinkId=order_link_id
                )
            self.position = signal
        except Exception as e:
            logger.error(f"{e}")

# === Основной запуск ===
def main():
    http = HTTP(
        testnet=TESTNET,
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    # --- Загрузка истории свечей ---
    logger.info(f"Loading historical candles... (testnet={TESTNET})")
    klines = http.get_kline(
        category="linear",
        symbol=SYMBOL,
        interval="5",
        limit=50
    )
    # klines['result']['list'] — список свечей от новых к старым!
    raw_candles = klines['result']['list'][::-1]  # теперь от старых к новым
    preload = []
    for c in raw_candles:
        preload.append({
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'start_time': datetime.fromtimestamp(int(c[0]) / 1000, timezone.utc)
        })
    # --- Вывод баланса ---
    try:
        balance = http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        usdt_balance = balance['result']['list'][0]['totalEquity']
        logger.info(f"Current USDT balance: {usdt_balance}")
    except Exception as e:
        logger.warning(f"Не удалось получить баланс: {e}")
    # --- Получение текущей позиции ---
    try:
        positions = http.get_positions(
            category="linear",
            symbol=SYMBOL
        )
        pos_list = positions['result']['list']
        current_position = 0
        found_position = False
        for pos in pos_list:
            size = float(pos['size'])
            side = pos['side']
            if size > 0:
                found_position = True
                if side == 'Buy':
                    current_position = 1
                elif side == 'Sell':
                    current_position = -1
                logger.info(f"Open position detected: {side} size={size}")
        if not found_position:
            logger.info(f"No open positions detected.")
    except Exception as e:
        logger.warning(f"Не удалось получить позицию: {e}")
        current_position = 0
    # --- Вывод последнего RSI и цены ---
    if preload:
        closes = [c['close'] for c in preload]
        last_rsi = compute_rsi(closes, RSI_PERIOD)
        last_price = closes[-1]
        last_time = preload[-1]['start_time']
        logger.info(f"Last candle: {last_time.strftime('%Y-%m-%d %H:%M:%S')} Close: {last_price} RSI({RSI_PERIOD}): {last_rsi:.2f}")
    # --- WebSocket ---
    ws = WebSocket(
        testnet=TESTNET,
        channel_type="linear"
    )
    bot = RSIBot(http, ws, SYMBOL, POSITION_SIZE)
    bot.candle_builder.preload_candles(preload)
    bot.position = current_position
    logger.info("Bot started. Waiting for ticks...")
    def handle_message(msg):
        if 'data' in msg and isinstance(msg['data'], list):
            for trade in msg['data']:
                price = float(trade['p'])
                ts = int(trade['T'])
                bot.on_tick(price, ts)
    try:
        ws.trade_stream(
            symbol=SYMBOL,
            callback=handle_message
        )
    except Exception as e:
        logger.error(f"[WS ERROR] {e}")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main() 