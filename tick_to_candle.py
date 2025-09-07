import gzip
import csv
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Candle:
    def __init__(self, start_time):
        self.start_time = start_time
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0.0

    def add_tick(self, price, volume):
        if self.open is None:
            self.open = price
        self.high = price if self.high is None else max(self.high, price)
        self.low = price if self.low is None else min(self.low, price)
        self.close = price
        self.volume += volume

    def to_tuple(self):
        return (self.start_time, self.open, self.high, self.low, self.close, self.volume)

    def to_dict(self):
        return {
            'start_time': self.start_time,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }

def timestamp_to_dt(ts):
    return datetime.utcfromtimestamp(int(ts) / 1000)

def dt_to_candle_start(dt, minutes=5):
    discard = timedelta(minutes=dt.minute % minutes,
                       seconds=dt.second,
                       microseconds=dt.microsecond)
    return dt - discard

def compute_rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) < period + 1:
        return 50.0  # Недостаточно данных, нейтральное значение
    deltas = np.diff(prices[-(period+1):])
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100. - 100. / (1. + rs)
    return rsi

class RSILiveStrategy:
    def __init__(self, rsi_period=14, rsi_buy=30, rsi_sell=70, candle_minutes=5):
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.position = 0  # 1 = long, 0 = flat
        self.equity = 1.0
        self.last_price = None
        self.equity_curve = []
        self.trades = []
        self.candles = []
        self.current_candle = None
        self.current_candle_time = None
        self.candle_minutes = candle_minutes
        self.rsi_values = []
        self.entry_points = []  # (datetime, цена)
        self.exit_points = []   # (datetime, цена)

    def on_tick(self, price, dt, volume=0):
        # --- Свечи ---
        candle_time = dt_to_candle_start(dt, self.candle_minutes)
        if self.current_candle is None or candle_time != self.current_candle_time:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
            self.current_candle = Candle(candle_time)
            self.current_candle_time = candle_time
        self.current_candle.add_tick(price, volume)
        # --- RSI по закрытым свечам + текущий close ---
        closes = [c.close for c in self.candles] + [self.current_candle.close]
        rsi = compute_rsi(closes, period=self.rsi_period)
        self.rsi_values.append(rsi)
        signal = self.position
        # Для входов/выходов используем время открытия текущей свечи и её close
        candle_dt = self.current_candle.start_time
        candle_close = self.current_candle.close
        if rsi < self.rsi_buy and self.position == 0:
            signal = 1  # открыть лонг
            self.entry_points.append((candle_dt, candle_close))
        elif rsi > self.rsi_sell and self.position == 1:
            signal = 0  # закрыть лонг
            self.exit_points.append((candle_dt, candle_close))
        # Управление позицией
        if signal != self.position:
            if self.position == 1 and self.last_price is not None:
                pnl = (price - self.last_price) / self.last_price
                self.equity *= (1 + pnl)
                self.trades.append(self.equity)
            if signal == 1:
                self.last_price = price
            self.position = signal
        self.equity_curve.append(self.equity)

    def on_finish(self, price):
        if self.current_candle is not None:
            self.candles.append(self.current_candle)
        if self.position == 1 and self.last_price is not None:
            pnl = (price - self.last_price) / self.last_price
            self.equity *= (1 + pnl)
            self.trades.append(self.equity)
            self.position = 0
        self.equity_curve.append(self.equity)

    def sharpe(self):
        returns = np.diff(self.trades)
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

    def plot(self, window=100):
        total = len(self.candles)
        i = 0
        while i < total:
            candles = self.candles[i:i+window]
            candle_times = [c.start_time for c in candles]
            candle_opens = [c.open for c in candles]
            candle_highs = [c.high for c in candles]
            candle_lows = [c.low for c in candles]
            candle_closes = [c.close for c in candles]
            # Пересчитываем RSI и equity только для этого окна
            closes = candle_closes
            rsi_values = [compute_rsi(closes[:j+1], period=self.rsi_period) for j in range(len(closes))]
            equity_curve = []
            position = 0
            last_price = None
            equity = 1.0
            trades = []
            entry_points = []
            exit_points = []
            for j in range(len(closes)):
                rsi = rsi_values[j]
                signal = position
                if rsi < self.rsi_buy and position == 0:
                    signal = 1
                    entry_points.append((candle_times[j], closes[j]))
                elif rsi > self.rsi_sell and position == 1:
                    signal = 0
                    exit_points.append((candle_times[j], closes[j]))
                if signal != position:
                    if position == 1 and last_price is not None:
                        pnl = (closes[j] - last_price) / last_price
                        equity *= (1 + pnl)
                        trades.append(equity)
                    if signal == 1:
                        last_price = closes[j]
                    position = signal
                equity_curve.append(equity)
            # --- График ---
            fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
            # 1. Свечи с входами/выходами
            ax0 = axs[0]
            width = timedelta(minutes=2)
            for k in range(len(candles)):
                color = 'green' if candle_closes[k] >= candle_opens[k] else 'red'
                ax0.plot([candle_times[k], candle_times[k]], [candle_lows[k], candle_highs[k]], color=color, linewidth=1)
                ax0.add_patch(plt.Rectangle((candle_times[k] - width/2, min(candle_opens[k], candle_closes[k])),
                                            width, abs(candle_closes[k] - candle_opens[k]),
                                            color=color, alpha=0.7))
            if entry_points:
                ax0.scatter([t for t, _ in entry_points], [p for _, p in entry_points], marker='^', color='blue', label='Entry', zorder=5)
            if exit_points:
                ax0.scatter([t for t, _ in exit_points], [p for _, p in exit_points], marker='v', color='orange', label='Exit', zorder=5)
            ax0.set_ylabel('Price (5m candles)')
            ax0.set_title(f'5m Candles с входами/выходами ({i+1}-{i+len(candles)})')
            ax0.legend()
            ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 2. RSI
            axs[1].plot(candle_times, rsi_values, label='RSI(14)', color='orange')
            axs[1].axhline(30, color='green', linestyle='--', alpha=0.5)
            axs[1].axhline(70, color='red', linestyle='--', alpha=0.5)
            axs[1].set_ylabel('RSI')
            axs[1].legend()
            axs[1].set_title('RSI(14) (по свечам)')
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # 3. Equity
            axs[2].plot(candle_times, equity_curve, label='Equity Curve', color='green')
            axs[2].set_ylabel('Equity')
            axs[2].set_xlabel('Time')
            axs[2].legend()
            axs[2].set_title('Equity Curve (по свечам)')
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.tight_layout()
            plt.show()
            i += window

def process_ticks_rsi_live_strategy(filename, rsi_period=14, rsi_buy=30, rsi_sell=70, candle_minutes=5, plot=True):
    strategy = RSILiveStrategy(rsi_period, rsi_buy, rsi_sell, candle_minutes)
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            strategy.on_tick(price, dt, volume)
        strategy.on_finish(price)
    print(f'Sharpe: {strategy.sharpe():.4f}')
    print(f'Equity: {strategy.equity:.4f}')
    if plot:
        strategy.plot()
    return strategy

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    print('--- Тест RSI-стратегии (tick, RSI по свечам+текущий close) ---')
    process_ticks_rsi_live_strategy(filename) 