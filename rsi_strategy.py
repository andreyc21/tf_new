import numpy as np
from datetime import datetime, timedelta

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib не установлен. Используется кастомная реализация RSI.")

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

# === КАСТОМНЫЕ ИНДИКАТОРЫ (наша реализация) ===

def compute_rsi_custom(prices, period=14):
    """Кастомная реализация RSI с простым средним (SMA-based)
    
    Эта версия показывала хорошие результаты в бэктестах.
    Использует простое среднее для расчёта средних прибылей/убытков.
    """
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

def compute_bollinger_bands_custom(prices, period=20, num_std=2):
    """Кастомная реализация Bollinger Bands"""
    prices = np.array(prices)
    if len(prices) < period:
        return None, None, None
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

# === СТАНДАРТНЫЕ ИНДИКАТОРЫ (TA-Lib или fallback) ===

def compute_rsi(prices, period=14):
    """Стандартная реализация RSI (TA-Lib или fallback к кастомной)"""
    if TALIB_AVAILABLE:
        prices_array = np.array(prices, dtype=np.float64)
        if len(prices_array) < period + 1:
            return 50.0
        rsi_values = talib.RSI(prices_array, timeperiod=period)
        return rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
    else:
        # Fallback к кастомной реализации
        return compute_rsi_custom(prices, period)

def compute_bollinger_bands(prices, period=20, num_std=2):
    """Стандартная реализация Bollinger Bands (TA-Lib или fallback)"""
    if TALIB_AVAILABLE:
        prices_array = np.array(prices, dtype=np.float64)
        if len(prices_array) < period:
            return None, None, None
        upper, middle, lower = talib.BBANDS(prices_array, timeperiod=period, nbdevup=num_std, nbdevdn=num_std, matype=0)
        if np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1]):
            return None, None, None
        return middle[-1], upper[-1], lower[-1]
    else:
        # Fallback к кастомной реализации
        return compute_bollinger_bands_custom(prices, period, num_std)

class RSIStrategyBase:
    def __init__(self, rsi_period=14, rsi_buy=30, rsi_sell=70, bb_period=20, bb_std=2, candle_minutes=5, 
                 use_custom_rsi=True, use_dual_rsi=False):  # 🏆 По умолчанию используем выигрышную стратегию!
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.candle_minutes = candle_minutes
        self.use_custom_rsi = use_custom_rsi  # Использовать только кастомный RSI
        self.use_dual_rsi = use_dual_rsi      # Использовать оба RSI для сигналов
        
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.last_price = None
        self.candles = []
        self.current_candle = None
        self.current_candle_time = None
        
        # Массивы для хранения значений индикаторов
        self.rsi_values = []           # Основной RSI (TA-Lib или кастомный)
        self.rsi_custom_values = []    # Кастомный RSI (если используется dual mode)
        self.bb_values = []
        
        self.entry_points = []  # (datetime, цена)
        self.exit_points = []   # (datetime, цена)
        self.equity = 1.0
        self.equity_curve = []
        self.trades = []
        
        # Оптимизация работы с данными
        self.cached_closes = []
        self.last_candle_count = 0
        
        # Информация о используемых индикаторах
        if use_dual_rsi:
            print(f"📊 🏆 Dual RSI Strategy: TA-Lib Wilder's + Custom SMA-based")
        elif use_custom_rsi:
            print(f"📊 🏆 Optimized Strategy: Custom SMA-based RSI + TA-Lib Bollinger Bands")
        else:
            rsi_type = "TA-Lib Wilder's" if TALIB_AVAILABLE else "Custom SMA-based (fallback)"
            print(f"📊 Standard Strategy: {rsi_type} RSI + TA-Lib Bollinger Bands")

    def dt_to_candle_start(self, dt):
        discard = timedelta(minutes=dt.minute % self.candle_minutes,
                            seconds=dt.second,
                            microseconds=dt.microsecond)
        return dt - discard

    def on_tick(self, price, dt, volume=0):
        # --- Свечи ---
        candle_time = self.dt_to_candle_start(dt)
        candle_closed = False
        
        if self.current_candle is None or candle_time != self.current_candle_time:
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
                candle_closed = True
            self.current_candle = Candle(candle_time)
            self.current_candle_time = candle_time
        self.current_candle.add_tick(price, volume)
        
        # --- Оптимизированный расчет индикаторов ---
        current_candle_count = len(self.candles)
        
        # Обновляем кэшированный массив closes только при необходимости
        if candle_closed or current_candle_count != self.last_candle_count:
            self.cached_closes = [c.close for c in self.candles]
            self.last_candle_count = current_candle_count
        
        # Добавляем текущую цену для расчетов
        closes_with_current = self.cached_closes + [self.current_candle.close]
        
        # 🏆 ОПТИМИЗИРОВАННАЯ СТРАТЕГИЯ:
        # - RSI: используем нашу выигрышную кастомную реализацию (SMA-based)  
        # - Bollinger Bands: используем TA-Lib (быстрее, результат тот же)
        
        if self.use_custom_rsi:
            # Используем только кастомный RSI (выигрышная стратегия!)
            rsi = compute_rsi_custom(closes_with_current, period=self.rsi_period)
            rsi_custom = rsi  # Для совместимости
        elif self.use_dual_rsi:
            # Используем оба варианта RSI для сравнения
            rsi = compute_rsi(closes_with_current, period=self.rsi_period)  # TA-Lib
            rsi_custom = compute_rsi_custom(closes_with_current, period=self.rsi_period)  # Кастомный
        else:
            # Fallback к стандартному RSI (TA-Lib или кастомный)
            rsi = compute_rsi(closes_with_current, period=self.rsi_period)
            rsi_custom = rsi  # Для совместимости
        
        # Bollinger Bands всегда через TA-Lib (если доступен) - быстрее и результат тот же
        ma, upper, lower = compute_bollinger_bands(closes_with_current, period=self.bb_period, num_std=self.bb_std)
        
        # Сохраняем значения только при закрытии свечи
        if candle_closed:
            self.rsi_values.append(rsi)
            self.bb_values.append((ma, upper, lower))
            if self.use_dual_rsi:
                self.rsi_custom_values.append(rsi_custom)
        elif len(self.rsi_values) == len(self.candles):
            # Для текущей свечи - обновляем последнее значение
            self.rsi_values.append(rsi)
            self.bb_values.append((ma, upper, lower))
            if self.use_dual_rsi:
                self.rsi_custom_values.append(rsi_custom)
        # --- Сигналы ---
        signal = self.position
        candle_dt = self.current_candle.start_time
        candle_close = self.current_candle.close
        
        # Логика для лонгов
        if rsi < self.rsi_buy and self.position == 0:
            signal = 1  # открыть лонг
            self.entry_points.append((candle_dt, candle_close))
        elif rsi > self.rsi_sell and self.position == 1:
            signal = 0  # закрыть лонг
            self.exit_points.append((candle_dt, candle_close))
            
        # Логика для шортов
        elif rsi > self.rsi_sell and self.position == 0:
            signal = -1  # открыть шорт
            self.entry_points.append((candle_dt, candle_close))
        elif rsi < self.rsi_buy and self.position == -1:
            signal = 0  # закрыть шорт
            self.exit_points.append((candle_dt, candle_close))
        # Управление позицией (эмулируем сделки для оффлайн-теста)
        if signal != self.position:
            # Закрываем предыдущую позицию и считаем PnL
            if self.position == 1 and self.last_price is not None:
                # Закрываем лонг
                pnl = (price - self.last_price) / self.last_price
                self.equity *= (1 + pnl)
                self.trades.append(self.equity)
            elif self.position == -1 and self.last_price is not None:
                # Закрываем шорт (обратный расчет PnL)
                pnl = (self.last_price - price) / self.last_price
                self.equity *= (1 + pnl)
                self.trades.append(self.equity)
            
            # Открываем новую позицию
            if signal == 1 or signal == -1:
                self.last_price = price
            
            self.position = signal
        # Сохраняем equity только при закрытии свечи
        if candle_closed:
            self.equity_curve.append(self.equity)
        elif len(self.equity_curve) == len(self.candles):
            # Для текущей свечи - обновляем последнее значение
            self.equity_curve.append(self.equity)
        
        # Возвращаем текущий сигнал для торгового бота
        return signal

    def on_finish(self, price):
        if self.current_candle is not None:
            self.candles.append(self.current_candle)
            
            # Добавляем финальные RSI/BB значения для последней свечи
            closes = [c.close for c in self.candles]
            
            # Вычисляем финальные значения индикаторов (используем ту же логику что и в on_tick)
            if self.use_custom_rsi:
                # Финальный расчет с выигрышной кастомной реализацией RSI
                rsi = compute_rsi_custom(closes, period=self.rsi_period)
                rsi_custom = rsi
            elif self.use_dual_rsi:
                rsi = compute_rsi(closes, period=self.rsi_period)
                rsi_custom = compute_rsi_custom(closes, period=self.rsi_period)
            else:
                rsi = compute_rsi(closes, period=self.rsi_period)
                rsi_custom = rsi
            
            # Bollinger Bands через TA-Lib (быстрее)
            ma, upper, lower = compute_bollinger_bands(closes, period=self.bb_period, num_std=self.bb_std)
            
            # Если у нас еще нет значения для последней свечи
            if len(self.rsi_values) < len(self.candles):
                self.rsi_values.append(rsi)
                self.bb_values.append((ma, upper, lower))
                if self.use_dual_rsi:
                    self.rsi_custom_values.append(rsi_custom)
        
        # Закрываем любую открытую позицию
        if self.position == 1 and self.last_price is not None:
            # Закрываем лонг
            pnl = (price - self.last_price) / self.last_price
            self.equity *= (1 + pnl)
            self.trades.append(self.equity)
            self.position = 0
        elif self.position == -1 and self.last_price is not None:
            # Закрываем шорт
            pnl = (self.last_price - price) / self.last_price
            self.equity *= (1 + pnl)
            self.trades.append(self.equity)
            self.position = 0
        
        # Добавляем финальное значение equity только если его еще нет
        if len(self.equity_curve) < len(self.candles):
            self.equity_curve.append(self.equity)

    def sharpe(self):
        returns = np.diff(self.trades)
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) 