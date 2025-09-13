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

# === ИНДИКАТОРЫ ВОЛАТИЛЬНОСТИ ===

def compute_atr_custom(candles, period=14):
    """Кастомная реализация Average True Range (ATR)"""
    if len(candles) < 2:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(candles)):
        prev_candle = candles[i-1]
        curr_candle = candles[i]
        
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        tr1 = curr_candle.high - curr_candle.low
        tr2 = abs(curr_candle.high - prev_candle.close)
        tr3 = abs(curr_candle.low - prev_candle.close)
        
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # ATR = среднее значение True Range за период
    if len(true_ranges) >= period:
        return np.mean(true_ranges[-period:])
    elif len(true_ranges) > 0:
        return np.mean(true_ranges)
    else:
        return 0.0

def compute_atr(candles, period=14):
    """Стандартная реализация ATR (TA-Lib или fallback к кастомной)"""
    if TALIB_AVAILABLE and len(candles) >= period:
        try:
            # Подготавливаем данные для TA-Lib
            highs = np.array([c.high for c in candles], dtype=np.float64)
            lows = np.array([c.low for c in candles], dtype=np.float64)
            closes = np.array([c.close for c in candles], dtype=np.float64)
            
            if len(highs) >= period:
                atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
                return atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0
            else:
                return 0.0
        except Exception:
            # Fallback к кастомной реализации
            return compute_atr_custom(candles, period)
    else:
        # Fallback к кастомной реализации
        return compute_atr_custom(candles, period)

def compute_volatility_ratio(candles, atr_period=14, lookback=50):
    """Вычисление коэффициента волатильности (текущая ATR / средняя ATR)"""
    if len(candles) < lookback:
        return 1.0
    
    current_atr = compute_atr(candles, atr_period)
    
    # Вычисляем ATR для каждого периода в lookback окне
    atr_values = []
    for i in range(max(atr_period + 1, len(candles) - lookback), len(candles)):
        atr_val = compute_atr(candles[:i+1], atr_period)
        if atr_val > 0:
            atr_values.append(atr_val)
    
    if len(atr_values) == 0 or current_atr == 0:
        return 1.0
    
    avg_atr = np.mean(atr_values)
    return current_atr / avg_atr if avg_atr > 0 else 1.0

class RSIStrategyBase:
    def __init__(self, rsi_period=14, rsi_buy=30, rsi_sell=70, bb_period=20, bb_std=2, candle_minutes=5, 
                 use_custom_rsi=True, use_dual_rsi=False, use_neural_filter=False, 
                 neural_confidence_threshold=0.6, limit_order_offset=0.0001, maker_fee=0.0001, 
                 taker_fee=0.0005, use_bb_exit=False):  # 🏆 Реалистичные параметры для отложенных ордеров!
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.candle_minutes = candle_minutes
        self.use_custom_rsi = use_custom_rsi  # Использовать только кастомный RSI
        self.use_dual_rsi = use_dual_rsi      # Использовать оба RSI для сигналов
        self.use_neural_filter = use_neural_filter  # 🧠 Использовать нейронный фильтр
        self.neural_confidence_threshold = neural_confidence_threshold
        self.use_bb_exit = use_bb_exit        # 🎯 Использовать выход по средней линии Боллинджера
        
        # 🏭 Реалистичные параметры торговли
        self.limit_order_offset = limit_order_offset  # 0.01% отступ для лимитных ордеров
        self.maker_fee = maker_fee      # 0.01% комиссия мейкера 
        self.taker_fee = taker_fee      # 0.05% комиссия тейкера
        
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.last_signal = 0  # 🏭 Отслеживание предыдущего сигнала для избежания дублирования ордеров
        self.last_price = None
        self.candles = []
        self.current_candle = None
        self.current_candle_time = None
        
        # 📊 Отслеживание исполнения отложенных ордеров
        self.pending_orders = []  # [(order_type, target_price, signal, timestamp)]
        self.executed_orders = 0  # Количество исполненных ордеров
        self.missed_orders = 0    # Количество неисполненных ордеров (цена не дошла)
        
        # Массивы для хранения значений индикаторов
        self.rsi_values = []           # Основной RSI (TA-Lib или кастомный)
        self.rsi_custom_values = []    # Кастомный RSI (если используется dual mode)
        self.bb_values = []
        self.atr_values = []           # 📊 Значения ATR (волатильность)
        self.volatility_ratios = []    # 📈 Коэффициенты волатильности
        
        self.entry_points = []  # (datetime, цена)
        self.exit_points = []   # (datetime, цена)
        self.equity = 1.0
        self.equity_curve = []
        self.trades = []
        
        # Оптимизация работы с данными
        self.cached_closes = []
        self.last_candle_count = 0
        
        # 🧠 Нейронный фильтр
        self.neural_filter = None
        if use_neural_filter:
            try:
                from neural_filter import NeuralSignalFilter
                self.neural_filter = NeuralSignalFilter()
                print("🧠 Нейронный фильтр загружен")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить нейронный фильтр: {e}")
                self.use_neural_filter = False
        
        # Информация о используемых индикаторах
        neural_info = " + 🧠 Neural Filter" if use_neural_filter else ""
        atr_type = "TA-Lib" if TALIB_AVAILABLE else "Custom"
        
        if use_dual_rsi:
            print(f"📊 🏆 Dual RSI Strategy: TA-Lib Wilder's + Custom SMA-based + {atr_type} ATR{neural_info}")
        elif use_custom_rsi:
            print(f"📊 🏆 AI-Enhanced Strategy: Custom SMA-based RSI + TA-Lib Bollinger Bands + {atr_type} ATR{neural_info}")
        else:
            rsi_type = "TA-Lib Wilder's" if TALIB_AVAILABLE else "Custom SMA-based (fallback)"
            print(f"📊 Standard Strategy: {rsi_type} RSI + TA-Lib Bollinger Bands + {atr_type} ATR{neural_info}")

    def dt_to_candle_start(self, dt):
        discard = timedelta(minutes=dt.minute % self.candle_minutes,
                            seconds=dt.second,
                            microseconds=dt.microsecond)
        return dt - discard

    def check_pending_orders(self, current_price, current_dt):
        """🏭 Проверяем исполнение отложенных ордеров"""
        executed_orders = []
        
        for i, (order_type, target_price, signal, timestamp) in enumerate(self.pending_orders):
            # Проверяем, достигла ли цена уровня исполнения
            order_executed = False
            
            if order_type == "buy_limit" and current_price <= target_price:
                # Лимитный ордер на покупку исполняется, когда цена опускается до уровня или ниже
                order_executed = True
            elif order_type == "sell_limit" and current_price >= target_price:
                # Лимитный ордер на продажу исполняется, когда цена поднимается до уровня или выше
                order_executed = True
            
            if order_executed:
                executed_orders.append(i)
                self.executed_orders += 1
                
                # Исполняем сделку
                if self.position == 1 and signal == 0:
                    # Закрываем лонг
                    pnl_before_fees = (target_price - self.last_price) / self.last_price
                    net_pnl = pnl_before_fees - self.maker_fee
                    self.equity *= (1 + net_pnl)
                    self.trades.append(self.equity)
                    
                elif self.position == -1 and signal == 0:
                    # Закрываем шорт
                    pnl_before_fees = (self.last_price - target_price) / self.last_price
                    net_pnl = pnl_before_fees - self.maker_fee
                    self.equity *= (1 + net_pnl)
                    self.trades.append(self.equity)
                
                elif signal == 1:
                    # Открываем лонг
                    self.last_price = target_price
                elif signal == -1:
                    # Открываем шорт
                    self.last_price = target_price
                
                self.position = signal
        
        # Удаляем исполненные ордера (в обратном порядке, чтобы не сбить индексы)
        for i in reversed(executed_orders):
            del self.pending_orders[i]
    
    def on_tick(self, price, dt, volume=0):
        # 🏭 Проверяем исполнение отложенных ордеров
        self.check_pending_orders(price, dt)
        
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
        
        # 📊 Вычисляем индикаторы волатильности
        atr = compute_atr(self.candles + [self.current_candle], period=14)
        volatility_ratio = compute_volatility_ratio(self.candles + [self.current_candle], atr_period=14, lookback=50)
        
        # Сохраняем значения только при закрытии свечи
        if candle_closed:
            self.rsi_values.append(rsi)
            self.bb_values.append((ma, upper, lower))
            self.atr_values.append(atr)
            self.volatility_ratios.append(volatility_ratio)
            if self.use_dual_rsi:
                self.rsi_custom_values.append(rsi_custom)
        elif len(self.rsi_values) == len(self.candles):
            # Для текущей свечи - обновляем последнее значение
            self.rsi_values.append(rsi)
            self.bb_values.append((ma, upper, lower))
            self.atr_values.append(atr)
            self.volatility_ratios.append(volatility_ratio)
            if self.use_dual_rsi:
                self.rsi_custom_values.append(rsi_custom)
        # --- Сигналы ---
        signal = self.position
        candle_dt = self.current_candle.start_time
        candle_close = self.current_candle.close
        
        # 🧠 Нейронная фильтрация сигналов
        neural_approved = True
        neural_confidence = 0.5
        
        if self.use_neural_filter and self.neural_filter and len(self.rsi_values) >= 20:
            try:
                # Подготавливаем признаки для нейронной сети
                lookback = min(20, len(self.rsi_values))
                recent_rsi = self.rsi_values[-lookback:]
                recent_bb = self.bb_values[-lookback:]
                recent_atr = self.atr_values[-lookback:]
                recent_vol_ratio = self.volatility_ratios[-lookback:]
                recent_prices = [c.close for c in self.candles[-lookback:]]
                
                features = self.neural_filter.prepare_features(
                    recent_rsi, recent_bb, recent_atr, recent_vol_ratio, recent_prices
                )
                
                if features is not None:
                    neural_approved, neural_confidence = self.neural_filter.should_trade(
                        features, self.neural_confidence_threshold
                    )
            except Exception as e:
                print(f"⚠️ Ошибка нейронного фильтра: {e}")
                neural_approved = True  # Fallback к обычной логике
        
        # Логика для лонгов (с нейронной фильтрацией)
        if rsi < self.rsi_buy and self.position == 0 and neural_approved:
            signal = 1  # открыть лонг
            self.entry_points.append((candle_dt, candle_close))
        elif self.position == 1:
            # 🎯 Выбор условия выхода из лонга
            if self.use_bb_exit:
                # Выход по средней линии Боллинджера
                if len(self.bb_values) > 0:
                    bb_ma, bb_upper, bb_lower = self.bb_values[-1]
                    if bb_ma is not None and candle_close <= bb_ma:
                        signal = 0  # закрыть лонг
                        self.exit_points.append((candle_dt, candle_close))
            else:
                # Стандартный выход по RSI
                if rsi > self.rsi_sell:
                    signal = 0  # закрыть лонг
                    self.exit_points.append((candle_dt, candle_close))
            
        # Логика для шортов (с нейронной фильтрацией)
        elif rsi > self.rsi_sell and self.position == 0 and neural_approved:
            signal = -1  # открыть шорт
            self.entry_points.append((candle_dt, candle_close))
        elif self.position == -1:
            # 🎯 Выбор условия выхода из шорта
            if self.use_bb_exit:
                # Выход по средней линии Боллинджера
                if len(self.bb_values) > 0:
                    bb_ma, bb_upper, bb_lower = self.bb_values[-1]
                    if bb_ma is not None and candle_close >= bb_ma:
                        signal = 0  # закрыть шорт
                        self.exit_points.append((candle_dt, candle_close))
            else:
                # Стандартный выход по RSI
                if rsi < self.rsi_buy:
                    signal = 0  # закрыть шорт
                    self.exit_points.append((candle_dt, candle_close))
        # 🏭 Создаем отложенные ордера ТОЛЬКО при изменении сигнала
        if signal != self.position and signal != self.last_signal:
            # Отменяем старые отложенные ордера (если есть)
            if self.pending_orders:
                self.missed_orders += len(self.pending_orders)
                self.pending_orders.clear()
            
            # Создаем новый отложенный ордер
            if signal == 1 and self.position == 0:
                # Открываем лонг: ставим лимитный ордер на покупку ниже рынка
                target_price = price * (1 - self.limit_order_offset)
                self.pending_orders.append(("buy_limit", target_price, signal, dt))
                
            elif signal == -1 and self.position == 0:
                # Открываем шорт: ставим лимитный ордер на продажу выше рынка
                target_price = price * (1 + self.limit_order_offset)
                self.pending_orders.append(("sell_limit", target_price, signal, dt))
                
            elif signal == 0 and self.position == 1:
                # Закрываем лонг: ставим лимитный ордер на продажу выше рынка
                target_price = price * (1 + self.limit_order_offset)
                self.pending_orders.append(("sell_limit", target_price, signal, dt))
                
            elif signal == 0 and self.position == -1:
                # Закрываем шорт: ставим лимитный ордер на покупку ниже рынка
                target_price = price * (1 - self.limit_order_offset)
                self.pending_orders.append(("buy_limit", target_price, signal, dt))
            
            # Обновляем последний сигнал
            self.last_signal = signal
        # Сохраняем equity только при закрытии свечи
        if candle_closed:
            self.equity_curve.append(self.equity)
        elif len(self.equity_curve) == len(self.candles):
            # Для текущей свечи - обновляем последнее значение
            self.equity_curve.append(self.equity)
        
        # Возвращаем текущий сигнал для торгового бота
        return signal

    def on_finish(self, price):
        # 🏭 Отменяем все неисполненные отложенные ордера
        if self.pending_orders:
            self.missed_orders += len(self.pending_orders)
            self.pending_orders.clear()
            
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
    
    def get_realistic_stats(self):
        """📊 Получить реалистичную статистику с учетом комиссий и отступов"""
        total_orders = self.executed_orders + self.missed_orders
        execution_rate = (self.executed_orders / total_orders * 100) if total_orders > 0 else 0
        
        stats = {
            'total_trades': len(self.trades),
            'final_equity': self.equity,
            'total_return_pct': (self.equity - 1.0) * 100,
            'sharpe_ratio': self.sharpe(),
            'limit_order_offset_pct': self.limit_order_offset * 100,
            'maker_fee_pct': self.maker_fee * 100,
            'taker_fee_pct': self.taker_fee * 100,
            
            # 🏭 Статистика исполнения отложенных ордеров
            'total_orders': total_orders,
            'executed_orders': self.executed_orders,
            'missed_orders': self.missed_orders,
            'execution_rate_pct': execution_rate,
        }
        
        if len(self.trades) > 0:
            # Оценка общих расходов на комиссии
            total_fees = len(self.trades) * self.maker_fee * 2  # Вход + выход
            stats['total_fees_pct'] = total_fees * 100
            stats['avg_fee_per_trade_pct'] = (total_fees / len(self.trades)) * 100
        
        return stats 