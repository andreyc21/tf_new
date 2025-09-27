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
    """🚀 ОПТИМИЗИРОВАННАЯ реализация ATR с кэшированием"""
    if len(candles) < period:
        return 0.0
        
    if TALIB_AVAILABLE:
        try:
            # Подготавливаем данные для TA-Lib (оптимизированно)
            highs = np.array([c.high for c in candles], dtype=np.float64)
            lows = np.array([c.low for c in candles], dtype=np.float64)
            closes = np.array([c.close for c in candles], dtype=np.float64)
            
            atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
            return atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0
        except Exception:
            return compute_atr_custom(candles, period)
    else:
        return compute_atr_custom(candles, period)

def compute_volatility_ratio(candles, atr_period=14, lookback=50):
    """🚀 ОПТИМИЗИРОВАННОЕ вычисление коэффициента волатильности"""
    if len(candles) < lookback:
        return 1.0
    
    # Используем TA-Lib для вычисления всех ATR значений за один вызов
    if TALIB_AVAILABLE and len(candles) >= atr_period:
        try:
            # Подготавливаем данные один раз
            highs = np.array([c.high for c in candles], dtype=np.float64)
            lows = np.array([c.low for c in candles], dtype=np.float64)
            closes = np.array([c.close for c in candles], dtype=np.float64)
            
            # Вычисляем все ATR значения за один вызов
            all_atr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
            
            # Фильтруем валидные значения
            valid_atr = all_atr[~np.isnan(all_atr)]
            
            if len(valid_atr) == 0:
                return 1.0
            
            current_atr = valid_atr[-1]
            
            # Берем последние lookback значений для средней
            lookback_atr = valid_atr[-min(lookback, len(valid_atr)):]
            avg_atr = np.mean(lookback_atr)
            
            return current_atr / avg_atr if avg_atr > 0 else 1.0
            
        except Exception:
            # Fallback к старой логике
            pass
    
    # Fallback: старая логика (медленная)
    current_atr = compute_atr(candles, atr_period)
    
    # Берем только последние значения вместо пересчета всех
    start_idx = max(0, len(candles) - lookback)
    atr_values = []
    
    for i in range(start_idx, len(candles), 5):  # Каждые 5 свечей вместо каждой
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
                 taker_fee=0.0005, use_bb_exit=False, use_support_resistance=False,
                 use_stop_loss=True, stop_loss_pct=0.02, use_trailing_stop=True, 
                 trailing_stop_pct=0.015, use_atr_stop=True, atr_multiplier=2.0):  # 🏆 Реалистичные параметры для отложенных ордеров!
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
        self.use_support_resistance = use_support_resistance  # 📊 Использовать уровни S&R
        
        # 🏭 Реалистичные параметры торговли
        self.limit_order_offset = limit_order_offset  # 0.01% отступ для лимитных ордеров
        self.maker_fee = maker_fee      # 0.01% комиссия мейкера 
        self.taker_fee = taker_fee      # 0.05% комиссия тейкера
        
        # 🛡️ Параметры стоп-лоссов
        self.use_stop_loss = use_stop_loss           # Использовать стоп-лоссы
        self.stop_loss_pct = stop_loss_pct           # Процент стоп-лосса
        self.use_trailing_stop = use_trailing_stop   # Трейлинг стоп
        self.trailing_stop_pct = trailing_stop_pct   # Процент трейлинг стопа
        self.use_atr_stop = use_atr_stop             # Адаптивный стоп на основе ATR
        self.atr_multiplier = atr_multiplier         # Множитель ATR для стоп-лосса
        
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
        
        # 🛡️ Стоп-лоссы
        self.entry_price = None         # Цена входа в позицию
        self.stop_loss_price = None     # Цена стоп-лосса
        self.trailing_high = None       # Максимум для трейлинг стопа (лонг)
        self.trailing_low = None        # Минимум для трейлинг стопа (шорт)
        self.stop_loss_triggered = 0    # Счетчик сработавших стоп-лоссов
        
        # Массивы для хранения значений индикаторов
        self.rsi_values = []           # Основной RSI (TA-Lib или кастомный)
        self.rsi_custom_values = []    # Кастомный RSI (если используется dual mode)
        self.bb_values = []
        self.atr_values = []           # 📊 Значения ATR (волатильность)
        self.volatility_ratios = []    # 📈 Коэффициенты волатильности
        
        # 🚀 Кэш для оптимизации производительности
        self.atr_cache = {}            # Кэш ATR значений {length: atr_value}
        self.numpy_arrays_cache = {    # Кэш numpy массивов
            'highs': None,
            'lows': None, 
            'closes': None,
            'last_length': 0
        }
        
        # 📊 Уровни поддержки и сопротивления
        self.use_support_resistance = False  # Флаг использования S&R
        self.sr_levels = []                  # Текущие уровни S&R
        self.sr_last_update = 0              # Последнее обновление S&R
        self.sr_update_interval = 20        # Обновлять каждые N свечей
        
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
        
        # 📚 Сбор данных для обучения
        self.training_data_collector = None
        self.collect_training_data = False
        
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
    
    def get_cached_arrays(self):
        """🚀 Получить кэшированные numpy массивы"""
        current_length = len(self.candles)
        cache = self.numpy_arrays_cache
        
        # Если кэш актуален, возвращаем его
        if (cache['last_length'] == current_length and 
            cache['highs'] is not None and 
            current_length > 0):
            return cache['highs'], cache['lows'], cache['closes']
        
        # Обновляем кэш
        if current_length > 0:
            cache['highs'] = np.array([c.high for c in self.candles], dtype=np.float64)
            cache['lows'] = np.array([c.low for c in self.candles], dtype=np.float64)  
            cache['closes'] = np.array([c.close for c in self.candles], dtype=np.float64)
            cache['last_length'] = current_length
            
            return cache['highs'], cache['lows'], cache['closes']
        
        return None, None, None
    
    def get_cached_atr(self, period=14):
        """🚀 Получить кэшированное значение ATR"""
        current_length = len(self.candles)
        cache_key = f"{current_length}_{period}"
        
        # Проверяем кэш
        if cache_key in self.atr_cache:
            return self.atr_cache[cache_key]
        
        # Вычисляем и кэшируем
        if current_length >= period:
            highs, lows, closes = self.get_cached_arrays()
            if highs is not None and TALIB_AVAILABLE:
                try:
                    atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
                    atr_val = atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0
                    self.atr_cache[cache_key] = atr_val
                    
                    # Ограничиваем размер кэша
                    if len(self.atr_cache) > 1000:
                        # Удаляем старые записи
                        old_keys = [k for k in self.atr_cache.keys() 
                                   if int(k.split('_')[0]) < current_length - 100]
                        for k in old_keys:
                            del self.atr_cache[k]
                    
                    return atr_val
                except Exception:
                    pass
        
        # Fallback
        atr_val = compute_atr_custom(self.candles, period)
        self.atr_cache[cache_key] = atr_val
        return atr_val
    
    def update_support_resistance_levels(self):
        """📊 Обновление уровней поддержки и сопротивления"""
        if not self.use_support_resistance:
            return
            
        current_candle_count = len(self.candles)
        
        # Обновляем только периодически для производительности
        if (current_candle_count - self.sr_last_update >= self.sr_update_interval or 
            (len(self.sr_levels) == 0 and current_candle_count >= 50)):
            try:
                from support_resistance import SupportResistanceFinder
                
                sr_finder = SupportResistanceFinder(
                    min_touches=2, 
                    tolerance_pct=0.3,   # Увеличиваем допуск для крипты
                    lookback=min(200, current_candle_count),
                    min_strength=0.1     # Снижаем минимальную силу
                )
                
                self.sr_levels = sr_finder.find_support_resistance_levels(self.candles)
                self.sr_last_update = current_candle_count
                
            except ImportError:
                # Если модуль недоступен, отключаем S&R
                self.use_support_resistance = False
    
    def get_sr_signal_modifier(self, signal, current_price):
        """🎯 Модификация сигналов на основе уровней S&R"""
        if not self.use_support_resistance or not self.sr_levels:
            return signal, 1.0  # Без изменений
        
        from support_resistance import SupportResistanceFinder
        
        sr_finder = SupportResistanceFinder()
        nearest = sr_finder.get_nearest_levels(self.sr_levels, current_price, max_distance_pct=2.0)
        
        signal_strength = 1.0
        
        # Логика модификации сигналов
        if signal == 1:  # Сигнал на покупку
            # Усиливаем сигнал рядом с поддержкой
            if nearest['support'] and nearest['support'].strength > 0.5:
                distance_pct = abs(current_price - nearest['support'].price) / current_price * 100
                if distance_pct < 0.5:  # Очень близко к поддержке
                    signal_strength = 1.5
                elif distance_pct < 1.0:
                    signal_strength = 1.2
            
            # Ослабляем сигнал рядом с сопротивлением
            if nearest['resistance'] and nearest['resistance'].strength > 0.5:
                distance_pct = abs(nearest['resistance'].price - current_price) / current_price * 100
                if distance_pct < 0.5:  # Очень близко к сопротивлению
                    signal_strength = 0.3  # Сильно ослабляем
                elif distance_pct < 1.0:
                    signal_strength = 0.7
        
        elif signal == -1:  # Сигнал на продажу
            # Усиливаем сигнал рядом с сопротивлением
            if nearest['resistance'] and nearest['resistance'].strength > 0.5:
                distance_pct = abs(nearest['resistance'].price - current_price) / current_price * 100
                if distance_pct < 0.5:
                    signal_strength = 1.5
                elif distance_pct < 1.0:
                    signal_strength = 1.2
            
            # Ослабляем сигнал рядом с поддержкой
            if nearest['support'] and nearest['support'].strength > 0.5:
                distance_pct = abs(current_price - nearest['support'].price) / current_price * 100
                if distance_pct < 0.5:
                    signal_strength = 0.3
                elif distance_pct < 1.0:
                    signal_strength = 0.7
        
        # Применяем модификацию: если сила < 0.5, блокируем сигнал
        if signal_strength < 0.5:
            return self.position, signal_strength  # Оставляем текущую позицию
        
        return signal, signal_strength

    def calculate_stop_loss(self, entry_price, position, atr_value=None):
        """🛡️ Вычисляем цену стоп-лосса с учетом S&R уровней"""
        if not self.use_stop_loss:
            return None
        
        # Базовый расчет стоп-лосса
        if self.use_atr_stop and atr_value is not None:
            # Адаптивный стоп на основе ATR
            atr_distance = atr_value * self.atr_multiplier
            
            if position == 1:  # Лонг
                base_stop = entry_price - atr_distance
            else:  # Шорт
                base_stop = entry_price + atr_distance
        else:
            # Фиксированный процентный стоп
            if position == 1:  # Лонг
                base_stop = entry_price * (1 - self.stop_loss_pct)
            else:  # Шорт
                base_stop = entry_price * (1 + self.stop_loss_pct)
        
        # 🎯 УМНЫЕ СТОП-ЛОССЫ: корректируем с учетом S&R уровней
        if self.use_support_resistance and self.sr_levels:
            adjusted_stop = self.adjust_stop_loss_for_sr(entry_price, base_stop, position)
            return adjusted_stop
        
        return base_stop
    
    def adjust_stop_loss_for_sr(self, entry_price, base_stop, position):
        """🎯 Корректируем стоп-лосс с учетом уровней S&R"""
        try:
            from support_resistance import SupportResistanceFinder
            
            if not hasattr(self, 'sr_finder'):
                self.sr_finder = SupportResistanceFinder()
            
            if position == 1:  # Лонг позиция
                # Ищем ближайший уровень поддержки ниже базового стопа
                support_below = None
                for level in self.sr_levels:
                    if level['price'] < base_stop and level['type'] == 'support':
                        if support_below is None or level['price'] > support_below['price']:
                            support_below = level
                
                if support_below and support_below['strength'] >= 3:  # Только сильные уровни
                    # Размещаем стоп чуть ниже уровня поддержки
                    buffer = entry_price * 0.002  # 0.2% буфер
                    adjusted_stop = support_below['price'] - buffer
                    
                    # Но не слишком далеко от базового стопа (максимум в 2 раза дальше)
                    max_distance = abs(entry_price - base_stop) * 2
                    if abs(entry_price - adjusted_stop) <= max_distance:
                        print(f"🎯 Умный стоп лонг: {base_stop:.2f} -> {adjusted_stop:.2f} (за поддержкой {support_below['price']:.2f})")
                        return adjusted_stop
            
            else:  # Шорт позиция
                # Ищем ближайший уровень сопротивления выше базового стопа
                resistance_above = None
                for level in self.sr_levels:
                    if level['price'] > base_stop and level['type'] == 'resistance':
                        if resistance_above is None or level['price'] < resistance_above['price']:
                            resistance_above = level
                
                if resistance_above and resistance_above['strength'] >= 3:  # Только сильные уровни
                    # Размещаем стоп чуть выше уровня сопротивления
                    buffer = entry_price * 0.002  # 0.2% буфер
                    adjusted_stop = resistance_above['price'] + buffer
                    
                    # Но не слишком далеко от базового стопа
                    max_distance = abs(entry_price - base_stop) * 2
                    if abs(entry_price - adjusted_stop) <= max_distance:
                        print(f"🎯 Умный стоп шорт: {base_stop:.2f} -> {adjusted_stop:.2f} (за сопротивлением {resistance_above['price']:.2f})")
                        return adjusted_stop
            
        except Exception as e:
            print(f"⚠️ Ошибка корректировки стоп-лосса: {e}")
        
        # Возвращаем базовый стоп, если корректировка не удалась
        return base_stop
    
    def analyze_trade_profitability(self, entry_price, signal):
        """💰 Анализируем ожидаемую прибыльность сделки перед входом"""
        try:
            # Рассчитываем стоп-лосс для оценки риска
            current_atr = self.atr_values[-1] if len(self.atr_values) > 0 else None
            stop_loss_price = self.calculate_stop_loss(entry_price, signal, current_atr)
            
            if stop_loss_price is None:
                # Если стоп-лосс отключен, используем фиксированный риск 2%
                if signal == 1:  # Лонг
                    stop_loss_price = entry_price * 0.98
                else:  # Шорт
                    stop_loss_price = entry_price * 1.02
            
            # Рассчитываем риск (расстояние до стоп-лосса)
            risk_pct = abs(entry_price - stop_loss_price) / entry_price * 100
            
            # Оцениваем потенциальную цель на основе S&R уровней и технических индикаторов
            target_price = self.estimate_target_price(entry_price, signal)
            target_pct = abs(target_price - entry_price) / entry_price * 100
            
            # Рассчитываем соотношение риск/прибыль
            risk_reward_ratio = target_pct / risk_pct if risk_pct > 0 else 0
            
            # Оцениваем вероятность успеха на основе технических факторов
            success_probability = self.estimate_success_probability(entry_price, signal)
            
            # Ожидаемое значение сделки
            expected_value = (success_probability * target_pct) - ((1 - success_probability) * risk_pct)
            
            return {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'target_price': target_price,
                'risk_pct': risk_pct,
                'target_pct': target_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'success_probability': success_probability,
                'expected_value': expected_value
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа прибыльности: {e}")
            # Возвращаем консервативную оценку
            return {
                'entry_price': entry_price,
                'stop_loss_price': entry_price * (0.98 if signal == 1 else 1.02),
                'target_price': entry_price * (1.03 if signal == 1 else 0.97),
                'risk_pct': 2.0,
                'target_pct': 3.0,
                'risk_reward_ratio': 1.5,
                'success_probability': 0.5,
                'expected_value': 0.5
            }
    
    def estimate_target_price(self, entry_price, signal):
        """🎯 Оцениваем потенциальную цель сделки"""
        try:
            # Базовая цель на основе ATR
            current_atr = self.atr_values[-1] if len(self.atr_values) > 0 else entry_price * 0.02
            
            if signal == 1:  # Лонг
                base_target = entry_price + (current_atr * 2.5)
            else:  # Шорт
                base_target = entry_price - (current_atr * 2.5)
            
            # Корректируем цель на основе S&R уровней
            if self.use_support_resistance and self.sr_levels:
                sr_target = self.find_sr_target(entry_price, signal)
                if sr_target:
                    # Используем ближайший к базовой цели S&R уровень
                    if abs(sr_target - entry_price) > abs(base_target - entry_price) * 0.5:
                        return sr_target
            
            # Корректируем на основе Bollinger Bands
            if len(self.bb_values) > 0:
                bb_upper, bb_middle, bb_lower = self.bb_values[-1]
                
                if signal == 1 and bb_upper:  # Лонг - цель верхняя граница
                    bb_target = bb_upper
                    if bb_target > entry_price:
                        return min(bb_target, base_target * 1.5)  # Не слишком жадно
                
                elif signal == -1 and bb_lower:  # Шорт - цель нижняя граница
                    bb_target = bb_lower
                    if bb_target < entry_price:
                        return max(bb_target, base_target * 1.5)  # Не слишком жадно
            
            return base_target
            
        except Exception as e:
            print(f"⚠️ Ошибка оценки цели: {e}")
            # Консервативная цель
            return entry_price * (1.03 if signal == 1 else 0.97)
    
    def find_sr_target(self, entry_price, signal):
        """🎯 Находим ближайший S&R уровень как цель"""
        try:
            if signal == 1:  # Лонг - ищем сопротивление выше
                resistance_above = None
                for level in self.sr_levels:
                    if level['price'] > entry_price and level['type'] == 'resistance':
                        if resistance_above is None or level['price'] < resistance_above['price']:
                            resistance_above = level
                
                if resistance_above:
                    return resistance_above['price'] * 0.999  # Чуть ниже сопротивления
            
            else:  # Шорт - ищем поддержку ниже
                support_below = None
                for level in self.sr_levels:
                    if level['price'] < entry_price and level['type'] == 'support':
                        if support_below is None or level['price'] > support_below['price']:
                            support_below = level
                
                if support_below:
                    return support_below['price'] * 1.001  # Чуть выше поддержки
            
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска S&R цели: {e}")
            return None
    
    def estimate_success_probability(self, entry_price, signal):
        """📊 Оцениваем вероятность успеха сделки"""
        try:
            probability = 0.5  # Базовая вероятность
            
            # Корректируем на основе RSI
            if len(self.rsi_values) > 0:
                rsi = self.rsi_values[-1]
                
                if signal == 1:  # Лонг
                    if rsi < 25:  # Очень перепродано
                        probability += 0.2
                    elif rsi < 30:  # Перепродано
                        probability += 0.1
                
                elif signal == -1:  # Шорт
                    if rsi > 75:  # Очень перекуплено
                        probability += 0.2
                    elif rsi > 70:  # Перекуплено
                        probability += 0.1
            
            # Корректируем на основе силы S&R уровней
            if self.use_support_resistance and self.sr_levels:
                from support_resistance import SupportResistanceFinder
                
                if not hasattr(self, 'sr_finder'):
                    self.sr_finder = SupportResistanceFinder()
                
                nearest_levels = self.sr_finder.get_nearest_levels(entry_price, self.sr_levels)
                
                if signal == 1 and nearest_levels['support']:  # Лонг рядом с поддержкой
                    distance_pct = abs(entry_price - nearest_levels['support'].price) / entry_price * 100
                    if distance_pct < 0.5:  # Очень близко к поддержке
                        probability += 0.15
                    elif distance_pct < 1.0:
                        probability += 0.1
                
                elif signal == -1 and nearest_levels['resistance']:  # Шорт рядом с сопротивлением
                    distance_pct = abs(nearest_levels['resistance'].price - entry_price) / entry_price * 100
                    if distance_pct < 0.5:  # Очень близко к сопротивлению
                        probability += 0.15
                    elif distance_pct < 1.0:
                        probability += 0.1
            
            # Ограничиваем вероятность разумными пределами
            return max(0.2, min(0.8, probability))
            
        except Exception as e:
            print(f"⚠️ Ошибка оценки вероятности: {e}")
            return 0.5
    
    def update_trailing_stop(self, current_price):
        """🛡️ Обновляем трейлинг стоп"""
        if not self.use_trailing_stop or self.position == 0:
            return
        
        if self.position == 1:  # Лонг позиция
            # Обновляем максимум
            if self.trailing_high is None or current_price > self.trailing_high:
                self.trailing_high = current_price
                # Пересчитываем трейлинг стоп
                new_stop = self.trailing_high * (1 - self.trailing_stop_pct)
                # Стоп может только расти для лонга
                if self.stop_loss_price is None or new_stop > self.stop_loss_price:
                    self.stop_loss_price = new_stop
        
        elif self.position == -1:  # Шорт позиция
            # Обновляем минимум
            if self.trailing_low is None or current_price < self.trailing_low:
                self.trailing_low = current_price
                # Пересчитываем трейлинг стоп
                new_stop = self.trailing_low * (1 + self.trailing_stop_pct)
                # Стоп может только снижаться для шорта
                if self.stop_loss_price is None or new_stop < self.stop_loss_price:
                    self.stop_loss_price = new_stop
    
    def check_stop_loss(self, current_price, current_dt):
        """🛡️ Проверяем срабатывание стоп-лосса"""
        if not self.use_stop_loss or self.position == 0 or self.stop_loss_price is None:
            return False
        
        stop_triggered = False
        
        if self.position == 1:  # Лонг позиция
            if current_price <= self.stop_loss_price:
                stop_triggered = True
        elif self.position == -1:  # Шорт позиция
            if current_price >= self.stop_loss_price:
                stop_triggered = True
        
        if stop_triggered:
            # Закрываем позицию по стоп-лоссу
            self.exit_points.append((current_dt, current_price))
            
            # Сохраняем позицию ДО обнуления
            old_position = self.position
            
            # Обновляем equity с учетом комиссии тейкера (стоп = маркет ордер)
            if len(self.entry_points) > 0:
                entry_price = self.entry_points[-1][1]
                if old_position == 1:  # Был лонг
                    pnl = (current_price - entry_price) / entry_price
                else:  # Был шорт
                    pnl = (entry_price - current_price) / entry_price
                
                # Вычитаем комиссии (вход + выход)
                total_fee = self.maker_fee + self.taker_fee  # Вход лимит + выход маркет
                pnl -= total_fee
                
                self.equity *= (1 + pnl)
            
            # Обнуляем позицию и стоп-лосс ПОСЛЕ расчета PnL
            self.position = 0
            self.entry_price = None
            self.stop_loss_price = None
            self.trailing_high = None
            self.trailing_low = None
            self.stop_loss_triggered += 1
            
            return True
        
        return False

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
                    self.entry_price = target_price
                    self.entry_points.append((current_dt, target_price))  # 🎯 Фиксируем фактический вход
                    # Устанавливаем стоп-лосс
                    current_atr = self.atr_values[-1] if len(self.atr_values) > 0 else None
                    self.stop_loss_price = self.calculate_stop_loss(target_price, 1, current_atr)
                    self.trailing_high = target_price  # Инициализируем трейлинг
                    self.trailing_low = None
                    
                elif signal == -1:
                    # Открываем шорт
                    self.last_price = target_price
                    self.entry_price = target_price
                    self.entry_points.append((current_dt, target_price))  # 🎯 Фиксируем фактический вход
                    # Устанавливаем стоп-лосс
                    current_atr = self.atr_values[-1] if len(self.atr_values) > 0 else None
                    self.stop_loss_price = self.calculate_stop_loss(target_price, -1, current_atr)
                    self.trailing_low = target_price  # Инициализируем трейлинг
                    self.trailing_high = None
                
                self.position = signal
        
        # Удаляем исполненные ордера (в обратном порядке, чтобы не сбить индексы)
        for i in reversed(executed_orders):
            del self.pending_orders[i]
    
    def on_tick(self, price, dt, volume=0):
        # 🛡️ Проверяем стоп-лоссы ПЕРВЫМИ (приоритет!)
        if self.check_stop_loss(price, dt):
            # Если сработал стоп-лосс, пропускаем остальную логику
            return
        
        # 🛡️ Обновляем трейлинг стоп
        self.update_trailing_stop(price)
        
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
        
        # 📊 Вычисляем индикаторы волатильности (оптимизированно)
        candles_with_current = self.candles + [self.current_candle]
        atr = compute_atr(candles_with_current, period=14)
        volatility_ratio = compute_volatility_ratio(candles_with_current, atr_period=14, lookback=50)
        
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
        # 📊 Обновляем уровни поддержки и сопротивления
        if candle_closed:
            self.update_support_resistance_levels()
        
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
            # entry_points добавляется в check_pending_orders при исполнении ордера
            
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
            # entry_points добавляется в check_pending_orders при исполнении ордера
            
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
                    
        
        # 📊 Применяем модификацию сигналов на основе S&R
        if self.use_support_resistance and signal != self.position:
            signal, sr_strength = self.get_sr_signal_modifier(signal, price)
        
        # 💰 ФИЛЬТР ПО ОЖИДАЕМОЙ ПРИБЫЛЬНОСТИ
        if signal != self.position and signal != self.last_signal:
            # Рассчитываем ожидаемую прибыльность перед входом
            trade_analysis = self.analyze_trade_profitability(price, signal)
            
            # Фильтруем сделки с плохим соотношением риск/прибыль
            if trade_analysis['risk_reward_ratio'] < 0.5:  # Минимум 1:0.5
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"❌ Сделка отклонена: R/R {trade_analysis['risk_reward_ratio']:.2f} < 0.5")
                return  # Пропускаем сделку
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"✅ Сделка одобрена: R/R {trade_analysis['risk_reward_ratio']:.2f}, "
                      f"риск {trade_analysis['risk_pct']:.2f}%, "
                      f"цель {trade_analysis['target_pct']:.2f}%")
        
        # 🏭 Создаем отложенные ордера ТОЛЬКО при изменении сигнала
        if signal != self.position and signal != self.last_signal:
            # 📚 Записываем торговое решение ТОЛЬКО при реальном изменении сигнала
            if self.collect_training_data and self.training_data_collector:
                signal_type = 'hold'  # По умолчанию
                if signal == 1 and self.position == 0:
                    signal_type = 'buy'
                elif signal == -1 and self.position == 0:
                    signal_type = 'sell_short'
                elif signal == 0 and self.position == 1:
                    signal_type = 'sell'
                elif signal == 0 and self.position == -1:
                    signal_type = 'buy_cover'
                
                self.training_data_collector.record_trading_moment(self, signal_type, price, dt)
            
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
            
            # 🛡️ Статистика стоп-лоссов
            'stop_loss_triggered': self.stop_loss_triggered,
            'stop_loss_rate_pct': (self.stop_loss_triggered / len(self.entry_points) * 100) if len(self.entry_points) > 0 else 0,
        }
        
        if len(self.trades) > 0:
            # Оценка общих расходов на комиссии
            total_fees = len(self.trades) * self.maker_fee * 2  # Вход + выход
            stats['total_fees_pct'] = total_fees * 100
            stats['avg_fee_per_trade_pct'] = (total_fees / len(self.trades)) * 100
        
        return stats
    
    def enable_training_data_collection(self, output_file=None):
        """📚 Включает сбор данных для обучения нейронной сети"""
        self.collect_training_data = True
        self.training_data_collector = TrainingDataCollector(output_file)
        print(f"📚 Сбор обучающих данных включен: {self.training_data_collector.output_file}")
    
    def save_training_data(self):
        """💾 Сохраняет собранные данные для обучения"""
        if self.training_data_collector:
            return self.training_data_collector.save_data()
        return None


class TrainingDataCollector:
    """📚 Собирает данные торговых моментов для обучения нейронной сети"""
    
    def __init__(self, output_file=None):
        self.output_file = output_file or f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.trading_moments = []
        self.signal_counter = 0
    
    def record_trading_moment(self, strategy, signal_type, current_price, current_dt):
        """📝 Записывает момент торгового решения"""
        try:
            # Проверяем что у нас достаточно данных
            if len(strategy.candles) < 20 or len(strategy.rsi_values) < 20:
                return
            
            # Извлекаем текущие признаки (как в нейронном фильтре)
            lookback = min(20, len(strategy.rsi_values))
            recent_rsi = strategy.rsi_values[-lookback:]
            recent_bb = strategy.bb_values[-lookback:]
            recent_atr = strategy.atr_values[-lookback:]
            recent_vol_ratio = strategy.volatility_ratios[-lookback:]
            recent_prices = [c.close for c in strategy.candles[-lookback:]]
            
            # Подготавливаем признаки через нейронный фильтр
            if strategy.neural_filter:
                features = strategy.neural_filter.prepare_features(
                    recent_rsi, recent_bb, recent_atr, recent_vol_ratio, recent_prices
                )
            else:
                # Базовые признаки если нет нейронного фильтра
                features = self._prepare_basic_features(
                    recent_rsi, recent_bb, recent_atr, recent_vol_ratio, recent_prices
                )
            
            if features is None:
                return
            
            # Сохраняем момент торгового решения
            trading_moment = {
                'timestamp': current_dt.isoformat(),
                'signal_type': signal_type,  # 'buy', 'sell', 'hold'
                'price': current_price,
                'features': features.tolist() if hasattr(features, 'tolist') else list(features),
                'rsi': recent_rsi[-1],
                'position': strategy.position,
                'candle_count': len(strategy.candles),
                'signal_id': self.signal_counter
            }
            
            # Добавляем информацию о S&R если доступно
            if hasattr(strategy, 'sr_levels') and strategy.sr_levels:
                try:
                    from support_resistance import SupportResistanceFinder
                    sr_finder = SupportResistanceFinder()
                    nearest = sr_finder.get_nearest_levels(strategy.sr_levels, current_price)
                    
                    trading_moment['sr_info'] = {
                        'nearest_support': nearest['support'].price if nearest['support'] else None,
                        'nearest_resistance': nearest['resistance'].price if nearest['resistance'] else None,
                        'support_strength': nearest['support'].strength if nearest['support'] else 0,
                        'resistance_strength': nearest['resistance'].strength if nearest['resistance'] else 0,
                    }
                except Exception:
                    pass
            
            self.trading_moments.append(trading_moment)
            self.signal_counter += 1
            
            # Периодически выводим статистику
            if self.signal_counter % 50 == 0:
                print(f"📚 Собрано торговых моментов: {self.signal_counter}")
                
        except Exception as e:
            print(f"⚠️ Ошибка записи торгового момента: {e}")
    
    def _prepare_basic_features(self, rsi_values, bb_values, atr_values, volatility_ratios, prices):
        """📊 Подготавливает базовые признаки если нет нейронного фильтра"""
        try:
            import numpy as np
            
            features = []
            
            # RSI статистики
            features.extend([
                np.mean(rsi_values),
                np.std(rsi_values), 
                rsi_values[-1],
                np.min(rsi_values),
                np.max(rsi_values),
            ])
            
            # Bollinger Bands позиции
            bb_positions = []
            for i, (ma, upper, lower) in enumerate(bb_values):
                if ma and upper and lower:
                    bb_pos = (prices[i] - lower) / (upper - lower) if upper != lower else 0.5
                    bb_positions.append(bb_pos)
                else:
                    bb_positions.append(0.5)
            
            features.extend([
                np.mean(bb_positions),
                np.std(bb_positions),
                bb_positions[-1],
            ])
            
            # ATR и волатильность
            features.extend([
                np.mean(atr_values),
                atr_values[-1],
                np.mean(volatility_ratios),
                volatility_ratios[-1],
            ])
            
            # Ценовая динамика
            price_changes = np.diff(prices)
            features.extend([
                np.mean(price_changes),
                np.std(price_changes),
                price_changes[-1] / prices[-2] if len(prices) > 1 else 0,
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ Ошибка подготовки базовых признаков: {e}")
            return None
    
    def save_data(self):
        """💾 Сохраняет собранные данные в JSON файл"""
        try:
            import json
            
            data = {
                'metadata': {
                    'total_moments': len(self.trading_moments),
                    'collection_time': datetime.now().isoformat(),
                    'features_count': len(self.trading_moments[0]['features']) if self.trading_moments else 0
                },
                'trading_moments': self.trading_moments
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Сохранено {len(self.trading_moments)} торговых моментов в {self.output_file}")
            return self.output_file
            
        except Exception as e:
            print(f"❌ Ошибка сохранения данных: {e}")
            return None 