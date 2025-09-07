#!/usr/bin/env python

import numpy as np
import gzip
import csv
from datetime import datetime, timezone

try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib доступен")
except ImportError:
    TALIB_AVAILABLE = False
    print("❌ TA-Lib недоступен")

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def compute_rsi_numpy(prices, period=14):
    """Наша реализация RSI"""
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

def compute_rsi_talib(prices, period=14):
    """TA-Lib реализация RSI"""
    if not TALIB_AVAILABLE:
        return None
    
    prices_array = np.array(prices, dtype=np.float64)
    if len(prices_array) < period + 1:
        return 50.0
    
    rsi_values = talib.RSI(prices_array, timeperiod=period)
    return rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0

def compute_bb_numpy(prices, period=20, num_std=2):
    """Наша реализация Bollinger Bands"""
    prices = np.array(prices)
    if len(prices) < period:
        return None, None, None
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def compute_bb_talib(prices, period=20, num_std=2):
    """TA-Lib реализация Bollinger Bands"""
    if not TALIB_AVAILABLE:
        return None, None, None
    
    prices_array = np.array(prices, dtype=np.float64)
    if len(prices_array) < period:
        return None, None, None
    
    upper, middle, lower = talib.BBANDS(prices_array, timeperiod=period, nbdevup=num_std, nbdevdn=num_std, matype=0)
    
    if np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1]):
        return None, None, None
    
    return middle[-1], upper[-1], lower[-1]

def analyze_differences(filename, max_candles=100):
    """Анализирует различия между TA-Lib и NumPy реализациями"""
    
    print(f"\n🔍 АНАЛИЗ РАЗЛИЧИЙ: {filename}")
    print("=" * 60)
    
    # Читаем данные и создаём свечи
    candles = []
    current_candle = None
    current_time = None
    
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            dt = timestamp_to_dt(row['timestamp'])
            
            # Группируем в 5-минутные свечи
            candle_time = dt.replace(second=0, microsecond=0)
            candle_time = candle_time.replace(minute=candle_time.minute // 5 * 5)
            
            if current_time != candle_time:
                if current_candle is not None:
                    candles.append(current_candle)
                    if len(candles) >= max_candles:
                        break
                current_candle = {'time': candle_time, 'open': price, 'high': price, 'low': price, 'close': price}
                current_time = candle_time
            else:
                current_candle['high'] = max(current_candle['high'], price)
                current_candle['low'] = min(current_candle['low'], price)
                current_candle['close'] = price
    
    if current_candle:
        candles.append(current_candle)
    
    print(f"📊 Создано свечей: {len(candles)}")
    
    # Извлекаем цены закрытия
    closes = [c['close'] for c in candles]
    
    print(f"\n📈 СРАВНЕНИЕ RSI:")
    print("-" * 40)
    
    rsi_differences = []
    bb_differences = []
    
    # Сравниваем RSI на разных участках
    for i in range(20, min(len(closes), max_candles), 10):
        prices_subset = closes[:i]
        
        rsi_numpy = compute_rsi_numpy(prices_subset, 14)
        rsi_talib = compute_rsi_talib(prices_subset, 14) if TALIB_AVAILABLE else None
        
        if rsi_talib is not None:
            diff = abs(rsi_numpy - rsi_talib)
            rsi_differences.append(diff)
            
            if diff > 0.1:  # Показываем только значимые различия
                print(f"Свеча {i:3d}: NumPy={rsi_numpy:6.2f}, TA-Lib={rsi_talib:6.2f}, Разница={diff:5.2f}")
    
    print(f"\n📊 СРАВНЕНИЕ BOLLINGER BANDS:")
    print("-" * 40)
    
    # Сравниваем BB на разных участках
    for i in range(25, min(len(closes), max_candles), 10):
        prices_subset = closes[:i]
        
        ma_numpy, upper_numpy, lower_numpy = compute_bb_numpy(prices_subset, 20, 2)
        ma_talib, upper_talib, lower_talib = compute_bb_talib(prices_subset, 20, 2) if TALIB_AVAILABLE else (None, None, None)
        
        if ma_talib is not None and ma_numpy is not None:
            ma_diff = abs(ma_numpy - ma_talib)
            upper_diff = abs(upper_numpy - upper_talib)
            lower_diff = abs(lower_numpy - lower_talib)
            
            bb_differences.extend([ma_diff, upper_diff, lower_diff])
            
            if ma_diff > 0.1 or upper_diff > 0.1 or lower_diff > 0.1:
                print(f"Свеча {i:3d}:")
                print(f"  MA:    NumPy={ma_numpy:8.2f}, TA-Lib={ma_talib:8.2f}, Разница={ma_diff:5.2f}")
                print(f"  Upper: NumPy={upper_numpy:8.2f}, TA-Lib={upper_talib:8.2f}, Разница={upper_diff:5.2f}")
                print(f"  Lower: NumPy={lower_numpy:8.2f}, TA-Lib={lower_talib:8.2f}, Разница={lower_diff:5.2f}")
    
    # Статистика различий
    if rsi_differences and TALIB_AVAILABLE:
        print(f"\n📊 СТАТИСТИКА РАЗЛИЧИЙ RSI:")
        print(f"Средняя разница: {np.mean(rsi_differences):.4f}")
        print(f"Максимальная разница: {np.max(rsi_differences):.4f}")
        print(f"Медианная разница: {np.median(rsi_differences):.4f}")
        
    if bb_differences and TALIB_AVAILABLE:
        print(f"\n📊 СТАТИСТИКА РАЗЛИЧИЙ BB:")
        print(f"Средняя разница: {np.mean(bb_differences):.4f}")
        print(f"Максимальная разница: {np.max(bb_differences):.4f}")
        print(f"Медианная разница: {np.median(bb_differences):.4f}")

def compare_rsi_algorithms():
    """Сравнивает разные алгоритмы RSI"""
    print(f"\n🧮 СРАВНЕНИЕ АЛГОРИТМОВ RSI:")
    print("=" * 50)
    
    # Тестовые данные
    test_prices = [44.0, 44.25, 44.5, 43.75, 44.5, 44.75, 47.0, 47.25, 46.5, 46.75, 46.0, 46.25, 
                   47.75, 47.5, 47.25, 47.75, 48.75, 47.5, 47.5, 47.0, 44.5, 44.25, 44.0, 44.0]
    
    print(f"Тестовые данные: {len(test_prices)} цен")
    print(f"Цены: {test_prices[:10]}... (показаны первые 10)")
    
    # Наш алгоритм
    rsi_numpy = compute_rsi_numpy(test_prices, 14)
    print(f"\nNumPy RSI(14): {rsi_numpy:.6f}")
    
    if TALIB_AVAILABLE:
        rsi_talib = compute_rsi_talib(test_prices, 14)
        print(f"TA-Lib RSI(14): {rsi_talib:.6f}")
        print(f"Разница: {abs(rsi_numpy - rsi_talib):.6f}")
        
        # Проверим разные периоды
        print(f"\nСравнение для разных периодов:")
        for period in [10, 14, 21]:
            numpy_val = compute_rsi_numpy(test_prices, period)
            talib_val = compute_rsi_talib(test_prices, period)
            diff = abs(numpy_val - talib_val)
            print(f"RSI({period:2d}): NumPy={numpy_val:7.3f}, TA-Lib={talib_val:7.3f}, Diff={diff:.3f}")

def investigate_rsi_calculation_details():
    """Детально исследует расчёт RSI"""
    print(f"\n🔬 ДЕТАЛЬНОЕ ИССЛЕДОВАНИЕ RSI:")
    print("=" * 50)
    
    # Простые тестовые данные
    prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107]
    period = 14
    
    print(f"Цены: {prices}")
    print(f"Период RSI: {period}")
    
    # Вычисляем изменения
    deltas = np.diff(prices)
    print(f"\nИзменения цен: {deltas}")
    
    # Разделяем на прибыли и убытки
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    print(f"Прибыли: {gains}")
    print(f"Убытки: {losses}")
    
    # Средние значения
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    print(f"\nСредняя прибыль: {avg_gain:.6f}")
    print(f"Средний убыток: {avg_loss:.6f}")
    
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
        print(f"RS: {rs:.6f}")
        print(f"RSI: {rsi:.6f}")
    
    if TALIB_AVAILABLE:
        talib_rsi = compute_rsi_talib(prices, period)
        print(f"TA-Lib RSI: {talib_rsi:.6f}")

if __name__ == '__main__':
    import sys
    
    # Сравнение алгоритмов на тестовых данных
    compare_rsi_algorithms()
    investigate_rsi_calculation_details()
    
    # Анализ на реальных данных
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_2025-03-01.csv.gz"
    if TALIB_AVAILABLE:
        analyze_differences(filename, 50)
    else:
        print("\n⚠️  TA-Lib не установлен. Установите для сравнения: pip install TA-Lib")
