#!/usr/bin/env python

import time
import gzip
import csv
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase as OriginalStrategy
from rsi_strategy_cached import RSIStrategyCached
from rsi_strategy_fast import RSIStrategyFast
from rsi_strategy_talib import RSIStrategyTALib

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def benchmark_comparison(filename):
    """Сравнивает производительность оригинальной и кэшированной стратегии"""
    
    print(f"🏁 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print(f"Файл: {filename}")
    print("=" * 60)
    
    # Читаем данные один раз
    ticks = []
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            ticks.append((price, dt, volume))
    
    print(f"📊 Загружено тиков: {len(ticks):,}")
    
    # Тест оригинальной стратегии
    print(f"\n🐌 ОРИГИНАЛЬНАЯ СТРАТЕГИЯ:")
    start_time = time.time()
    
    original_strategy = OriginalStrategy()
    for price, dt, volume in ticks:
        original_strategy.on_tick(price, dt, volume)
    original_strategy.on_finish(price)
    
    original_time = time.time() - start_time
    
    print(f"⏱️  Время: {original_time:.2f} сек")
    print(f"💰 Equity: {original_strategy.equity:.4f}")
    print(f"🔄 Сделок: {len(original_strategy.trades)}")
    print(f"📊 Тиков/сек: {len(ticks)/original_time:,.0f}")
    
    # Тест кэшированной стратегии
    print(f"\n🚀 КЭШИРОВАННАЯ СТРАТЕГИЯ:")
    start_time = time.time()
    
    cached_strategy = RSIStrategyCached()
    for price, dt, volume in ticks:
        cached_strategy.on_tick(price, dt, volume)
    cached_strategy.on_finish(price)
    
    cached_time = time.time() - start_time
    
    print(f"⏱️  Время: {cached_time:.2f} сек")
    print(f"💰 Equity: {cached_strategy.equity:.4f}")
    print(f"🔄 Сделок: {len(cached_strategy.trades)}")
    print(f"📊 Тиков/сек: {len(ticks)/cached_time:,.0f}")
    
    # Тест быстрой стратегии
    print(f"\n⚡ БЫСТРАЯ СТРАТЕГИЯ (инкрементальная):")
    start_time = time.time()
    
    fast_strategy = RSIStrategyFast()
    for price, dt, volume in ticks:
        fast_strategy.on_tick(price, dt, volume)
    fast_strategy.on_finish(price)
    
    fast_time = time.time() - start_time
    
    print(f"⏱️  Время: {fast_time:.2f} сек")
    print(f"💰 Equity: {fast_strategy.equity:.4f}")
    print(f"🔄 Сделок: {len(fast_strategy.trades)}")
    print(f"📊 Тиков/сек: {len(ticks)/fast_time:,.0f}")
    
    # Тест TA-Lib стратегии
    print(f"\n🚀 TA-LIB СТРАТЕГИЯ:")
    start_time = time.time()
    
    talib_strategy = RSIStrategyTALib()
    for price, dt, volume in ticks:
        talib_strategy.on_tick(price, dt, volume)
    talib_strategy.on_finish(price)
    
    talib_time = time.time() - start_time
    
    print(f"⏱️  Время: {talib_time:.2f} сек")
    print(f"💰 Equity: {talib_strategy.equity:.4f}")
    print(f"🔄 Сделок: {len(talib_strategy.trades)}")
    print(f"📊 Тиков/сек: {len(ticks)/talib_time:,.0f}")
    
    # Сравнение всех версий
    cached_speedup = original_time / cached_time
    fast_speedup = original_time / fast_time
    talib_speedup = original_time / talib_time
    fast_vs_cached = cached_time / fast_time
    talib_vs_cached = cached_time / talib_time
    
    print(f"\n📈 СРАВНЕНИЕ ВСЕХ ВЕРСИЙ:")
    print(f"⚡ Кэшированная vs Оригинальная: {cached_speedup:.1f}x")
    print(f"⚡ Быстрая vs Оригинальная: {fast_speedup:.1f}x")  
    print(f"⚡ TA-Lib vs Оригинальная: {talib_speedup:.1f}x")
    print(f"⚡ Быстрая vs Кэшированная: {fast_vs_cached:.1f}x")
    print(f"⚡ TA-Lib vs Кэшированная: {talib_vs_cached:.1f}x")
    
    # Проверка корректности
    equity_diff_cached = abs(original_strategy.equity - cached_strategy.equity)
    trades_diff_cached = abs(len(original_strategy.trades) - len(cached_strategy.trades))
    equity_diff_fast = abs(original_strategy.equity - fast_strategy.equity)
    trades_diff_fast = abs(len(original_strategy.trades) - len(fast_strategy.trades))
    equity_diff_talib = abs(original_strategy.equity - talib_strategy.equity)
    trades_diff_talib = abs(len(original_strategy.trades) - len(talib_strategy.trades))
    
    print(f"\n✅ ПРОВЕРКА КОРРЕКТНОСТИ:")
    if equity_diff_cached < 1e-4 and trades_diff_cached == 0:
        print(f"✅ Кэшированная версия: идентична оригинальной")
    else:
        print(f"⚠️  Кэшированная версия отличается: equity={equity_diff_cached:.6f}, trades={trades_diff_cached}")
    
    if equity_diff_fast < 1e-4 and trades_diff_fast == 0:
        print(f"✅ Быстрая версия: идентична оригинальной")
    else:
        print(f"⚠️  Быстрая версия отличается: equity={equity_diff_fast:.6f}, trades={trades_diff_fast}")
        
    if equity_diff_talib < 1e-4 and trades_diff_talib == 0:
        print(f"✅ TA-Lib версия: идентична оригинальной")
    else:
        print(f"⚠️  TA-Lib версия отличается: equity={equity_diff_talib:.6f}, trades={trades_diff_talib}")
    
    # Прогноз для оптимизации
    times = {'кэшированная': cached_time, 'быстрая': fast_time, 'TA-Lib': talib_time}
    best_time = min(times.values())
    best_name = [name for name, time in times.items() if time == best_time][0]
    
    print(f"\n🔮 ПРОГНОЗ ДЛЯ ОПТИМИЗАЦИИ (64 комбинации):")
    print(f"⏱️  Лучшая версия ({best_name}): {best_time * 64:.0f} сек ({best_time * 64/60:.1f} мин)")
    print(f"⏰ Экономия vs оригинальной: {(original_time - best_time) * 64:.0f} сек ({(original_time - best_time) * 64/60:.1f} мин)")
    
    return {
        'original_time': original_time,
        'cached_time': cached_time,
        'fast_time': fast_time,
        'talib_time': talib_time,
        'best_speedup': original_time / best_time,
        'best_name': best_name,
        'ticks': len(ticks)
    }

if __name__ == '__main__':
    import sys
    
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_2025-03-01.csv.gz"
    benchmark_comparison(filename)
