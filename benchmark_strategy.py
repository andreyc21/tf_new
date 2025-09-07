#!/usr/bin/env python

import time
import gzip
import csv
import glob
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def benchmark_single_file(filename, strategy_params=None):
    """Измеряет время выполнения стратегии на одном файле"""
    if strategy_params is None:
        strategy_params = {'rsi_period': 14, 'rsi_buy': 30, 'rsi_sell': 70}
    
    print(f"📊 Бенчмарк файла: {filename}")
    
    start_time = time.time()
    
    strategy = RSIStrategyBase(**strategy_params)
    tick_count = 0
    
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            strategy.on_tick(price, dt, volume)
            tick_count += 1
        strategy.on_finish(price)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"⏱️  Время выполнения: {execution_time:.2f} секунд")
    print(f"📈 Обработано тиков: {tick_count:,}")
    print(f"🕯️  Создано свечей: {len(strategy.candles)}")
    print(f"🔄 Сделок: {len(strategy.trades)}")
    print(f"📊 Тиков в секунду: {tick_count/execution_time:,.0f}")
    print(f"💰 Итоговая equity: {strategy.equity:.4f}")
    
    return {
        'execution_time': execution_time,
        'tick_count': tick_count,
        'candles': len(strategy.candles),
        'trades': len(strategy.trades),
        'ticks_per_second': tick_count/execution_time,
        'equity': strategy.equity
    }

def benchmark_month(pattern, max_files=31):
    """Измеряет время выполнения на месяце данных"""
    files = sorted(glob.glob(pattern))[:max_files]
    
    if not files:
        print(f"❌ Файлы не найдены по паттерну: {pattern}")
        return
    
    print(f"🗓️  Бенчмарк месяца: {len(files)} файлов")
    print("=" * 60)
    
    total_start = time.time()
    total_ticks = 0
    total_trades = 0
    
    for i, filename in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {filename.split('/')[-1]}")
        try:
            result = benchmark_single_file(filename)
            total_ticks += result['tick_count']
            total_trades += result['trades']
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("📊 ИТОГИ БЕНЧМАРКА:")
    print(f"⏱️  Общее время: {total_time:.2f} секунд ({total_time/60:.1f} минут)")
    print(f"📈 Всего тиков: {total_ticks:,}")
    print(f"🔄 Всего сделок: {total_trades}")
    print(f"📊 Средняя скорость: {total_ticks/total_time:,.0f} тиков/сек")
    print(f"⚡ Время на файл: {total_time/len(files):.2f} сек")
    
    # Прогноз времени для оптимизации
    time_per_combination = total_time
    total_combinations = 64
    estimated_optimization_time = time_per_combination * total_combinations
    
    print(f"\n🔮 ПРОГНОЗ ДЛЯ ОПТИМИЗАЦИИ:")
    print(f"📊 Комбинаций параметров: {total_combinations}")
    print(f"⏱️  Время на комбинацию: {time_per_combination:.2f} сек")
    print(f"🕐 Общее время оптимизации: {estimated_optimization_time:.0f} сек ({estimated_optimization_time/60:.1f} мин)")
    
    if estimated_optimization_time > 300:  # 5 минут
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Оптимизация займёт больше 5 минут!")
        print(f"💡 Рекомендации для ускорения:")
        print(f"   - Уменьшить количество файлов для теста")
        print(f"   - Сократить диапазоны параметров")
        print(f"   - Оптимизировать код стратегии")

def profile_strategy_bottlenecks(filename):
    """Профилирует узкие места в стратегии"""
    import cProfile
    import pstats
    
    print(f"🔍 Профилирование узких мест: {filename}")
    
    def run_strategy():
        strategy = RSIStrategyBase()
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
            strategy.on_finish(price)
        return strategy
    
    # Запускаем профилировщик
    pr = cProfile.Profile()
    pr.enable()
    
    strategy = run_strategy()
    
    pr.disable()
    
    # Анализируем результаты
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    
    print(f"\n📊 ТОП-10 самых медленных функций:")
    stats.print_stats(10)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--profile':
            # Профилирование
            pattern = sys.argv[2] if len(sys.argv) > 2 else "data/BTCUSDT_2025-03-01.csv.gz"
            files = glob.glob(pattern)
            if files:
                profile_strategy_bottlenecks(files[0])
        else:
            # Бенчмарк конкретного паттерна
            pattern = sys.argv[1]
            benchmark_month(pattern)
    else:
        # По умолчанию тестируем март 2025
        benchmark_month("data/BTCUSDT_2025-03-*.csv.gz")
