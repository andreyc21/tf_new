#!/usr/bin/env python3
"""
🚀 Тест оптимизации производительности
"""

import time
from backtester import run_backtest_on_file

def test_optimization():
    """🚀 Тестируем оптимизированную версию"""
    
    test_file = "data/BTCUSDT_2025-01-01.csv.gz"
    
    print("🚀 ТЕСТ ОПТИМИЗИРОВАННОЙ ВЕРСИИ")
    print("=" * 60)
    
    # Тест оптимизированной версии
    print("📊 Запуск оптимизированного бэктеста...")
    start_time = time.time()
    
    strategy = run_backtest_on_file(test_file, plot=False, verbose=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Статистика
    tick_count = 1079455  # Известное количество тиков в файле
    tps = tick_count / total_time
    
    print(f"⏱️ Время выполнения: {total_time:.3f} сек")
    print(f"📊 Тиков обработано: {tick_count:,}")
    print(f"🚀 Скорость: {tps:.0f} тиков/сек")
    
    # Сравнение с предыдущим результатом
    old_tps = 590  # Из предыдущего профилирования
    speedup = tps / old_tps
    
    print(f"\n📈 РЕЗУЛЬТАТ ОПТИМИЗАЦИИ:")
    print(f"  Было: {old_tps} тиков/сек")
    print(f"  Стало: {tps:.0f} тиков/сек")
    print(f"  🎯 Ускорение: {speedup:.1f}x")
    
    if speedup > 10:
        print(f"  🎉 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Ускорение более чем в 10 раз!")
    elif speedup > 5:
        print(f"  ✅ Хороший результат! Значительное ускорение.")
    elif speedup > 2:
        print(f"  👍 Неплохое ускорение.")
    else:
        print(f"  😐 Небольшое улучшение.")
    
    # Проверяем корректность результатов
    stats = strategy.get_realistic_stats()
    print(f"\n🔍 ПРОВЕРКА КОРРЕКТНОСТИ:")
    print(f"  💰 Final equity: {stats['final_equity']:.4f}")
    print(f"  📊 Sharpe ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  🔄 Сделок: {stats['total_trades']}")
    print(f"  📋 Исполнение ордеров: {stats['execution_rate_pct']:.1f}%")
    
    return tps

if __name__ == '__main__':
    tps = test_optimization()
    
    print(f"\n🎯 ИТОГ: {tps:.0f} тиков/сек")
    
    # Оценка для реального использования
    million_ticks_time = 1_000_000 / tps / 60  # в минутах
    print(f"📊 Время на 1M тиков: {million_ticks_time:.1f} минут")
    
    if million_ticks_time < 5:
        print("🚀 Отлично! Подходит для реального времени.")
    elif million_ticks_time < 15:
        print("✅ Хорошо! Приемлемо для бэктестинга.")
    else:
        print("⚠️ Все еще медленно для больших объемов данных.")