"""
Тестирование разных порогов уверенности AI-фильтра
"""

import gzip
import csv
import numpy as np
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
import os

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_threshold(threshold, test_file="data/BTCUSDT_2023-02-01.csv.gz"):
    """Тестирует стратегию с определенным порогом"""
    
    if not os.path.exists(test_file):
        return None
    
    # Создаем стратегию с заданным порогом
    strategy = RSIStrategyBase(
        use_custom_rsi=True,
        use_neural_filter=True,
        neural_confidence_threshold=threshold
    )
    
    # Обрабатываем файл
    try:
        with gzip.open(test_file, 'rt') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i > 50000:  # Ограничиваем для скорости
                    break
                    
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
        
        strategy.on_finish(price)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None
    
    # Возвращаем результаты
    return {
        'threshold': threshold,
        'candles': len(strategy.candles),
        'trades': len(strategy.trades),
        'entries': len(strategy.entry_points),
        'exits': len(strategy.exit_points),
        'final_equity': strategy.equity,
        'return': (strategy.equity - 1.0) * 100
    }

def main():
    """Тестируем разные пороги"""
    
    print("🎯 ТЕСТИРОВАНИЕ РАЗНЫХ ПОРОГОВ AI-ФИЛЬТРА")
    print("=" * 60)
    
    # Тестируем разные пороги
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    for threshold in thresholds:
        print(f"\n🔧 Тестирую порог {threshold:.1f}...")
        result = test_threshold(threshold)
        
        if result:
            results.append(result)
            print(f"   📊 Сделок: {result['trades']}, Доходность: {result['return']:.2f}%")
        else:
            print(f"   ❌ Ошибка тестирования")
    
    # Анализируем результаты
    if results:
        print(f"\n📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
        print("=" * 60)
        print(f"{'Порог':<8} {'Сделки':<8} {'Входы':<8} {'Выходы':<8} {'Доходность':<12}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['threshold']:<8.1f} {r['trades']:<8} {r['entries']:<8} {r['exits']:<8} {r['return']:<12.2f}%")
        
        # Найдем оптимальный порог
        best_result = max(results, key=lambda x: x['return'])
        most_active = max(results, key=lambda x: x['trades'])
        
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"   💰 Максимальная доходность: {best_result['return']:.2f}% при пороге {best_result['threshold']:.1f}")
        print(f"   🔄 Максимум сделок: {most_active['trades']} при пороге {most_active['threshold']:.1f}")
        
        # Рекомендация
        if best_result['trades'] > 0:
            print(f"\n💡 РЕКОМЕНДАЦИЯ:")
            print(f"   Оптимальный порог: {best_result['threshold']:.1f}")
            print(f"   Ожидаемые результаты: {best_result['trades']} сделок, {best_result['return']:.2f}% доходность")
        else:
            print(f"\n⚠️ Все пороги дают 0 сделок - возможно, нужно переобучить модель")

if __name__ == "__main__":
    main()

