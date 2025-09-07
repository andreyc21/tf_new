"""
СУПЕР-БЫСТРЫЙ тест AI-модели 2025 на ОДНОМ файле
"""

import os
import gzip
import csv
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
from config import *
import time

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def run_strategy_on_file(file_path, use_neural_filter=False):
    """Запускает стратегию на файле и возвращает результаты"""
    
    strategy = RSIStrategyBase(
        use_custom_rsi=True,
        use_neural_filter=use_neural_filter,
        neural_confidence_threshold=NEURAL_CONFIDENCE_THRESHOLD
    )
    
    tick_count = 0
    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            strategy.on_tick(price, dt, volume)
            tick_count += 1
    
    strategy.on_finish(price)
    
    return {
        'total_pnl': (strategy.equity - 1.0),
        'total_trades': len(strategy.trades),
        'equity': strategy.equity,
        'sharpe': strategy.sharpe() if hasattr(strategy, 'sharpe') else 0,
        'candles': len(strategy.candles),
        'tick_count': tick_count
    }

def main():
    """Основная функция тестирования"""
    
    print("⚡ СУПЕР-БЫСТРЫЙ ТЕСТ AI-МОДЕЛИ 2025")
    print("=" * 50)
    print(f"📊 Порог уверенности: {NEURAL_CONFIDENCE_THRESHOLD}")
    print(f"🧠 Модель: {NEURAL_MODEL_PATH}")
    print()
    
    # Берем один конкретный файл
    test_file = "data/BTCUSDT_2025-02-01.csv.gz"
    
    if not os.path.exists(test_file):
        print(f"❌ Файл {test_file} не найден!")
        return
    
    print(f"🎯 Тестируем файл: {os.path.basename(test_file)}")
    print()
    
    start_time = time.time()
    
    # Тестируем обе стратегии
    print("🔄 Тестируем базовую стратегию...")
    base_result = run_strategy_on_file(test_file, use_neural_filter=False)
    
    print("🔄 Тестируем AI-стратегию...")
    ai_result = run_strategy_on_file(test_file, use_neural_filter=True)
    
    end_time = time.time()
    
    # Анализируем результаты
    print("\n📊 РЕЗУЛЬТАТЫ:")
    print("=" * 40)
    
    base_pnl = base_result['total_pnl']
    ai_pnl = ai_result['total_pnl']
    base_trades = base_result['total_trades']
    ai_trades = ai_result['total_trades']
    improvement = ai_pnl - base_pnl
    filter_rate = 1 - (ai_trades / max(base_trades, 1))
    
    print(f"📈 Базовая стратегия:")
    print(f"   💰 PnL: {base_pnl:+.3%}")
    print(f"   🔄 Сделок: {base_trades}")
    print(f"   📊 Equity: {base_result['equity']:.4f}")
    print(f"   📈 Свечей: {base_result['candles']}")
    print()
    
    print(f"🧠 AI-стратегия:")
    print(f"   💰 PnL: {ai_pnl:+.3%}")
    print(f"   🔄 Сделок: {ai_trades}")
    print(f"   📊 Equity: {ai_result['equity']:.4f}")
    print(f"   📈 Свечей: {ai_result['candles']}")
    print()
    
    print(f"🎯 СРАВНЕНИЕ:")
    print(f"   📊 Улучшение PnL: {improvement:+.3%}")
    print(f"   🎯 Фильтрация сделок: {filter_rate:.1%}")
    print(f"   ⏱️  Время выполнения: {end_time - start_time:.1f} сек")
    print()
    
    # Вывод
    if improvement > 0:
        print("✅ AI-ФИЛЬТР РАБОТАЕТ! 🎉")
        print("   📈 Улучшает прибыльность")
        print("   🎯 Эффективно фильтрует плохие сигналы")
        if filter_rate > 0:
            print(f"   🔥 Сократил сделки на {filter_rate:.0%} и улучшил результат!")
    elif improvement == 0:
        print("➡️  AI-фильтр нейтрален")
        print("   ⚖️  Не улучшает, но и не ухудшает")
    else:
        print("❌ AI-фильтр ухудшает результаты")
        print("   📉 Слишком агрессивная фильтрация")
    
    print()
    print("🎉 Тест завершен!")

if __name__ == "__main__":
    main()
