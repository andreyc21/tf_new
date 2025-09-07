"""
Финальное тестирование AI-модели 2025 с оптимизированным порогом 0.51
"""

import os
import glob
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
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

def test_single_file(file_path):
    """Тестирует стратегию на одном файле"""
    
    try:
        # Тестируем обе стратегии
        base_result = run_strategy_on_file(file_path, use_neural_filter=False)
        ai_result = run_strategy_on_file(file_path, use_neural_filter=True)
        
        filename = os.path.basename(file_path)
        
        return {
            'file': filename,
            'base': base_result,
            'ai': ai_result,
            'ai_improvement': ai_result['total_pnl'] - base_result['total_pnl'],
            'ai_filter_rate': 1 - (ai_result['total_trades'] / max(base_result['total_trades'], 1))
        }
        
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }

def main():
    """Основная функция тестирования"""
    
    print("🚀 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ AI-МОДЕЛИ 2025")
    print("=" * 60)
    print(f"📊 Порог уверенности: {NEURAL_CONFIDENCE_THRESHOLD}")
    print(f"🧠 Модель: {NEURAL_MODEL_PATH}")
    print(f"📏 Скалер: {NEURAL_SCALER_PATH}")
    print()
    
    # Находим все файлы 2025 года
    data_pattern = "data/BTCUSDT_2025-*.csv.gz"
    files = sorted(glob.glob(data_pattern))
    
    if not files:
        print("❌ Файлы данных не найдены!")
        return
    
    print(f"📁 Найдено файлов: {len(files)}")
    print(f"🔧 Используем процессоров: {multiprocessing.cpu_count()}")
    print()
    
    start_time = time.time()
    
    # Параллельное тестирование
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(test_single_file, files))
    
    end_time = time.time()
    
    # Анализируем результаты
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)
    
    base_total_pnl = 0
    ai_total_pnl = 0
    base_total_trades = 0
    ai_total_trades = 0
    successful_tests = 0
    ai_improvements = 0
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['file']}: {result['error']}")
            continue
            
        successful_tests += 1
        base_pnl = result['base']['total_pnl']
        ai_pnl = result['ai']['total_pnl']
        base_trades = result['base']['total_trades']
        ai_trades = result['ai']['total_trades']
        improvement = result['ai_improvement']
        filter_rate = result['ai_filter_rate']
        
        base_total_pnl += base_pnl
        ai_total_pnl += ai_pnl
        base_total_trades += base_trades
        ai_total_trades += ai_trades
        
        if improvement > 0:
            ai_improvements += 1
        
        # Показываем детали только для значимых результатов
        if base_trades > 0 or ai_trades > 0:
            print(f"📅 {result['file'][:15]}:")
            print(f"   Базовая: {base_pnl:+.2%} ({base_trades} сделок)")
            print(f"   AI:      {ai_pnl:+.2%} ({ai_trades} сделок)")
            print(f"   Улучшение: {improvement:+.2%}, Фильтрация: {filter_rate:.1%}")
            print()
    
    # Общая статистика
    print("🏆 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 40)
    print(f"📊 Успешных тестов: {successful_tests}/{len(files)}")
    print(f"⏱️  Время выполнения: {end_time - start_time:.1f} сек")
    print()
    
    print(f"💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Базовая стратегия:  {base_total_pnl:+.2%} ({base_total_trades} сделок)")
    print(f"   AI-стратегия:       {ai_total_pnl:+.2%} ({ai_total_trades} сделок)")
    print(f"   Абсолютное улучшение: {ai_total_pnl - base_total_pnl:+.2%}")
    print(f"   Относительное улучшение: {((ai_total_pnl / max(base_total_pnl, 0.001)) - 1):+.1%}")
    print()
    
    print(f"🎯 AI-ФИЛЬТР СТАТИСТИКА:")
    total_filter_rate = 1 - (ai_total_trades / max(base_total_trades, 1))
    print(f"   Общая фильтрация: {total_filter_rate:.1%}")
    print(f"   Файлов с улучшением: {ai_improvements}/{successful_tests} ({ai_improvements/max(successful_tests,1):.1%})")
    print()
    
    # Рекомендации
    if ai_total_pnl > base_total_pnl:
        print("✅ РЕКОМЕНДАЦИЯ: AI-фильтр показывает положительные результаты!")
        print("   Можно использовать в продакшене с текущими настройками.")
    else:
        print("⚠️  РЕКОМЕНДАЦИЯ: AI-фильтр требует доработки.")
        print("   Рассмотрите изменение порога уверенности или переобучение модели.")
    
    print()
    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    main()
