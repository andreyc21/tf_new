"""
Тестирование AI-enhanced стратегии на исторических данных
"""

import gzip
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
import matplotlib.pyplot as plt
import os

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_ai_strategy_on_file(filename, use_ai_filter=True):
    """Тестирует AI-стратегию на одном файле"""
    
    print(f"📊 Тестируем {'AI-enhanced' if use_ai_filter else 'обычную'} стратегию на: {os.path.basename(filename)}")
    
    # Создаем стратегию
    strategy = RSIStrategyBase(
        use_custom_rsi=True,
        use_neural_filter=use_ai_filter,
        neural_confidence_threshold=0.6
    )
    
    # Загружаем и обрабатываем данные
    ticks_processed = 0
    try:
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
                ticks_processed += 1
        
        strategy.on_finish(price)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None
    
    # Анализируем результаты
    results = {
        'filename': os.path.basename(filename),
        'use_ai': use_ai_filter,
        'ticks_processed': ticks_processed,
        'candles_created': len(strategy.candles),
        'total_trades': len(strategy.trades),
        'entry_points': len(strategy.entry_points),
        'exit_points': len(strategy.exit_points),
        'final_equity': strategy.equity,
        'total_return': (strategy.equity - 1.0) * 100,
        'equity_curve': strategy.equity_curve.copy(),
    }
    
    # Рассчитываем дополнительные метрики
    if len(strategy.equity_curve) > 0:
        equity_array = np.array(strategy.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        results['max_drawdown'] = calculate_max_drawdown(equity_array)
        results['sharpe_ratio'] = calculate_sharpe_ratio(returns)
        results['win_rate'] = calculate_win_rate(strategy.trades)
    
    print(f"   📈 Свечей: {results['candles_created']}")
    print(f"   🔄 Сделок: {results['total_trades']}")
    print(f"   📊 Доходность: {results['total_return']:.2f}%")
    if 'max_drawdown' in results:
        print(f"   📉 Макс. просадка: {results['max_drawdown']:.2f}%")
        print(f"   📊 Sharpe: {results['sharpe_ratio']:.3f}")
    
    return results

def calculate_max_drawdown(equity_curve):
    """Рассчитывает максимальную просадку"""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min() * 100)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Рассчитывает коэффициент Шарпа"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float((np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252))

def calculate_win_rate(trades):
    """Рассчитывает процент выигрышных сделок"""
    if len(trades) <= 1:
        return 0.0
    
    winning_trades = sum(1 for i in range(1, len(trades)) if trades[i] > trades[i-1])
    return float(winning_trades / (len(trades) - 1) * 100)

def compare_strategies(test_files):
    """Сравнивает обычную и AI-enhanced стратегии"""
    
    print("🚀 СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 60)
    
    normal_results = []
    ai_results = []
    
    for filename in test_files:
        print(f"\n📂 Обрабатываем: {os.path.basename(filename)}")
        
        # Тестируем обычную стратегию
        normal_result = test_ai_strategy_on_file(filename, use_ai_filter=False)
        if normal_result:
            normal_results.append(normal_result)
        
        # Тестируем AI-enhanced стратегию
        ai_result = test_ai_strategy_on_file(filename, use_ai_filter=True)
        if ai_result:
            ai_results.append(ai_result)
    
    # Анализируем общие результаты
    print(f"\n📊 СВОДНЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    
    if normal_results and ai_results:
        normal_returns = [r['total_return'] for r in normal_results]
        ai_returns = [r['total_return'] for r in ai_results]
        
        normal_trades = sum(r['total_trades'] for r in normal_results)
        ai_trades = sum(r['total_trades'] for r in ai_results)
        
        normal_drawdowns = [r.get('max_drawdown', 0) for r in normal_results]
        ai_drawdowns = [r.get('max_drawdown', 0) for r in ai_results]
        
        normal_sharpe = [r.get('sharpe_ratio', 0) for r in normal_results]
        ai_sharpe = [r.get('sharpe_ratio', 0) for r in ai_results]
        
        print(f"📈 Обычная стратегия:")
        print(f"   💰 Средняя доходность: {np.mean(normal_returns):.2f}%")
        print(f"   🔄 Всего сделок: {normal_trades}")
        print(f"   📉 Средняя просадка: {np.mean(normal_drawdowns):.2f}%")
        print(f"   📊 Средний Sharpe: {np.mean(normal_sharpe):.3f}")
        
        print(f"\n🧠 AI-enhanced стратегия:")
        print(f"   💰 Средняя доходность: {np.mean(ai_returns):.2f}%")
        print(f"   🔄 Всего сделок: {ai_trades}")
        print(f"   📉 Средняя просадка: {np.mean(ai_drawdowns):.2f}%")
        print(f"   📊 Средний Sharpe: {np.mean(ai_sharpe):.3f}")
        
        # Улучшения
        return_improvement = np.mean(ai_returns) - np.mean(normal_returns)
        trade_reduction = ((normal_trades - ai_trades) / normal_trades * 100) if normal_trades > 0 else 0
        drawdown_improvement = np.mean(normal_drawdowns) - np.mean(ai_drawdowns)
        sharpe_improvement = np.mean(ai_sharpe) - np.mean(normal_sharpe)
        
        print(f"\n🎯 УЛУЧШЕНИЯ ОТ ИИ:")
        print(f"   💹 Доходность: {return_improvement:+.2f}% пунктов")
        print(f"   🔄 Сокращение сделок: {trade_reduction:.1f}%")
        print(f"   📉 Улучшение просадки: {drawdown_improvement:+.2f}% пунктов")
        print(f"   📊 Улучшение Sharpe: {sharpe_improvement:+.3f}")
        
        # Визуализация
        plot_strategy_comparison(normal_results, ai_results)
    
    return normal_results, ai_results

def plot_strategy_comparison(normal_results, ai_results):
    """Создает графики сравнения стратегий"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График доходности
    axes[0,0].bar(['Обычная', 'AI-enhanced'], 
                  [np.mean([r['total_return'] for r in normal_results]),
                   np.mean([r['total_return'] for r in ai_results])],
                  color=['blue', 'green'], alpha=0.7)
    axes[0,0].set_title('Средняя доходность (%)')
    axes[0,0].set_ylabel('Доходность (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # График количества сделок
    axes[0,1].bar(['Обычная', 'AI-enhanced'], 
                  [sum(r['total_trades'] for r in normal_results),
                   sum(r['total_trades'] for r in ai_results)],
                  color=['blue', 'green'], alpha=0.7)
    axes[0,1].set_title('Общее количество сделок')
    axes[0,1].set_ylabel('Количество сделок')
    axes[0,1].grid(True, alpha=0.3)
    
    # График просадки
    axes[1,0].bar(['Обычная', 'AI-enhanced'], 
                  [np.mean([r.get('max_drawdown', 0) for r in normal_results]),
                   np.mean([r.get('max_drawdown', 0) for r in ai_results])],
                  color=['red', 'orange'], alpha=0.7)
    axes[1,0].set_title('Средняя максимальная просадка (%)')
    axes[1,0].set_ylabel('Просадка (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # График Sharpe ratio
    axes[1,1].bar(['Обычная', 'AI-enhanced'], 
                  [np.mean([r.get('sharpe_ratio', 0) for r in normal_results]),
                   np.mean([r.get('sharpe_ratio', 0) for r in ai_results])],
                  color=['purple', 'cyan'], alpha=0.7)
    axes[1,1].set_title('Средний коэффициент Шарпа')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 График сравнения сохранен: models/strategy_comparison.png")

def main():
    """Основная функция тестирования"""
    
    # Тестируем на нескольких файлах из февраля 2023
    test_files = [
        "data/BTCUSDT_2023-02-01.csv.gz",
        "data/BTCUSDT_2023-02-02.csv.gz", 
        "data/BTCUSDT_2023-02-03.csv.gz",
        "data/BTCUSDT_2023-02-04.csv.gz",
        "data/BTCUSDT_2023-02-05.csv.gz",
    ]
    
    # Проверяем наличие файлов
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Тестовые файлы не найдены!")
        return
    
    print(f"🧪 Найдено {len(existing_files)} файлов для тестирования")
    
    # Сравниваем стратегии
    normal_results, ai_results = compare_strategies(existing_files)
    
    print(f"\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    main()

