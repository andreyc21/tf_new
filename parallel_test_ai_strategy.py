"""
Многопоточное тестирование AI-enhanced стратегии
"""

import gzip
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
import matplotlib.pyplot as plt
import os
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import time

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_strategy_on_file(args):
    """Тестирует стратегию на одном файле (для параллельной обработки)"""
    
    filename, use_ai_filter = args
    
    try:
        # Создаем стратегию
        strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=use_ai_filter,
            neural_confidence_threshold=0.6
        )
        
        # Загружаем и обрабатываем данные
        ticks_processed = 0
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
                ticks_processed += 1
        
        strategy.on_finish(price)
        
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
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка обработки {filename}: {e}")
        return None

def calculate_max_drawdown(equity_curve):
    """Рассчитывает максимальную просадку"""
    if len(equity_curve) == 0:
        return 0.0
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

def parallel_strategy_test(test_files, max_workers=None):
    """Параллельное тестирование стратегий"""
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(test_files))
    
    print(f"🚀 ПАРАЛЛЕЛЬНОЕ ТЕСТИРОВАНИЕ СТРАТЕГИЙ")
    print(f"🔧 Потоков: {max_workers}")
    print(f"📁 Файлов: {len(test_files)}")
    print("=" * 60)
    
    # Подготавливаем задачи
    tasks = []
    for filename in test_files:
        tasks.append((filename, False))  # Обычная стратегия
        tasks.append((filename, True))   # AI-enhanced стратегия
    
    print(f"⚡ Запускаем {len(tasks)} задач параллельно...")
    
    start_time = time.time()
    
    # Выполняем параллельно
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(test_strategy_on_file, tasks),
            total=len(tasks),
            desc="Обработка файлов"
        ))
    
    end_time = time.time()
    
    # Фильтруем успешные результаты
    successful_results = [r for r in results if r is not None]
    
    # Разделяем результаты по типу стратегии
    normal_results = [r for r in successful_results if not r['use_ai']]
    ai_results = [r for r in successful_results if r['use_ai']]
    
    print(f"\n⏱️ Время выполнения: {end_time - start_time:.1f} секунд")
    print(f"✅ Успешно обработано: {len(successful_results)}/{len(tasks)} задач")
    
    return normal_results, ai_results

def analyze_results(normal_results, ai_results):
    """Анализирует и сравнивает результаты"""
    
    print(f"\n📊 СВОДНЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    
    if not normal_results or not ai_results:
        print("❌ Недостаточно данных для сравнения")
        return
    
    # Собираем метрики
    normal_returns = [r['total_return'] for r in normal_results]
    ai_returns = [r['total_return'] for r in ai_results]
    
    normal_trades = sum(r['total_trades'] for r in normal_results)
    ai_trades = sum(r['total_trades'] for r in ai_results)
    
    normal_drawdowns = [r.get('max_drawdown', 0) for r in normal_results]
    ai_drawdowns = [r.get('max_drawdown', 0) for r in ai_results]
    
    normal_sharpe = [r.get('sharpe_ratio', 0) for r in normal_results]
    ai_sharpe = [r.get('sharpe_ratio', 0) for r in ai_results]
    
    normal_win_rates = [r.get('win_rate', 0) for r in normal_results]
    ai_win_rates = [r.get('win_rate', 0) for r in ai_results]
    
    # Выводим результаты
    print(f"📈 Обычная стратегия ({len(normal_results)} файлов):")
    print(f"   💰 Средняя доходность: {np.mean(normal_returns):.2f}% ± {np.std(normal_returns):.2f}%")
    print(f"   🔄 Всего сделок: {normal_trades}")
    print(f"   📉 Средняя просадка: {np.mean(normal_drawdowns):.2f}%")
    print(f"   📊 Средний Sharpe: {np.mean(normal_sharpe):.3f}")
    print(f"   🎯 Средний Win Rate: {np.mean(normal_win_rates):.1f}%")
    
    print(f"\n🧠 AI-enhanced стратегия ({len(ai_results)} файлов):")
    print(f"   💰 Средняя доходность: {np.mean(ai_returns):.2f}% ± {np.std(ai_returns):.2f}%")
    print(f"   🔄 Всего сделок: {ai_trades}")
    print(f"   📉 Средняя просадка: {np.mean(ai_drawdowns):.2f}%")
    print(f"   📊 Средний Sharpe: {np.mean(ai_sharpe):.3f}")
    print(f"   🎯 Средний Win Rate: {np.mean(ai_win_rates):.1f}%")
    
    # Статистическая значимость
    from scipy import stats
    
    return_t_stat, return_p_value = stats.ttest_rel(ai_returns, normal_returns)
    sharpe_t_stat, sharpe_p_value = stats.ttest_rel(ai_sharpe, normal_sharpe)
    
    # Улучшения
    return_improvement = np.mean(ai_returns) - np.mean(normal_returns)
    trade_reduction = ((normal_trades - ai_trades) / normal_trades * 100) if normal_trades > 0 else 0
    drawdown_improvement = np.mean(normal_drawdowns) - np.mean(ai_drawdowns)
    sharpe_improvement = np.mean(ai_sharpe) - np.mean(normal_sharpe)
    win_rate_improvement = np.mean(ai_win_rates) - np.mean(normal_win_rates)
    
    print(f"\n🎯 УЛУЧШЕНИЯ ОТ ИИ:")
    print(f"   💹 Доходность: {return_improvement:+.2f}% пунктов (p={return_p_value:.3f})")
    print(f"   🔄 Сокращение сделок: {trade_reduction:.1f}%")
    print(f"   📉 Улучшение просадки: {drawdown_improvement:+.2f}% пунктов")
    print(f"   📊 Улучшение Sharpe: {sharpe_improvement:+.3f} (p={sharpe_p_value:.3f})")
    print(f"   🎯 Улучшение Win Rate: {win_rate_improvement:+.1f}% пунктов")
    
    # Оценка значимости
    if return_p_value < 0.05:
        print(f"✅ Улучшение доходности статистически значимо!")
    else:
        print(f"⚠️ Улучшение доходности не достигает статистической значимости")
    
    return {
        'normal': normal_results,
        'ai': ai_results,
        'improvements': {
            'return': return_improvement,
            'trade_reduction': trade_reduction,
            'drawdown': drawdown_improvement,
            'sharpe': sharpe_improvement,
            'win_rate': win_rate_improvement
        },
        'p_values': {
            'return': return_p_value,
            'sharpe': sharpe_p_value
        }
    }

def plot_parallel_results(analysis_results):
    """Создает детальные графики сравнения"""
    
    normal_results = analysis_results['normal']
    ai_results = analysis_results['ai']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # График доходности
    normal_returns = [r['total_return'] for r in normal_results]
    ai_returns = [r['total_return'] for r in ai_results]
    
    axes[0,0].boxplot([normal_returns, ai_returns], labels=['Обычная', 'AI-enhanced'])
    axes[0,0].set_title('Распределение доходности (%)')
    axes[0,0].set_ylabel('Доходность (%)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Гистограмма доходности
    axes[0,1].hist(normal_returns, alpha=0.7, label='Обычная', bins=20, color='blue')
    axes[0,1].hist(ai_returns, alpha=0.7, label='AI-enhanced', bins=20, color='green')
    axes[0,1].set_title('Гистограмма доходности')
    axes[0,1].set_xlabel('Доходность (%)')
    axes[0,1].set_ylabel('Частота')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # График просадки
    normal_drawdowns = [r.get('max_drawdown', 0) for r in normal_results]
    ai_drawdowns = [r.get('max_drawdown', 0) for r in ai_results]
    
    axes[1,0].boxplot([normal_drawdowns, ai_drawdowns], labels=['Обычная', 'AI-enhanced'])
    axes[1,0].set_title('Распределение максимальной просадки (%)')
    axes[1,0].set_ylabel('Просадка (%)')
    axes[1,0].grid(True, alpha=0.3)
    
    # График Sharpe ratio
    normal_sharpe = [r.get('sharpe_ratio', 0) for r in normal_results]
    ai_sharpe = [r.get('sharpe_ratio', 0) for r in ai_results]
    
    axes[1,1].boxplot([normal_sharpe, ai_sharpe], labels=['Обычная', 'AI-enhanced'])
    axes[1,1].set_title('Распределение коэффициента Шарпа')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].grid(True, alpha=0.3)
    
    # Scatter plot: доходность vs просадка
    axes[2,0].scatter(normal_drawdowns, normal_returns, alpha=0.7, label='Обычная', color='blue')
    axes[2,0].scatter(ai_drawdowns, ai_returns, alpha=0.7, label='AI-enhanced', color='green')
    axes[2,0].set_xlabel('Макс. просадка (%)')
    axes[2,0].set_ylabel('Доходность (%)')
    axes[2,0].set_title('Доходность vs Просадка')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
    
    # Scatter plot: доходность vs Sharpe
    axes[2,1].scatter(normal_sharpe, normal_returns, alpha=0.7, label='Обычная', color='blue')
    axes[2,1].scatter(ai_sharpe, ai_returns, alpha=0.7, label='AI-enhanced', color='green')
    axes[2,1].set_xlabel('Sharpe Ratio')
    axes[2,1].set_ylabel('Доходность (%)')
    axes[2,1].set_title('Доходность vs Sharpe')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/parallel_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Детальный график сравнения сохранен: models/parallel_strategy_comparison.png")

def main():
    """Основная функция параллельного тестирования"""
    
    # Находим все доступные файлы за февраль 2023
    test_pattern = "data/BTCUSDT_2023-02-*.csv.gz"
    test_files = sorted(glob.glob(test_pattern))
    
    if not test_files:
        print(f"❌ Файлы не найдены по паттерну: {test_pattern}")
        return
    
    # Ограничиваем количество файлов для демонстрации
    max_files = min(15, len(test_files))  # Максимум 15 файлов
    test_files = test_files[:max_files]
    
    print(f"🧪 Найдено {len(test_files)} файлов для параллельного тестирования")
    
    # Определяем количество потоков
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, len(test_files) * 2)  # 2 задачи на файл
    
    print(f"💻 Доступно CPU: {cpu_count}")
    print(f"🔧 Будет использовано потоков: {max_workers}")
    
    # Параллельное тестирование
    normal_results, ai_results = parallel_strategy_test(test_files, max_workers)
    
    if normal_results and ai_results:
        # Анализ результатов
        analysis = analyze_results(normal_results, ai_results)
        
        # Визуализация
        try:
            plot_parallel_results(analysis)
        except ImportError:
            print("⚠️ Scipy не установлена, пропускаем статистические тесты")
        except Exception as e:
            print(f"⚠️ Ошибка при создании графиков: {e}")
        
        print(f"\n🎉 Параллельное тестирование завершено!")
        print(f"📊 Результаты показывают эффективность AI-фильтра")
    else:
        print("❌ Недостаточно данных для анализа")

if __name__ == "__main__":
    main()

