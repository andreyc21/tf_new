"""
Сравнение стратегий на свежих данных (не использовавшихся для обучения)
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
from concurrent.futures import ThreadPoolExecutor
import time

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_single_file(args):
    """Тестирует один файл с обеими стратегиями"""
    
    filename, = args
    
    results = {}
    
    try:
        # 1. Тестируем обычную стратегию
        normal_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=False
        )
        
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                normal_strategy.on_tick(price, dt, volume)
        
        normal_strategy.on_finish(price)
        
        # 2. Тестируем AI-enhanced стратегию
        ai_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=True,
            neural_confidence_threshold=0.5
        )
        
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                ai_strategy.on_tick(price, dt, volume)
        
        ai_strategy.on_finish(price)
        
        # Сохраняем результаты
        results = {
            'filename': os.path.basename(filename),
            'normal': {
                'trades': len(normal_strategy.trades),
                'entries': len(normal_strategy.entry_points),
                'exits': len(normal_strategy.exit_points),
                'return': (normal_strategy.equity - 1.0) * 100,
                'equity_curve': normal_strategy.equity_curve.copy()
            },
            'ai': {
                'trades': len(ai_strategy.trades),
                'entries': len(ai_strategy.entry_points),
                'exits': len(ai_strategy.exit_points),
                'return': (ai_strategy.equity - 1.0) * 100,
                'equity_curve': ai_strategy.equity_curve.copy()
            }
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка обработки {filename}: {e}")
        return None

def test_fresh_period(data_pattern="data/BTCUSDT_2023-0[6-9]*.csv.gz", max_files=20):
    """Тестирует стратегии на свежих данных"""
    
    print("🚀 ТЕСТИРОВАНИЕ НА СВЕЖИХ ДАННЫХ")
    print("=" * 60)
    
    # Находим файлы за более поздние месяцы
    files = sorted(glob.glob(data_pattern))
    
    if not files:
        print(f"❌ Файлы не найдены по паттерну: {data_pattern}")
        return
    
    # Ограничиваем количество файлов
    files = files[:max_files]
    
    print(f"📁 Найдено файлов: {len(files)}")
    print(f"📅 Период: {os.path.basename(files[0])} - {os.path.basename(files[-1])}")
    print(f"🧠 AI обучался на январе 2023, тестируем на более поздних данных")
    
    # Подготавливаем задачи
    tasks = [(f,) for f in files]
    
    start_time = time.time()
    
    # Параллельная обработка
    print(f"\n⚡ Запускаем параллельную обработку...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(test_single_file, tasks))
    
    end_time = time.time()
    
    # Фильтруем успешные результаты
    successful_results = [r for r in results if r is not None]
    
    print(f"⏱️ Время обработки: {end_time - start_time:.1f} секунд")
    print(f"✅ Успешно обработано: {len(successful_results)}/{len(files)} файлов")
    
    if not successful_results:
        print("❌ Нет данных для анализа")
        return
    
    # Анализируем результаты
    analyze_fresh_results(successful_results)
    
    return successful_results

def analyze_fresh_results(results):
    """Анализирует результаты тестирования на свежих данных"""
    
    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ НА СВЕЖИХ ДАННЫХ:")
    print("=" * 60)
    
    # Собираем метрики
    normal_returns = [r['normal']['return'] for r in results]
    ai_returns = [r['ai']['return'] for r in results]
    
    normal_trades = [r['normal']['trades'] for r in results]
    ai_trades = [r['ai']['trades'] for r in results]
    
    total_normal_trades = sum(normal_trades)
    total_ai_trades = sum(ai_trades)
    
    # Статистика
    print(f"📈 ОБЫЧНАЯ СТРАТЕГИЯ:")
    print(f"   💰 Средняя доходность: {np.mean(normal_returns):.2f}% ± {np.std(normal_returns):.2f}%")
    print(f"   📊 Медианная доходность: {np.median(normal_returns):.2f}%")
    print(f"   🔄 Всего сделок: {total_normal_trades}")
    print(f"   📈 Прибыльных дней: {sum(1 for r in normal_returns if r > 0)}/{len(normal_returns)}")
    print(f"   📉 Убыточных дней: {sum(1 for r in normal_returns if r < 0)}/{len(normal_returns)}")
    
    print(f"\n🧠 AI-ENHANCED СТРАТЕГИЯ:")
    print(f"   💰 Средняя доходность: {np.mean(ai_returns):.2f}% ± {np.std(ai_returns):.2f}%")
    print(f"   📊 Медианная доходность: {np.median(ai_returns):.2f}%")
    print(f"   🔄 Всего сделок: {total_ai_trades}")
    print(f"   📈 Прибыльных дней: {sum(1 for r in ai_returns if r > 0)}/{len(ai_returns)}")
    print(f"   📉 Убыточных дней: {sum(1 for r in ai_returns if r < 0)}/{len(ai_returns)}")
    
    # Сравнение
    return_improvement = np.mean(ai_returns) - np.mean(normal_returns)
    trade_reduction = ((total_normal_trades - total_ai_trades) / total_normal_trades * 100) if total_normal_trades > 0 else 0
    
    print(f"\n🎯 СРАВНЕНИЕ (AI vs Обычная):")
    print("=" * 50)
    print(f"💹 Улучшение доходности: {return_improvement:+.2f}% пунктов")
    print(f"🔄 Изменение количества сделок: {-trade_reduction:+.1f}%")
    
    # Статистическая значимость
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(ai_returns, normal_returns)
        print(f"📊 Статистическая значимость: p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"✅ Различие статистически значимо (p < 0.05)")
        else:
            print(f"⚠️ Различие не достигает статистической значимости")
            
    except ImportError:
        print(f"⚠️ scipy не доступен для статистических тестов")
    
    # Win rate сравнение
    normal_wins = sum(1 for r in normal_returns if r > 0)
    ai_wins = sum(1 for r in ai_returns if r > 0)
    
    print(f"\n🏆 WIN RATE:")
    print(f"📈 Обычная стратегия: {normal_wins}/{len(normal_returns)} ({normal_wins/len(normal_returns)*100:.1f}%)")
    print(f"🧠 AI-enhanced: {ai_wins}/{len(ai_returns)} ({ai_wins/len(ai_returns)*100:.1f}%)")
    
    # Детальная таблица по дням
    print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ДНЯМ:")
    print(f"{'Файл':<25} {'Обычная':<10} {'AI':<10} {'Разница':<10} {'Сделки (N/AI)':<15}")
    print("-" * 80)
    
    for r in results[:10]:  # Показываем первые 10 дней
        diff = r['ai']['return'] - r['normal']['return']
        trades_info = f"{r['normal']['trades']}/{r['ai']['trades']}"
        print(f"{r['filename']:<25} {r['normal']['return']:>8.2f}% {r['ai']['return']:>8.2f}% {diff:>8.2f}% {trades_info:>13}")
    
    if len(results) > 10:
        print(f"... и еще {len(results)-10} дней")
    
    # Визуализация
    plot_fresh_comparison(results)

def plot_fresh_comparison(results):
    """Создает графики сравнения на свежих данных"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    normal_returns = [r['normal']['return'] for r in results]
    ai_returns = [r['ai']['return'] for r in results]
    
    # 1. Гистограмма доходности
    axes[0,0].hist(normal_returns, alpha=0.7, bins=20, label='Обычная', color='blue')
    axes[0,0].hist(ai_returns, alpha=0.7, bins=20, label='AI-enhanced', color='green')
    axes[0,0].set_title('Распределение дневной доходности')
    axes[0,0].set_xlabel('Доходность (%)')
    axes[0,0].set_ylabel('Частота')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Временной ряд доходности
    dates = [r['filename'] for r in results]
    x_pos = range(len(dates))
    
    axes[0,1].plot(x_pos, normal_returns, 'b-', alpha=0.7, label='Обычная')
    axes[0,1].plot(x_pos, ai_returns, 'g-', alpha=0.7, label='AI-enhanced')
    axes[0,1].set_title('Дневная доходность по времени')
    axes[0,1].set_xlabel('День')
    axes[0,1].set_ylabel('Доходность (%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Scatter plot: AI vs Обычная
    axes[1,0].scatter(normal_returns, ai_returns, alpha=0.7, color='purple')
    axes[1,0].plot([-5, 5], [-5, 5], 'r--', alpha=0.5)  # Диагональ
    axes[1,0].set_xlabel('Обычная стратегия (%)')
    axes[1,0].set_ylabel('AI-enhanced (%)')
    axes[1,0].set_title('Сравнение доходности по дням')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Кумулятивная доходность
    cumulative_normal = np.cumsum(normal_returns)
    cumulative_ai = np.cumsum(ai_returns)
    
    axes[1,1].plot(x_pos, cumulative_normal, 'b-', linewidth=2, label='Обычная')
    axes[1,1].plot(x_pos, cumulative_ai, 'g-', linewidth=2, label='AI-enhanced')
    axes[1,1].set_title('Кумулятивная доходность')
    axes[1,1].set_xlabel('День')
    axes[1,1].set_ylabel('Кумулятивная доходность (%)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/fresh_period_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 График сравнения на свежих данных: models/fresh_period_comparison.png")

def main():
    """Основная функция"""
    
    print("🔍 Поиск свежих данных для тестирования...")
    
    # Пробуем разные паттерны для поиска свежих данных
    patterns = [
        "data/BTCUSDT_2023-0[6-9]*.csv.gz",  # Июнь-сентябрь
        "data/BTCUSDT_2023-1[0-2]*.csv.gz",  # Октябрь-декабрь
        "data/BTCUSDT_2023-0[4-5]*.csv.gz",  # Апрель-май
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            print(f"✅ Найдены файлы по паттерну: {pattern}")
            results = test_fresh_period(pattern, max_files=15)
            if results:
                break
    else:
        print("❌ Не найдено подходящих файлов для тестирования")
        print("💡 Доступные файлы:")
        all_files = glob.glob("data/BTCUSDT_2023-*.csv.gz")
        for f in sorted(all_files)[:10]:
            print(f"   {os.path.basename(f)}")

if __name__ == "__main__":
    main()

