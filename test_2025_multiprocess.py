"""
Тестирование на данных 2025 года с многопроцессорностью и отладкой AI
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
from multiprocessing import Pool, cpu_count
import time
import pickle

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_single_file_with_debug(filename):
    """Тестирует один файл с обеими стратегиями и отладкой AI"""
    
    results = {'filename': os.path.basename(filename)}
    
    try:
        # 1. Тестируем обычную стратегию
        print(f"📊 Обрабатываем {os.path.basename(filename)} - обычная стратегия")
        
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
        
        # 2. Тестируем AI-enhanced стратегию с отладкой
        print(f"🧠 Обрабатываем {os.path.basename(filename)} - AI стратегия")
        
        ai_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=True,
            neural_confidence_threshold=0.5
        )
        
        # Счетчики для отладки AI
        ai_debug = {
            'potential_signals': 0,
            'ai_calls': 0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'ai_decisions': []
        }
        
        # Перехватываем метод для отладки
        original_on_tick = ai_strategy.on_tick
        
        def debug_on_tick(price, dt, volume=0):
            result = original_on_tick(price, dt, volume)
            
            # Проверяем AI решения
            if len(ai_strategy.candles) > 20 and len(ai_strategy.rsi_values) > 0:
                current_rsi = ai_strategy.rsi_values[-1]
                
                # Потенциальный сигнал?
                if (current_rsi < 30 or current_rsi > 70) and ai_strategy.position == 0:
                    ai_debug['potential_signals'] += 1
                    
                    if ai_strategy.use_neural_filter and ai_strategy.neural_filter:
                        ai_debug['ai_calls'] += 1
                        
                        try:
                            # Подготавливаем признаки
                            lookback = min(20, len(ai_strategy.rsi_values))
                            recent_rsi = ai_strategy.rsi_values[-lookback:]
                            recent_bb = ai_strategy.bb_values[-lookback:]
                            recent_atr = ai_strategy.atr_values[-lookback:]
                            recent_vol_ratio = ai_strategy.volatility_ratios[-lookback:]
                            recent_prices = [c.close for c in ai_strategy.candles[-lookback:]]
                            
                            features = ai_strategy.neural_filter.prepare_features(
                                recent_rsi, recent_bb, recent_atr, recent_vol_ratio, recent_prices
                            )
                            
                            if features is not None:
                                confidence = ai_strategy.neural_filter.predict(features)
                                neural_approved, neural_confidence = ai_strategy.neural_filter.should_trade(
                                    features, ai_strategy.neural_confidence_threshold
                                )
                                
                                # Сохраняем решение
                                decision = {
                                    'rsi': current_rsi,
                                    'confidence': confidence,
                                    'approved': neural_approved,
                                    'signal_type': 'buy' if current_rsi < 30 else 'sell'
                                }
                                ai_debug['ai_decisions'].append(decision)
                                
                                if neural_approved:
                                    ai_debug['ai_approved'] += 1
                                else:
                                    ai_debug['ai_rejected'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ AI ошибка в {os.path.basename(filename)}: {e}")
            
            return result
        
        ai_strategy.on_tick = debug_on_tick
        
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                ai_strategy.on_tick(price, dt, volume)
        
        ai_strategy.on_finish(price)
        
        # Сохраняем результаты
        results.update({
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
            },
            'ai_debug': ai_debug
        })
        
        print(f"✅ {os.path.basename(filename)}: Обычная={results['normal']['trades']} сделок, "
              f"AI={results['ai']['trades']} сделок, "
              f"AI отклонил {ai_debug['ai_rejected']}/{ai_debug['potential_signals']} сигналов")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка обработки {filename}: {e}")
        return {'filename': os.path.basename(filename), 'error': str(e)}

def test_2025_data_multiprocess(data_pattern="data/BTCUSDT_2025-*.csv.gz", max_files=10):
    """Многопроцессорное тестирование на данных 2025 года"""
    
    print("🚀 ТЕСТИРОВАНИЕ НА ДАННЫХ 2025 ГОДА (МНОГОПРОЦЕССОРНОЕ)")
    print("=" * 70)
    
    # Находим файлы за 2025 год
    files = sorted(glob.glob(data_pattern))
    
    if not files:
        print(f"❌ Файлы не найдены по паттерну: {data_pattern}")
        return
    
    # Ограничиваем количество файлов
    files = files[:max_files]
    
    print(f"📁 Найдено файлов: {len(files)}")
    print(f"📅 Период: {os.path.basename(files[0])} - {os.path.basename(files[-1])}")
    print(f"🧠 AI обучался на январе 2023, тестируем на 2025 году (2 года спустя!)")
    print(f"💻 Доступно CPU: {cpu_count()}")
    print(f"🔧 Будет использовано процессов: {min(cpu_count(), len(files))}")
    
    start_time = time.time()
    
    # МНОГОПРОЦЕССОРНАЯ обработка (вместо ThreadPoolExecutor)
    print(f"\n⚡ Запускаем многопроцессорную обработку...")
    
    with Pool(processes=min(cpu_count(), len(files))) as pool:
        results = pool.map(test_single_file_with_debug, files)
    
    end_time = time.time()
    
    # Фильтруем успешные результаты
    successful_results = [r for r in results if 'error' not in r and 'normal' in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"⏱️ Время обработки: {end_time - start_time:.1f} секунд")
    print(f"✅ Успешно обработано: {len(successful_results)}/{len(files)} файлов")
    
    if failed_results:
        print(f"❌ Ошибки в файлах: {[r['filename'] for r in failed_results]}")
    
    if not successful_results:
        print("❌ Нет данных для анализа")
        return
    
    # Анализируем результаты
    analyze_2025_results(successful_results)
    
    return successful_results

def analyze_2025_results(results):
    """Анализирует результаты тестирования на данных 2025 года"""
    
    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ НА ДАННЫХ 2025 ГОДА:")
    print("=" * 70)
    
    # Собираем метрики
    normal_returns = [r['normal']['return'] for r in results]
    ai_returns = [r['ai']['return'] for r in results]
    
    normal_trades = [r['normal']['trades'] for r in results]
    ai_trades = [r['ai']['trades'] for r in results]
    
    total_normal_trades = sum(normal_trades)
    total_ai_trades = sum(ai_trades)
    
    # AI отладочная информация
    total_potential_signals = sum(r['ai_debug']['potential_signals'] for r in results)
    total_ai_calls = sum(r['ai_debug']['ai_calls'] for r in results)
    total_ai_approved = sum(r['ai_debug']['ai_approved'] for r in results)
    total_ai_rejected = sum(r['ai_debug']['ai_rejected'] for r in results)
    
    print(f"🧠 AI ФИЛЬТРАЦИЯ АНАЛИЗ:")
    print(f"   🚨 Всего потенциальных сигналов: {total_potential_signals:,}")
    print(f"   🤖 Вызовов AI: {total_ai_calls:,}")
    print(f"   ✅ AI одобрил: {total_ai_approved:,}")
    print(f"   🚫 AI отклонил: {total_ai_rejected:,}")
    if total_potential_signals > 0:
        print(f"   📊 Процент фильтрации: {total_ai_rejected/total_potential_signals*100:.1f}%")
    
    # Статистика стратегий
    print(f"\n📈 ОБЫЧНАЯ СТРАТЕГИЯ:")
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
    
    # Детальная таблица
    print(f"\n📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ДНЯМ:")
    print(f"{'Файл':<25} {'Обычная':<10} {'AI':<10} {'Разница':<10} {'Сделки (N/AI)':<15} {'AI фильтр':<15}")
    print("-" * 100)
    
    for r in results:
        diff = r['ai']['return'] - r['normal']['return']
        trades_info = f"{r['normal']['trades']}/{r['ai']['trades']}"
        ai_filter_info = f"{r['ai_debug']['ai_rejected']}/{r['ai_debug']['potential_signals']}"
        
        print(f"{r['filename']:<25} {r['normal']['return']:>8.2f}% {r['ai']['return']:>8.2f}% "
              f"{diff:>8.2f}% {trades_info:>13} {ai_filter_info:>13}")
    
    # Проверяем, работает ли AI фильтр
    if total_ai_rejected == 0:
        print(f"\n⚠️ ВНИМАНИЕ: AI-фильтр не отклонил ни одного сигнала!")
        print(f"   Возможные причины:")
        print(f"   1. Модель обучена на старых данных (2023) и не работает на 2025")
        print(f"   2. Порог уверенности слишком низкий (текущий: 0.5)")
        print(f"   3. Признаки изменились за 2 года")
    elif total_ai_rejected > 0:
        print(f"\n✅ AI-фильтр работает! Отклонил {total_ai_rejected} из {total_potential_signals} сигналов")
        
        # Анализируем эффективность фильтрации
        if abs(return_improvement) > 0.1:  # Значимое улучшение
            if return_improvement > 0:
                print(f"🎉 AI-фильтр улучшил результаты на {return_improvement:.2f}%!")
            else:
                print(f"🔻 AI-фильтр ухудшил результаты на {abs(return_improvement):.2f}%")
        else:
            print(f"➖ AI-фильтр не дал значимого улучшения")

def main():
    """Основная функция"""
    
    print("🔍 Поиск данных за 2025 год...")
    
    # Тестируем на свежих данных 2025 года
    pattern = "data/BTCUSDT_2025-*.csv.gz"
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ Файлы за 2025 год не найдены: {pattern}")
        print("💡 Доступные файлы:")
        all_files = glob.glob("data/BTCUSDT_*.csv.gz")
        for f in sorted(all_files)[-10:]:
            print(f"   {os.path.basename(f)}")
        return
    
    print(f"✅ Найдено {len(files)} файлов за 2025 год")
    
    # Запускаем многопроцессорное тестирование
    results = test_2025_data_multiprocess(pattern, max_files=10)
    
    if results:
        print(f"\n🎉 ТЕСТИРОВАНИЕ НА 2025 ГОДУ ЗАВЕРШЕНО!")
        print(f"💻 Использовалась многопроцессорность для ускорения")
        print(f"🧠 AI-фильтр протестирован на данных через 2 года после обучения")

if __name__ == "__main__":
    main()

