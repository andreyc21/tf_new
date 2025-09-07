#!/usr/bin/env python

import gzip
import csv
import os
import glob
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
import itertools

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_strategy_params(files, strategy_params, max_files=50):
    """Тестирует стратегию с заданными параметрами на списке файлов"""
    results = []
    total_equity = 1.0
    
    for filename in files[:max_files]:
        try:
            strategy = RSIStrategyBase(**strategy_params)
            
            with gzip.open(filename, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    strategy.on_tick(price, dt, volume)
                strategy.on_finish(price)
            
            total_equity *= strategy.equity
            results.append({
                'filename': filename,
                'equity': strategy.equity,
                'trades': len(strategy.trades),
                'sharpe': strategy.sharpe()
            })
            
        except Exception as e:
            results.append({'filename': filename, 'error': str(e)})
    
    successful = [r for r in results if 'error' not in r]
    if not successful:
        return None
    
    avg_daily_return = sum(r['equity'] - 1.0 for r in successful) / len(successful) * 100
    profitable_days = len([r for r in successful if r['equity'] > 1.0])
    win_rate = profitable_days / len(successful) * 100
    total_return = (total_equity - 1.0) * 100
    
    return {
        'params': strategy_params,
        'total_equity': total_equity,
        'total_return_pct': total_return,
        'avg_daily_return_pct': avg_daily_return,
        'win_rate_pct': win_rate,
        'total_trades': sum(r['trades'] for r in successful),
        'tested_days': len(successful),
        'avg_sharpe': sum(r['sharpe'] for r in successful) / len(successful)
    }

def optimize_rsi_parameters(files, max_files=50):
    """Оптимизирует параметры RSI стратегии"""
    
    print("🔍 Оптимизация параметров RSI стратегии")
    print(f"Тестируем на {min(len(files), max_files)} файлах...")
    print("=" * 80)
    
    # Диапазоны параметров для тестирования (сокращенные для скорости)
    rsi_buy_values = [25, 30, 35]      # 3 значения вместо 4
    rsi_sell_values = [65, 70, 75]     # 3 значения вместо 4  
    rsi_periods = [10, 14, 18]         # 3 значения вместо 4
    
    best_result = None
    best_score = -float('inf')
    
    total_combinations = len(rsi_buy_values) * len(rsi_sell_values) * len(rsi_periods)
    current_combination = 0
    
    results = []
    
    for rsi_period in rsi_periods:
        for rsi_buy in rsi_buy_values:
            for rsi_sell in rsi_sell_values:
                current_combination += 1
                
                if rsi_sell <= rsi_buy:
                    continue
                
                params = {
                    'rsi_period': rsi_period,
                    'rsi_buy': rsi_buy,
                    'rsi_sell': rsi_sell,
                    'use_custom_rsi': True  # 🏆 Используем нашу выигрышную кастомную реализацию RSI!
                }
                
                print(f"[{current_combination}/{total_combinations}] RSI({rsi_period}): buy={rsi_buy}, sell={rsi_sell}")
                
                result = test_strategy_params(files, params, max_files)
                if result is None:
                    continue
                
                results.append(result)
                
                # Скоринг: комбинация общей доходности и винрейта
                score = result['total_return_pct'] * 0.7 + result['win_rate_pct'] * 0.3
                
                print(f"  📊 Итог: {result['total_return_pct']:+6.2f}% | Винрейт: {result['win_rate_pct']:.1f}% | Скор: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    print(f"  🏆 НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ!")
    
    print("\n" + "=" * 80)
    print("🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    
    # Сортируем по общей доходности
    results.sort(key=lambda x: x['total_return_pct'], reverse=True)
    
    print("\nТОП-5 по общей доходности:")
    for i, result in enumerate(results[:5]):
        params = result['params']
        print(f"{i+1}. RSI({params['rsi_period']}): {params['rsi_buy']}/{params['rsi_sell']} "
              f"-> {result['total_return_pct']:+6.2f}% | Винрейт: {result['win_rate_pct']:.1f}%")
    
    # Сортируем по винрейту
    results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
    
    print("\nТОП-5 по винрейту:")
    for i, result in enumerate(results[:5]):
        params = result['params']
        print(f"{i+1}. RSI({params['rsi_period']}): {params['rsi_buy']}/{params['rsi_sell']} "
              f"-> {result['total_return_pct']:+6.2f}% | Винрейт: {result['win_rate_pct']:.1f}%")
    
    if best_result:
        print(f"\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:")
        params = best_result['params']
        print(f"RSI период: {params['rsi_period']}")
        print(f"RSI покупка: {params['rsi_buy']}")
        print(f"RSI продажа: {params['rsi_sell']}")
        print(f"Ожидаемая доходность: {best_result['total_return_pct']:+.2f}%")
        print(f"Винрейт: {best_result['win_rate_pct']:.1f}%")
    
    return best_result, results

def analyze_current_strategy_problems(files, max_files=20):
    """Анализирует проблемы текущей стратегии"""
    
    print("🔍 АНАЛИЗ ПРОБЛЕМ ТЕКУЩЕЙ СТРАТЕГИИ")
    print("=" * 50)
    
    # Текущие параметры (оптимизированная стратегия)
    current_params = {'rsi_period': 14, 'rsi_buy': 30, 'rsi_sell': 70, 'use_custom_rsi': True}
    
    print("Текущие параметры:")
    print(f"- RSI период: {current_params['rsi_period']}")
    print(f"- RSI покупка: {current_params['rsi_buy']}")
    print(f"- RSI продажа: {current_params['rsi_sell']}")
    
    result = test_strategy_params(files, current_params, max_files)
    
    if result:
        print(f"\nРезультаты на {result['tested_days']} днях:")
        print(f"📊 Общая доходность: {result['total_return_pct']:+6.2f}%")
        print(f"📈 Средняя дневная: {result['avg_daily_return_pct']:+6.2f}%")
        print(f"🎯 Винрейт: {result['win_rate_pct']:.1f}%")
        print(f"🔄 Всего сделок: {result['total_trades']}")
        print(f"📊 Средний Sharpe: {result['avg_sharpe']:.2f}")
        
        print(f"\n❌ ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")
        
        if result['win_rate_pct'] < 50:
            print(f"- Низкий винрейт ({result['win_rate_pct']:.1f}%) - стратегия чаще проигрывает")
        
        if result['avg_daily_return_pct'] < 0:
            print(f"- Отрицательная средняя доходность ({result['avg_daily_return_pct']:+.2f}%)")
        
        if result['avg_sharpe'] < 0:
            print(f"- Отрицательный Sharpe ratio ({result['avg_sharpe']:.2f}) - плохое соотношение риск/доходность")
        
        avg_trades_per_day = result['total_trades'] / result['tested_days']
        if avg_trades_per_day > 15:
            print(f"- Слишком много сделок ({avg_trades_per_day:.1f} в день) - возможно овертрейдинг")
        elif avg_trades_per_day < 2:
            print(f"- Слишком мало сделок ({avg_trades_per_day:.1f} в день) - упускаем возможности")
    
    return result

if __name__ == '__main__':
    import sys
    
    # Получаем список файлов для тестирования
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "data/BTCUSDT_2024-07-*.csv.gz"
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"Файлы не найдены по паттерну: {pattern}")
        exit(1)
    
    print(f"Найдено файлов: {len(files)}")
    
    if len(sys.argv) > 2 and sys.argv[2] == '--analyze':
        # Анализ текущей стратегии
        analyze_current_strategy_problems(files, 30)
    else:
        # Оптимизация
        best_result, all_results = optimize_rsi_parameters(files, 30)
