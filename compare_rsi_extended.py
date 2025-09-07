#!/usr/bin/env python

import gzip
import csv
import glob
import time
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_strategy_on_file(filename, strategy_params):
    """Тестирует стратегию на одном файле"""
    strategy = RSIStrategyBase(**strategy_params)
    
    try:
        with gzip.open(filename, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
            strategy.on_finish(price)
        
        return {
            'filename': filename,
            'equity': strategy.equity,
            'trades': len(strategy.trades),
            'sharpe': strategy.sharpe(),
            'candles': len(strategy.candles),
            'entries': len(strategy.entry_points),
            'exits': len(strategy.exit_points),
            'pnl_percent': (strategy.equity - 1.0) * 100,
            'success': True
        }
    except Exception as e:
        return {
            'filename': filename,
            'error': str(e),
            'success': False
        }

def compare_rsi_variants_extended(pattern, max_files=60):
    """Сравнивает варианты RSI на расширенном периоде"""
    
    files = sorted(glob.glob(pattern))[:max_files]
    
    if not files:
        print(f"❌ Файлы не найдены по паттерну: {pattern}")
        return
    
    print(f"📊 РАСШИРЕННОЕ СРАВНЕНИЕ RSI ВАРИАНТОВ")
    print(f"Паттерн: {pattern}")
    print(f"Файлов для анализа: {len(files)}")
    print(f"Период: {len(files)} дней (~{len(files)/30:.1f} месяца)")
    print("=" * 70)
    
    variants = [
        {"name": "TA-Lib RSI", "params": {}},
        {"name": "Custom RSI", "params": {"use_custom_rsi": True}},
        {"name": "Dual RSI", "params": {"use_dual_rsi": True}}
    ]
    
    all_results = {}
    
    for variant in variants:
        print(f"\n🔬 Тестируем: {variant['name']}")
        print("-" * 50)
        
        start_time = time.time()
        variant_results = []
        cumulative_equity = 1.0
        
        for i, filename in enumerate(files, 1):
            print(f"[{i:2d}/{len(files)}] {filename.split('/')[-1]}", end=" ")
            
            result = test_strategy_on_file(filename, variant['params'])
            
            if result['success']:
                cumulative_equity *= result['equity']
                result['cumulative_equity'] = cumulative_equity
                variant_results.append(result)
                
                print(f"-> {result['pnl_percent']:+6.2f}% (Кумул: {cumulative_equity:.4f})")
            else:
                print(f"-> ❌ {result['error']}")
        
        test_time = time.time() - start_time
        
        # Статистика по варианту
        if variant_results:
            successful_days = len(variant_results)
            total_trades = sum(r['trades'] for r in variant_results)
            avg_daily_return = sum(r['pnl_percent'] for r in variant_results) / successful_days
            profitable_days = len([r for r in variant_results if r['pnl_percent'] > 0])
            win_rate = profitable_days / successful_days * 100
            total_return = (cumulative_equity - 1.0) * 100
            avg_sharpe = sum(r['sharpe'] for r in variant_results) / successful_days
            
            print(f"\n📊 СТАТИСТИКА {variant['name']}:")
            print(f"⏱️  Время выполнения: {test_time:.1f} сек")
            print(f"✅ Успешных дней: {successful_days}/{len(files)}")
            print(f"💰 Итоговая доходность: {total_return:+6.2f}%")
            print(f"📈 Средняя дневная: {avg_daily_return:+6.2f}%")
            print(f"🎯 Винрейт: {win_rate:.1f}% ({profitable_days}/{successful_days})")
            print(f"🔄 Всего сделок: {total_trades}")
            print(f"📊 Средний Sharpe: {avg_sharpe:.2f}")
            print(f"💎 Финальная Equity: {cumulative_equity:.4f}")
            
            all_results[variant['name']] = {
                'results': variant_results,
                'total_return': total_return,
                'avg_daily_return': avg_daily_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'avg_sharpe': avg_sharpe,
                'cumulative_equity': cumulative_equity,
                'successful_days': successful_days,
                'test_time': test_time
            }
    
    # Сравнительная таблица
    if all_results:
        print(f"\n" + "=" * 70)
        print(f"📊 ИТОГОВОЕ СРАВНЕНИЕ ({len(files)} дней)")
        print("=" * 70)
        print(f"{'Вариант':<15} {'Доходность':<12} {'Винрейт':<9} {'Сделок':<8} {'Sharpe':<8} {'Время':<7}")
        print("-" * 70)
        
        for name, stats in all_results.items():
            print(f"{name:<15} {stats['total_return']:+6.2f}%     {stats['win_rate']:5.1f}%    "
                  f"{stats['total_trades']:<8} {stats['avg_sharpe']:6.2f}  {stats['test_time']:5.1f}s")
        
        # Определяем лучший вариант
        best_return = max(all_results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(all_results.items(), key=lambda x: x[1]['avg_sharpe'])
        best_winrate = max(all_results.items(), key=lambda x: x[1]['win_rate'])
        
        print(f"\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"💰 По доходности: {best_return[0]} ({best_return[1]['total_return']:+.2f}%)")
        print(f"📊 По Sharpe: {best_sharpe[0]} ({best_sharpe[1]['avg_sharpe']:.2f})")
        print(f"🎯 По винрейту: {best_winrate[0]} ({best_winrate[1]['win_rate']:.1f}%)")
        
        # Анализ различий в торговле
        print(f"\n🔍 АНАЛИЗ ТОРГОВОЙ АКТИВНОСТИ:")
        for name, stats in all_results.items():
            trades_per_day = stats['total_trades'] / stats['successful_days']
            print(f"{name}: {trades_per_day:.1f} сделок/день")
    
    return all_results

def analyze_dual_rsi_signals(files_sample):
    """Анализирует различия в сигналах Dual RSI"""
    print(f"\n🔬 АНАЛИЗ СИГНАЛОВ DUAL RSI")
    print("-" * 40)
    
    # Берём несколько файлов для детального анализа
    sample_files = files_sample[:5]
    
    for filename in sample_files:
        print(f"\n📊 Анализ: {filename.split('/')[-1]}")
        
        strategy = RSIStrategyBase(use_dual_rsi=True)
        
        try:
            with gzip.open(filename, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    strategy.on_tick(price, dt, volume)
                strategy.on_finish(price)
            
            # Сравниваем последние значения RSI
            if len(strategy.rsi_values) >= 10 and len(strategy.rsi_custom_values) >= 10:
                talib_rsi = strategy.rsi_values[-10:]
                custom_rsi = strategy.rsi_custom_values[-10:]
                
                differences = [abs(t - c) for t, c in zip(talib_rsi, custom_rsi)]
                avg_diff = sum(differences) / len(differences)
                max_diff = max(differences)
                
                print(f"  Средняя разница RSI: {avg_diff:.2f}")
                print(f"  Максимальная разница: {max_diff:.2f}")
                print(f"  Сделок: {len(strategy.trades)}, Equity: {strategy.equity:.4f}")
        
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

if __name__ == '__main__':
    import sys
    
    # По умолчанию тестируем март-апрель 2025
    pattern = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_2025-0[34]-*.csv.gz"
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    print(f"🎯 Паттерн файлов: {pattern}")
    print(f"📅 Максимум файлов: {max_files}")
    
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ Файлы не найдены. Попробуйте другой паттерн.")
        print(f"Примеры:")
        print(f"  python {sys.argv[0]} 'data/BTCUSDT_2025-03-*.csv.gz' 31")
        print(f"  python {sys.argv[0]} 'data/BTCUSDT_2025-0[34]-*.csv.gz' 60")
        exit(1)
    
    print(f"📂 Найдено файлов: {len(files)}")
    
    # Основное сравнение
    results = compare_rsi_variants_extended(pattern, max_files)
    
    # Детальный анализ Dual RSI
    if files:
        analyze_dual_rsi_signals(files)
