import gzip
import csv
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def run_backtest_on_file(filename, strategy_params=None, plot=True, verbose=True):
    if strategy_params is None:
        strategy_params = {}
    strategy = RSIStrategyBase(**strategy_params)
    
    tick_count = 0
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            strategy.on_tick(price, dt, volume)
            tick_count += 1
        strategy.on_finish(price)
    
    if verbose:
        # 📊 Реалистичная статистика с учетом комиссий
        stats = strategy.get_realistic_stats()
        
        print(f'Файл: {filename}')
        print(f'🏭 РЕАЛИСТИЧНЫЙ БЭКТЕСТ (с отложенными ордерами):')
        print(f'  📈 Sharpe: {stats["sharpe_ratio"]:.4f}')
        print(f'  💰 Equity: {stats["final_equity"]:.4f} ({stats["total_return_pct"]:+.2f}%)')
        print(f'  🔄 Сделок: {stats["total_trades"]}')
        print(f'  📊 Свечей: {len(strategy.candles)}')
        print(f'  ⚡ Тиков: {tick_count}')
        print(f'  🎯 Входов: {len(strategy.entry_points)}')
        print(f'  🚪 Выходов: {len(strategy.exit_points)}')
        print(f'  💸 Отступ лимитных ордеров: {stats["limit_order_offset_pct"]:.3f}%')
        print(f'  💳 Комиссия мейкера: {stats["maker_fee_pct"]:.3f}%')
        
        # 🏭 Статистика исполнения отложенных ордеров
        print(f'  📋 Всего отложенных ордеров: {stats["total_orders"]}')
        print(f'  ✅ Исполнено ордеров: {stats["executed_orders"]}')
        print(f'  ❌ Цена не дошла: {stats["missed_orders"]}')
        print(f'  📊 Процент исполнения: {stats["execution_rate_pct"]:.1f}%')
        
        if stats["total_trades"] > 0:
            print(f'  🧾 Общие комиссии: {stats["total_fees_pct"]:.3f}%')
            print(f'  📝 Средняя комиссия за сделку: {stats["avg_fee_per_trade_pct"]:.3f}%')
        
    if plot:
        plot_strategy(strategy)
    return strategy

def plot_strategy(strategy, window=100):
    total = len(strategy.candles)
    i = 0
    while i < total:
        candles = strategy.candles[i:i+window]
        candle_times = [c.start_time for c in candles]
        candle_opens = [c.open for c in candles]
        candle_highs = [c.high for c in candles]
        candle_lows = [c.low for c in candles]
        candle_closes = [c.close for c in candles]
        closes = candle_closes
        rsi_values = [strategy.rsi_values[j+i] for j in range(len(closes))]
        bb_values = [strategy.bb_values[j+i] for j in range(len(closes))]
        equity_curve = strategy.equity_curve[i:i+window]
        # Обрезаем equity_curve до длины свечей
        equity_curve = equity_curve[:len(candle_times)]
        entry_points = [(t, p) for t, p in strategy.entry_points if t >= candle_times[0] and t <= candle_times[-1]]
        exit_points = [(t, p) for t, p in strategy.exit_points if t >= candle_times[0] and t <= candle_times[-1]]
        # --- График ---
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        # 1. Свечи с входами/выходами и BB
        ax0 = axs[0]
        width = 2
        for k in range(len(candles)):
            color = 'green' if candle_closes[k] >= candle_opens[k] else 'red'
            ax0.plot([candle_times[k], candle_times[k]], [candle_lows[k], candle_highs[k]], color=color, linewidth=1)
            ax0.add_patch(plt.Rectangle((mdates.date2num(candle_times[k]) - width/2880, min(candle_opens[k], candle_closes[k])),
                                        width/1440, abs(candle_closes[k] - candle_opens[k]),
                                        color=color, alpha=0.7))
        # BB
        bb_ma = [bb[0] for bb in bb_values]
        bb_upper = [bb[1] for bb in bb_values]
        bb_lower = [bb[2] for bb in bb_values]
        ax0.plot(candle_times, bb_ma, color='blue', linestyle='--', label='BB MA')
        ax0.plot(candle_times, bb_upper, color='purple', linestyle=':', label='BB Upper')
        ax0.plot(candle_times, bb_lower, color='purple', linestyle=':', label='BB Lower')
        if entry_points:
            ax0.scatter([t for t, _ in entry_points], [p for _, p in entry_points], marker='^', color='blue', label='Entry', zorder=5)
        if exit_points:
            ax0.scatter([t for t, _ in exit_points], [p for _, p in exit_points], marker='v', color='orange', label='Exit', zorder=5)
        ax0.set_ylabel('Price (5m candles)')
        ax0.set_title(f'5m Candles с входами/выходами и BB ({i+1}-{i+len(candles)})')
        ax0.legend()
        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # 2. RSI
        axs[1].plot(candle_times, rsi_values, label='RSI(14)', color='orange')
        axs[1].axhline(30, color='green', linestyle='--', alpha=0.5)
        axs[1].axhline(70, color='red', linestyle='--', alpha=0.5)
        axs[1].set_ylabel('RSI')
        axs[1].legend()
        axs[1].set_title('RSI(14) (по свечам)')
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # 3. Equity
        axs[2].plot(candle_times, equity_curve, label='Equity Curve', color='green')
        axs[2].set_ylabel('Equity')
        axs[2].set_xlabel('Time')
        axs[2].legend()
        axs[2].set_title('Equity Curve (по свечам)')
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.tight_layout()
        plt.show()
        i += window

def run_multiple_backtests(pattern="data/BTCUSDT_2024-07-*.csv.gz", max_files=10, strategy_params=None):
    """Запускает бэктесты на нескольких файлах без визуализации"""
    import glob
    
    files = sorted(glob.glob(pattern))[:max_files]
    
    print(f"📊 Массовый бэктест")
    print(f"Паттерн: {pattern}")
    print(f"Найдено файлов: {len(files)}")
    print(f"Тестируем первые {min(len(files), max_files)} файлов...")
    print("=" * 80)
    
    results = []
    total_equity = 1.0
    
    for i, filename in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(filename)}")
        try:
            strategy = run_backtest_on_file(filename, strategy_params, plot=False, verbose=False)
            stats = strategy.get_realistic_stats()  # 🏭 Реалистичная статистика
            
            result = {
                'filename': filename,
                'sharpe': stats['sharpe_ratio'],
                'equity': stats['final_equity'],
                'trades_count': stats['total_trades'],
                'candles_count': len(strategy.candles),
                'entry_points': len(strategy.entry_points),
                'exit_points': len(strategy.exit_points),
                'pnl_percent': stats['total_return_pct'],
                'total_fees_pct': stats.get('total_fees_pct', 0),
                'limit_offset_pct': stats['limit_order_offset_pct'],
                'maker_fee_pct': stats['maker_fee_pct'],
                'total_orders': stats['total_orders'],
                'executed_orders': stats['executed_orders'],
                'missed_orders': stats['missed_orders'],
                'execution_rate_pct': stats['execution_rate_pct']
            }
            
            # Кумулятивная доходность
            total_equity *= strategy.equity
            result['cumulative_equity'] = total_equity
            
            results.append(result)
            
            print(f"  📈 Sharpe: {result['sharpe']:8.4f}")
            print(f"  💰 Equity: {result['equity']:8.4f} ({result['pnl_percent']:+6.2f}%)")
            print(f"  🔄 Сделок: {result['trades_count']:3d}")
            print(f"  💸 Комиссии: {result['total_fees_pct']:5.2f}%")
            print(f"  📋 Ордеров: {result['executed_orders']}/{result['total_orders']} ({result['execution_rate_pct']:.0f}%)")
            print(f"  📊 Кумул.: {result['cumulative_equity']:8.4f}")
            
        except Exception as e:
            print(f"  ❌ ОШИБКА: {e}")
            results.append({'filename': filename, 'error': str(e)})
    
    # Общая статистика
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        print("\n" + "=" * 80)
        print("📊 ИТОГОВАЯ СТАТИСТИКА (РЕАЛИСТИЧНЫЙ БЭКТЕСТ):")
        print(f"Успешных тестов: {len(successful_results)}/{len(results)}")
        
        # Показываем параметры торговли
        if successful_results:
            first_result = successful_results[0]
            print(f"🏭 Параметры торговли:")
            print(f"  💸 Отступ лимитных ордеров: {first_result['limit_offset_pct']:.3f}%")
            print(f"  💳 Комиссия мейкера: {first_result['maker_fee_pct']:.3f}%")
        
        sharpe_values = [r['sharpe'] for r in successful_results]
        equity_values = [r['equity'] for r in successful_results]
        pnl_values = [r['pnl_percent'] for r in successful_results]
        fees_values = [r['total_fees_pct'] for r in successful_results]
        execution_rates = [r['execution_rate_pct'] for r in successful_results]
        total_orders_sum = sum([r['total_orders'] for r in successful_results])
        executed_orders_sum = sum([r['executed_orders'] for r in successful_results])
        missed_orders_sum = sum([r['missed_orders'] for r in successful_results])
        
        print(f"Средний Sharpe: {sum(sharpe_values)/len(sharpe_values):8.4f}")
        print(f"Медианный Sharpe: {sorted(sharpe_values)[len(sharpe_values)//2]:8.4f}")
        print(f"Мин/Макс Sharpe: {min(sharpe_values):8.4f} / {max(sharpe_values):8.4f}")
        
        print(f"Средняя дневная доходность: {sum(pnl_values)/len(pnl_values):+6.2f}%")
        print(f"Медианная доходность: {sorted(pnl_values)[len(pnl_values)//2]:+6.2f}%")
        print(f"Мин/Макс доходность: {min(pnl_values):+6.2f}% / {max(pnl_values):+6.2f}%")
        
        profitable_days = len([p for p in pnl_values if p > 0])
        print(f"Прибыльных дней: {profitable_days}/{len(pnl_values)} ({profitable_days/len(pnl_values)*100:.1f}%)")
        
        # Статистика по комиссиям
        if fees_values:
            avg_fees = sum(fees_values) / len(fees_values)
            total_fees = sum(fees_values)
            print(f"Средние комиссии за день: {avg_fees:.3f}%")
            print(f"Общие комиссии за период: {total_fees:.3f}%")
        
        # 🏭 Статистика исполнения отложенных ордеров
        if total_orders_sum > 0:
            overall_execution_rate = (executed_orders_sum / total_orders_sum) * 100
            avg_execution_rate = sum(execution_rates) / len(execution_rates)
            print(f"📋 Всего отложенных ордеров: {total_orders_sum}")
            print(f"✅ Исполнено ордеров: {executed_orders_sum}")
            print(f"❌ Цена не дошла: {missed_orders_sum}")
            print(f"📊 Общий процент исполнения: {overall_execution_rate:.1f}%")
            print(f"📊 Средний процент исполнения: {avg_execution_rate:.1f}%")
        
        total_return = (total_equity - 1.0) * 100
        print(f"Итоговая кумулятивная доходность: {total_return:+6.2f}%")
        print(f"Итоговый Equity: {total_equity:.4f}")
        print("=" * 80)
    
    return results

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--multiple' or sys.argv[1] == '-m':
            # Массовое тестирование
            pattern = sys.argv[2] if len(sys.argv) > 2 else "data/BTCUSDT_2024-07-*.csv.gz"
            max_files = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            print(f'--- Массовый бэктест: {pattern} (макс {max_files} файлов) ---')
            run_multiple_backtests(pattern, max_files)
        else:
            # Одиночный файл
            filename = sys.argv[1]
            plot = '--no-plot' not in sys.argv
            print(f'--- Бэктест на {filename} ---')
            run_backtest_on_file(filename, plot=plot)
    else:
        # По умолчанию массовое тестирование июля 2024
        print('--- Массовый бэктест (по умолчанию: июль 2024, первые 10 файлов) ---')
        print('Для одиночного файла: python backtester.py <filename>')
        print('Для массового теста: python backtester.py --multiple <pattern> <max_files>')
        print('Для отключения графиков: python backtester.py <filename> --no-plot')
        print()
        run_multiple_backtests() 