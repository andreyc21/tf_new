import gzip
import csv
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def run_backtest(files, strategy_params=None, plot=False):
    """
    🚀 Универсальная функция бэктестирования
    
    Args:
        files: Строка (один файл), список файлов или паттерн (glob)
        strategy_params: Параметры стратегии
        plot: Показывать графики (только для одного файла)
    
    Returns:
        strategy: Объект стратегии с результатами
    """
    import glob
    
    # Определяем список файлов
    if isinstance(files, str):
        if '*' in files or '?' in files or '[' in files:
            # Это glob паттерн
            file_list = sorted(glob.glob(files))
            if not file_list:
                raise ValueError(f"Паттерн {files} не соответствует ни одному файлу")
        else:
            # Это один файл
            file_list = [files]
    else:
        # Это уже список
        file_list = list(files)
    
    if not file_list:
        raise ValueError("Не указаны файлы для обработки")
    
    # Создаем стратегию
    if strategy_params is None:
        strategy_params = {}
    strategy = RSIStrategyBase(**strategy_params)
    
    # Обрабатываем файлы
    results = []
    total_ticks = 0
    
    # Выводим заголовок для множественных файлов
    if len(file_list) > 1:
        print(f"📊 Бэктест на {len(file_list)} файлах")
        print(f"🔄 Непрерывные буферы: ✅ Включены")
        print("=" * 60)
    
    for i, filename in enumerate(file_list, 1):
        if len(file_list) > 1:
            print(f"\n[{i}/{len(file_list)}] {os.path.basename(filename)}")
        
        # Всегда используем одну стратегию (непрерывные буферы)
        
        # Обрабатываем файл
        tick_count = 0
        last_price = None
        
        try:
            with gzip.open(filename, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    strategy.on_tick(price, dt, volume)
                    tick_count += 1
                    last_price = price
            
            total_ticks += tick_count
            
            # Сохраняем статистику по файлу
            if len(file_list) > 1:
                stats = strategy.get_realistic_stats()
                result = {
                    'filename': filename,
                    'tick_count': tick_count,
                    'stats': stats
                }
                results.append(result)
                
                # Краткий вывод для множественных файлов
                print(f"  📈 Sharpe: {stats['sharpe_ratio']:8.4f}")
                print(f"  💰 Equity: {stats['final_equity']:8.4f} ({stats['total_return_pct']:+6.2f}%)")
                print(f"  🔄 Сделок: {stats['total_trades']:3d}")
                print(f"  ⚡ Тиков: {tick_count:,}")
                
        except Exception as e:
            print(f"  ❌ ОШИБКА в {filename}: {e}")
            continue
    
    # Завершаем стратегию
    if last_price is not None:
        strategy.on_finish(last_price)
    
    # Всегда подробный вывод
    stats = strategy.get_realistic_stats()
    
    if len(file_list) == 1:
        # Подробная статистика для одного файла
        filename = file_list[0]
        
        print(f'📊 Файл: {os.path.basename(filename)}')
        print(f'🏭 РЕАЛИСТИЧНЫЙ БЭКТЕСТ (с отложенными ордерами):')
        print(f'  📈 Sharpe: {stats["sharpe_ratio"]:.4f}')
        print(f'  💰 Equity: {stats["final_equity"]:.4f} ({stats["total_return_pct"]:+.2f}%)')
        print(f'  🔄 Сделок: {stats["total_trades"]}')
        print(f'  📊 Свечей: {len(strategy.candles):,}')
        print(f'  ⚡ Тиков: {total_ticks:,}')
        print(f'  🎯 Входов: {len(strategy.entry_points)}')
        print(f'  🚪 Выходов: {len(strategy.exit_points)}')
        print(f'  💸 Отступ лимитных ордеров: {stats["limit_order_offset_pct"]:.3f}%')
        print(f'  💳 Комиссия мейкера: {stats["maker_fee_pct"]:.3f}%')
        print(f'  📋 Всего отложенных ордеров: {stats["total_orders"]}')
        print(f'  ✅ Исполнено ордеров: {stats["executed_orders"]}')
        print(f'  ❌ Цена не дошла: {stats["missed_orders"]}')
        print(f'  📊 Процент исполнения: {stats["execution_rate_pct"]:.1f}%')
        
        if stats["total_trades"] > 0:
            print(f'  🧾 Общие комиссии: {stats["total_fees_pct"]:.3f}%')
            print(f'  📝 Средняя комиссия за сделку: {stats["avg_fee_per_trade_pct"]:.3f}%')
        
        if 'stop_loss_triggered' in stats:
            print(f'  🛡️ Сработавших стоп-лоссов: {stats["stop_loss_triggered"]}')
            print(f'  📊 Доля стоп-лоссов: {stats["stop_loss_rate_pct"]:.1f}%')
    
    else:
        # Итоговая статистика для множественных файлов
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА:")
        
        successful_results = [r for r in results if 'stats' in r]
        print(f"Успешно обработано: {len(successful_results)}/{len(file_list)} файлов")
        print(f"Общее количество тиков: {total_ticks:,}")
        print(f"Итоговый Equity: {stats['final_equity']:.6f}")
        print(f"Итоговая доходность: {stats['total_return_pct']:+.2f}%")
        print(f"Общее количество сделок: {stats['total_trades']}")
        print(f"Итоговый Sharpe: {stats['sharpe_ratio']:.4f}")
        
        if stats["total_trades"] > 0:
            print(f"Общие комиссии: {stats['total_fees_pct']:.3f}%")
        
        if 'stop_loss_triggered' in stats:
            print(f"Сработавших стоп-лоссов: {stats['stop_loss_triggered']}")
    
    # Показываем графики только для одного файла
    if plot and len(file_list) == 1:
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


if __name__ == '__main__':
    import sys
    import glob
    
    # Парсим опции
    plot = '--plot' in sys.argv
    
    # Получаем все аргументы кроме опций
    file_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    # Определяем файлы
    if file_args:
        # Есть аргументы - раскрываем через glob и сортируем
        all_files = []
        for arg in file_args:
            matched = glob.glob(arg)
            if matched:
                all_files.extend(matched)
            else:
                all_files.append(arg)  # Добавляем как есть, если glob не сработал
        files = sorted(set(all_files))  # Убираем дубликаты и сортируем
    else:
        # Нет аргументов - используем дефолтный файл
        files = ["data/BTCUSDT_2025-01-01.csv.gz"]
    
    print(f'--- Бэктест: {" ".join(files) if len(files) > 1 else files[0]} ---')
    run_backtest(files, plot=plot) 