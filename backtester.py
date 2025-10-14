import gzip
import csv
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def run_backtest(files, strategy_params=None, plot=False, plot_plotly=False, collect_training_data=False, training_data_file=None, log_trades=True):
    """
    🚀 Универсальная функция бэктестирования
    
    Args:
        files: Строка (один файл), список файлов или паттерн (glob)
        strategy_params: Параметры стратегии
        plot: Показывать графики (только для одного файла)
        collect_training_data: Собирать данные для обучения нейронной сети
        training_data_file: Файл для сохранения обучающих данных
    
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
    
    # Создаем функцию логирования сделок
    trade_log = []
    
    def log_trade_callback(trade_info):
        """📝 Callback для логирования сделок в бэктестере"""
        trade_log.append(trade_info)
        
        # Выводим информацию о сделке
        trade_type = trade_info['type']
        price = trade_info['price']
        timestamp = trade_info['timestamp']
        equity = trade_info.get('equity', 1.0)
        
        # Эмодзи для разных типов сделок
        emoji_map = {
            'open_long': '🟢 LONG',
            'close_long': '🔴 CLOSE LONG',
            'open_short': '🔴 SHORT', 
            'close_short': '🟢 CLOSE SHORT',
            'stop_loss': '❌ STOP LOSS'
        }
        
        emoji = emoji_map.get(trade_type, '📊')
        
        # Форматируем дополнительную информацию
        extra_info = ""
        if 'pnl' in trade_info:
            pnl_pct = trade_info['pnl'] * 100
            extra_info += f" | PnL: {pnl_pct:+.2f}%"
        if 'entry_price' in trade_info and trade_info['entry_price']:
            extra_info += f" | Entry: ${trade_info['entry_price']:.2f}"
        if 'stop_loss_price' in trade_info and trade_info['stop_loss_price']:
            extra_info += f" | SL: ${trade_info['stop_loss_price']:.2f}"
        
        print(f"{emoji} ${price:.2f} | Equity: {equity:.4f}{extra_info} | {timestamp}")
    
    # Создаем стратегию
    if strategy_params is None:
        strategy_params = {}
    
    # Добавляем логирование сделок если нужно
    if log_trades:
        strategy_params['trade_logger'] = log_trade_callback
    
    strategy = RSIStrategyBase(**strategy_params)
    
    # 📚 Включаем сбор обучающих данных если нужно
    if collect_training_data:
        strategy.enable_training_data_collection(training_data_file)
        print(f"📚 Сбор обучающих данных включен")
    
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
    
    # 📚 Сохраняем собранные обучающие данные
    if collect_training_data and strategy.training_data_collector:
        saved_file = strategy.save_training_data()
        if saved_file:
            print(f"💾 Обучающие данные сохранены: {saved_file}")
    
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
    if plot:
        plot_strategy(strategy)
    
    # Показываем интерактивные Plotly графики
    if plot_plotly:
        if len(file_list) == 1:
            filename = os.path.basename(file_list[0]).replace('.csv.gz', '')
            plot_strategy_plotly(strategy, filename_prefix=filename)
        else:
            plot_strategy_plotly(strategy, filename_prefix="multi_file_backtest")
    
    # Добавляем trade_log к результатам стратегии
    strategy._trade_log = trade_log
    
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
        stop_loss_points = [(t, p) for t, p in strategy.stop_loss_points if t >= candle_times[0] and t <= candle_times[-1]]
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
            ax0.scatter([t for t, _ in entry_points], [p for _, p in entry_points], marker='^', color='blue', s=60, label='Entry', zorder=5)
        if exit_points:
            ax0.scatter([t for t, _ in exit_points], [p for _, p in exit_points], marker='v', color='orange', s=60, label='Exit', zorder=5)
        
        # 💥 Сработавшие стоп-лоссы (отдельно от обычных выходов!)
        if stop_loss_points:
            ax0.scatter([t for t, _ in stop_loss_points], [p for _, p in stop_loss_points], 
                       marker='X', color='red', s=80, label='Stop Loss', zorder=8, 
                       linewidths=2, edgecolors='darkred', alpha=0.9)
        
        # 🛡️ Добавляем установленные стоп-лоссы из логов
        if hasattr(strategy, '_trade_log'):
            stop_loss_set_points_log = []  # Установленные
            
            for trade in strategy._trade_log:
                timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                
                # Показываем только стоп-лоссы в текущем окне
                if timestamp >= candle_times[0] and timestamp <= candle_times[-1]:
                    if 'stop_loss_price' in trade and trade['stop_loss_price']:
                        # Установленный стоп-лосс
                        stop_loss_set_points_log.append((timestamp, trade['stop_loss_price']))
            
            # Установленные стоп-лоссы (оранжевые щиты)
            if stop_loss_set_points_log:
                stop_set_times = [t for t, _ in stop_loss_set_points_log]
                stop_set_prices = [p for _, p in stop_loss_set_points_log]
                ax0.scatter(stop_set_times, stop_set_prices, marker='H', color='orange', s=100, 
                           label='SL Set', zorder=6, linewidths=2, 
                           edgecolors='darkorange', alpha=0.8)
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


def plot_strategy_plotly(strategy, filename_prefix="backtest"):
    """
    🚀 Интерактивные графики бэктестера на Plotly с отображением стоп-лоссов
    
    Args:
        strategy: Объект стратегии с результатами
        filename_prefix: Префикс для файла HTML
    """
    
    if not strategy.candles:
        print("❌ Нет данных для построения графиков")
        return
    
    # Подготовка данных
    candle_times = [c.start_time for c in strategy.candles]
    candle_opens = [c.open for c in strategy.candles]
    candle_highs = [c.high for c in strategy.candles]
    candle_lows = [c.low for c in strategy.candles]
    candle_closes = [c.close for c in strategy.candles]
    candle_volumes = [c.volume for c in strategy.candles]
    
    rsi_values = strategy.rsi_values
    bb_values = strategy.bb_values
    equity_curve = strategy.equity_curve
    
    # Извлекаем Bollinger Bands
    bb_ma = [bb[0] if bb else None for bb in bb_values]
    bb_upper = [bb[1] if bb else None for bb in bb_values]
    bb_lower = [bb[2] if bb else None for bb in bb_values]
    
    # Подготавливаем данные о сделках
    entry_points = strategy.entry_points
    exit_points = strategy.exit_points
    stop_loss_hit_points = strategy.stop_loss_points  # 💥 Используем отдельный список сработавших стоп-лоссов
    
    # 📊 Извлекаем установленные стоп-лоссы из логов сделок
    stop_loss_set_points = []  # Установленные стоп-лоссы
    
    if hasattr(strategy, '_trade_log'):
        for trade in strategy._trade_log:
            timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            
            if 'stop_loss_price' in trade and trade['stop_loss_price']:
                # Это установка стоп-лосса при открытии позиции
                stop_loss_set_points.append((timestamp, trade['stop_loss_price'], 'set'))
    
    # Создаем субплоты
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            '📊 Свечи + Bollinger Bands + Входы/Выходы + Стоп-лоссы',
            '📈 RSI (14)',
            '📊 Объемы',
            '💰 Equity Curve'
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # 1. Японские свечи
    fig.add_trace(
        go.Candlestick(
            x=candle_times,
            open=candle_opens,
            high=candle_highs,
            low=candle_lows,
            close=candle_closes,
            name="BTCUSDT",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # 2. Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=candle_times,
            y=bb_upper,
            mode='lines',
            line=dict(color='rgba(128, 0, 128, 0.5)', width=1, dash='dot'),
            name='BB Upper',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=candle_times,
            y=bb_lower,
            mode='lines',
            line=dict(color='rgba(128, 0, 128, 0.5)', width=1, dash='dot'),
            name='BB Lower',
            fill='tonexty',
            fillcolor='rgba(128, 0, 128, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=candle_times,
            y=bb_ma,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='BB MA',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 3. Входы в позицию
    if entry_points:
        entry_times = [t for t, _ in entry_points]
        entry_prices = [p for _, p in entry_points]
        fig.add_trace(
            go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=8,
                    color='#00ff00',
                    line=dict(color='darkgreen', width=2)
                ),
                name='📈 Входы',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 4. Выходы из позиции
    if exit_points:
        exit_times = [t for t, _ in exit_points]
        exit_prices = [p for _, p in exit_points]
        fig.add_trace(
            go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=8,
                    color='#ff6600',
                    line=dict(color='darkorange', width=2)
                ),
                name='📉 Выходы',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 5. 🛡️ Установленные стоп-лоссы (оранжевые щиты)
    if stop_loss_set_points:
        stop_set_times = [t for t, _, _ in stop_loss_set_points]
        stop_set_prices = [p for _, p, _ in stop_loss_set_points]
        fig.add_trace(
            go.Scatter(
                x=stop_set_times,
                y=stop_set_prices,
                mode='markers',
                marker=dict(
                    symbol='hexagram',  # Звезда Давида - выделяющийся символ
                    size=12,
                    color='#ff8800',
                    line=dict(color='darkorange', width=2),
                    opacity=0.8
                ),
                name='🛡️ SL установлен',
                showlegend=True,
                hovertemplate='<b>🛡️ Стоп-лосс установлен</b><br>' +
                             'Цена: $%{y:.2f}<br>' +
                             'Время: %{x}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 6. 💥 Сработавшие стоп-лоссы (красные взрывы)
    if stop_loss_hit_points:
        stop_hit_times = [t for t, _ in stop_loss_hit_points]
        stop_hit_prices = [p for _, p in stop_loss_hit_points]
        fig.add_trace(
            go.Scatter(
                x=stop_hit_times,
                y=stop_hit_prices,
                mode='markers',
                marker=dict(
                    symbol='x',  # Крестик - более компактный символ
                    size=12,
                    color='#ff0000',
                    line=dict(color='darkred', width=2),
                    opacity=0.9
                ),
                name='💥 SL сработал',
                showlegend=True,
                hovertemplate='<b>💥 Стоп-лосс сработал!</b><br>' +
                             'Цена: $%{y:.2f}<br>' +
                             'Время: %{x}<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 7. RSI
    fig.add_trace(
        go.Scatter(
            x=candle_times,
            y=rsi_values,
            mode='lines',
            line=dict(color='orange', width=2),
            name='RSI (14)',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # RSI зоны
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, row=2, col=1)
    
    # 8. Объемы
    colors = ['green' if candle_closes[i] >= candle_opens[i] else 'red' for i in range(len(candle_closes))]
    fig.add_trace(
        go.Bar(
            x=candle_times,
            y=candle_volumes,
            marker_color=colors,
            name='Объем',
            opacity=0.7,
            showlegend=True
        ),
        row=3, col=1
    )
    
    # 9. Equity Curve
    fig.add_trace(
        go.Scatter(
            x=candle_times[:len(equity_curve)],
            y=equity_curve,
            mode='lines',
            line=dict(color='purple', width=3),
            name='Equity',
            showlegend=True
        ),
        row=4, col=1
    )
    
    # Настройка макета
    fig.update_layout(
        title=f"📊 Интерактивный бэктест - {filename_prefix}",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Настройка осей
    fig.update_yaxes(title_text="Цена ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Объем", row=3, col=1)
    fig.update_yaxes(title_text="Equity", row=4, col=1)
    fig.update_xaxes(title_text="Время", row=4, col=1)
    
    # Сохранение и показ
    html_filename = f"{filename_prefix}_interactive.html"
    fig.write_html(html_filename)
    print(f"📊 Интерактивный график сохранен: {html_filename}")
    
    # Показываем график
    fig.show()
    
    return html_filename


if __name__ == '__main__':
    import sys
    import glob
    
    # Парсим опции
    plot = '--plot' in sys.argv
    plot_plotly = '--plot-plotly' in sys.argv
    collect_training_data = '--collect-data' in sys.argv
    log_trades = '--no-log-trades' not in sys.argv  # По умолчанию включено
    
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
        files = ["data/BTCUSDT_2025-03-01.csv.gz"]
    
    print(f'--- Бэктест: {" ".join(files) if len(files) > 1 else files[0]} ---')
    run_backtest(files, plot=plot, plot_plotly=plot_plotly, collect_training_data=collect_training_data, log_trades=log_trades) 