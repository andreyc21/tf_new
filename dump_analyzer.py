#!/usr/bin/env python3.12
"""
Анализатор дебаг-дампов торгового бота
Помогает понять состояние бота и найти проблемы в RSI расчетах
"""

import json
import sys
import os
from datetime import datetime
import glob

def load_dump(filename):
    """Загружает дамп из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки {filename}: {e}")
        return None

def analyze_rsi(dump_data):
    """Анализирует RSI данные"""
    print("📊 RSI АНАЛИЗ")
    print("=" * 50)
    
    bot_state = dump_data.get('bot_state', {})
    strategy_state = dump_data.get('strategy_state', {})
    manual_rsi = dump_data.get('manual_rsi_check', {})
    
    print(f"Текущий RSI бота: {bot_state.get('last_rsi', 'N/A')}")
    print(f"Последний RSI стратегии: {manual_rsi.get('last_strategy_rsi', 'N/A')}")
    print(f"Пересчитанный RSI: {manual_rsi.get('calculated_rsi', 'N/A')}")
    
    # Проверяем консистентность
    if manual_rsi.get('calculated_rsi') and manual_rsi.get('last_strategy_rsi'):
        diff = abs(manual_rsi['calculated_rsi'] - manual_rsi['last_strategy_rsi'])
        if diff > 0.01:
            print(f"⚠️ Расхождение в RSI: {diff:.4f}")
        else:
            print("✅ RSI расчеты консистентны")
    
    # Показываем последние цены закрытия
    closes = manual_rsi.get('closes_used', [])
    if closes:
        print(f"\nПоследние {len(closes)} цен закрытия:")
        for i, close in enumerate(closes):
            print(f"   {i+1:2d}. {close:8.2f}")
    
    # Анализируем RSI значения
    rsi_values = dump_data.get('rsi_values', [])
    if rsi_values:
        print(f"\nПоследние 10 RSI значений:")
        for i, rsi in enumerate(rsi_values[-10:], len(rsi_values)-9):
            print(f"   {i:3d}. {rsi:6.2f}")
        
        # Статистика RSI
        min_rsi = min(rsi_values)
        max_rsi = max(rsi_values)
        avg_rsi = sum(rsi_values) / len(rsi_values)
        
        print(f"\nRSI статистика:")
        print(f"   Мин: {min_rsi:.2f}")
        print(f"   Макс: {max_rsi:.2f}")
        print(f"   Среднее: {avg_rsi:.2f}")
        print(f"   Всего значений: {len(rsi_values)}")

def analyze_candles(dump_data):
    """Анализирует свечи"""
    print("\n🕯️ АНАЛИЗ СВЕЧЕЙ")
    print("=" * 50)
    
    recent_candles = dump_data.get('recent_candles', [])
    current_candle = dump_data.get('current_candle')
    
    print(f"Исторических свечей: {len(recent_candles)}")
    if current_candle:
        print("✅ Есть текущая свеча")
    else:
        print("❌ Нет текущей свечи")
    
    # Показываем последние 5 свечей
    if recent_candles:
        print(f"\nПоследние 5 исторических свечей:")
        for i, candle in enumerate(recent_candles[-5:], len(recent_candles)-4):
            dt = datetime.fromisoformat(candle['start_time'].replace('Z', '+00:00'))
            print(f"   {i:2d}. {dt.strftime('%H:%M')} | O:{candle['open']:8.1f} H:{candle['high']:8.1f} L:{candle['low']:8.1f} C:{candle['close']:8.1f}")
    
    # Показываем текущую свечу
    if current_candle:
        dt = datetime.fromisoformat(current_candle['start_time'].replace('Z', '+00:00'))
        print(f"\nТекущая свеча:")
        print(f"   {dt.strftime('%H:%M')} | O:{current_candle['open']:8.1f} H:{current_candle['high']:8.1f} L:{current_candle['low']:8.1f} C:{current_candle['close']:8.1f} (текущая)")

def analyze_positions(dump_data):
    """Анализирует позиции и сигналы"""
    print("\n💼 АНАЛИЗ ПОЗИЦИЙ")
    print("=" * 50)
    
    bot_state = dump_data.get('bot_state', {})
    strategy_state = dump_data.get('strategy_state', {})
    
    position_names = {0: "БЕЗ ПОЗИЦИИ", 1: "ЛОНГ", -1: "ШОРТ"}
    
    bot_pos = bot_state.get('position', 0)
    strategy_pos = strategy_state.get('position', 0)
    last_signal = bot_state.get('last_signal', 0)
    
    print(f"Позиция бота: {position_names.get(bot_pos, bot_pos)}")
    print(f"Позиция стратегии: {position_names.get(strategy_pos, strategy_pos)}")
    print(f"Последний сигнал: {last_signal}")
    
    if bot_pos != strategy_pos:
        print("⚠️ Позиции бота и стратегии не совпадают!")
    else:
        print("✅ Позиции синхронизированы")
    
    # Анализируем торговые точки
    entry_points = dump_data.get('entry_points', [])
    exit_points = dump_data.get('exit_points', [])
    
    print(f"\nТорговая активность:")
    print(f"   Входов в позиции: {len(entry_points)}")
    print(f"   Выходов из позиций: {len(exit_points)}")
    
    # Показываем последние торговые точки
    if entry_points:
        print(f"\nПоследние 3 входа:")
        for entry in entry_points[-3:]:
            dt = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))
            print(f"   {dt.strftime('%H:%M')} - вход по цене {entry['price']}")
    
    if exit_points:
        print(f"\nПоследние 3 выхода:")
        for exit in exit_points[-3:]:
            dt = datetime.fromisoformat(exit['time'].replace('Z', '+00:00'))
            print(f"   {dt.strftime('%H:%M')} - выход по цене {exit['price']}")

def analyze_strategy_params(dump_data):
    """Анализирует параметры стратегии"""
    print("\n⚙️ ПАРАМЕТРЫ СТРАТЕГИИ")
    print("=" * 50)
    
    strategy_state = dump_data.get('strategy_state', {})
    
    print(f"RSI период: {strategy_state.get('rsi_period', 'N/A')}")
    print(f"RSI покупка: {strategy_state.get('rsi_buy', 'N/A')}")
    print(f"RSI продажа: {strategy_state.get('rsi_sell', 'N/A')}")
    print(f"BB период: {strategy_state.get('bb_period', 'N/A')}")
    print(f"Эквити: {strategy_state.get('equity', 'N/A')}")
    print(f"Всего сделок: {strategy_state.get('total_trades', 'N/A')}")

def analyze_dump_file(filename):
    """Анализирует один дамп-файл"""
    print(f"🔍 АНАЛИЗ ДАМПА: {filename}")
    print("=" * 80)
    
    dump_data = load_dump(filename)
    if not dump_data:
        return
    
    # Общая информация
    timestamp = dump_data.get('timestamp', 'N/A')
    signal_name = dump_data.get('signal_name', 'N/A')
    
    print(f"Время создания: {timestamp}")
    print(f"Сигнал: {signal_name}")
    
    # Анализ по разделам
    analyze_strategy_params(dump_data)
    analyze_positions(dump_data)
    analyze_rsi(dump_data)
    analyze_candles(dump_data)
    
    print("\n" + "=" * 80)

def find_dump_files():
    """Находит все дамп-файлы"""
    pattern = "debug_dump_*.json"
    files = glob.glob(pattern)
    files.sort(reverse=True)  # новые сначала
    return files

def main():
    print("🔍 АНАЛИЗАТОР ДЕБАГ-ДАМПОВ")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        # Анализируем конкретный файл
        filename = sys.argv[1]
        if os.path.exists(filename):
            analyze_dump_file(filename)
        else:
            print(f"❌ Файл {filename} не найден")
    else:
        # Ищем все дамп-файлы
        dump_files = find_dump_files()
        
        if not dump_files:
            print("❌ Дамп-файлы не найдены")
            print("💡 Создайте дамп командой: kill -USR1 <pid_бота>")
            return
        
        print(f"📁 Найдено дамп-файлов: {len(dump_files)}")
        print()
        
        # Показываем список файлов
        for i, filename in enumerate(dump_files[:5], 1):
            file_time = os.path.getmtime(filename)
            dt = datetime.fromtimestamp(file_time)
            print(f"   {i}. {filename} ({dt.strftime('%H:%M:%S')})")
        
        if len(dump_files) > 5:
            print(f"   ... и еще {len(dump_files) - 5} файлов")
        
        print()
        choice = input("Введите номер файла для анализа (или Enter для последнего): ").strip()
        
        if choice == "":
            analyze_dump_file(dump_files[0])
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(dump_files):
                    analyze_dump_file(dump_files[idx])
                else:
                    print("❌ Неверный номер файла")
            except ValueError:
                print("❌ Введите число")

if __name__ == "__main__":
    main()
