#!/usr/bin/env python

import gzip
import csv
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_rsi_variants(filename):
    """Тестирует все варианты RSI стратегии"""
    
    print(f"🧪 ТЕСТ ВАРИАНТОВ RSI СТРАТЕГИИ")
    print(f"Файл: {filename}")
    print("=" * 60)
    
    # Читаем данные один раз
    ticks = []
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            ticks.append((price, dt, volume))
    
    print(f"📊 Загружено тиков: {len(ticks):,}")
    
    variants = [
        {"name": "TA-Lib RSI", "params": {"use_custom_rsi": False}},  # Явно отключаем кастомный RSI
        {"name": "Custom RSI", "params": {"use_custom_rsi": True}},   # Наша выигрышная стратегия
        {"name": "Dual RSI", "params": {"use_dual_rsi": True}}        # Сравнение обеих
    ]
    
    results = []
    
    for variant in variants:
        print(f"\n🔬 Тестируем: {variant['name']}")
        print("-" * 40)
        
        strategy = RSIStrategyBase(**variant['params'])
        
        for price, dt, volume in ticks:
            strategy.on_tick(price, dt, volume)
        strategy.on_finish(price)
        
        result = {
            'name': variant['name'],
            'equity': strategy.equity,
            'trades': len(strategy.trades),
            'sharpe': strategy.sharpe(),
            'candles': len(strategy.candles),
            'entries': len(strategy.entry_points),
            'exits': len(strategy.exit_points)
        }
        
        results.append(result)
        
        print(f"💰 Equity: {result['equity']:.4f}")
        print(f"🔄 Сделок: {result['trades']}")
        print(f"📊 Sharpe: {result['sharpe']:.4f}")
        print(f"⬆️  Входов: {result['entries']}")
        print(f"⬇️  Выходов: {result['exits']}")
        
        # Показываем несколько последних значений RSI для сравнения
        if len(strategy.rsi_values) >= 5:
            print(f"📈 Последние RSI: {strategy.rsi_values[-5:]}")
            
        if hasattr(strategy, 'rsi_custom_values') and len(strategy.rsi_custom_values) >= 5:
            print(f"📈 Custom RSI: {strategy.rsi_custom_values[-5:]}")
    
    # Сравнительная таблица
    print(f"\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
    print("=" * 60)
    print(f"{'Вариант':<15} {'Equity':<8} {'Сделок':<7} {'Sharpe':<8} {'Входов':<7}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<15} {result['equity']:<8.4f} {result['trades']:<7} {result['sharpe']:<8.2f} {result['entries']:<7}")
    
    return results

if __name__ == '__main__':
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_2025-03-01.csv.gz"
    test_rsi_variants(filename)
