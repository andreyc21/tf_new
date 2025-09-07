#!/usr/bin/env python

import gzip
import csv
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase

def timestamp_to_dt(ts):
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def run_strategy_test(filename):
    """Простой тест стратегии для замера времени"""
    strategy = RSIStrategyBase()
    
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['price'])
            volume = float(row['volume'])
            dt = timestamp_to_dt(row['timestamp'])
            strategy.on_tick(price, dt, volume)
        strategy.on_finish(price)
    
    print(f"Результат: Equity={strategy.equity:.4f}, Сделок={len(strategy.trades)}")
    return strategy

if __name__ == '__main__':
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_2025-03-01.csv.gz"
    run_strategy_test(filename)
