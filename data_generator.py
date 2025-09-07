"""
Генератор обучающих данных для нейронной сети
Анализирует исторические данные и создает dataset с прибыльными/убыточными сигналами
"""

import gzip
import csv
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
from neural_filter import NeuralSignalFilter
import json
from tqdm import tqdm
from config import RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

class TrainingDataGenerator:
    def __init__(self, lookback_candles=10, min_profit_threshold=0.002, max_loss_threshold=-0.01):
        self.lookback_candles = lookback_candles
        self.min_profit_threshold = min_profit_threshold  # 0.2% минимальная прибыль
        self.max_loss_threshold = max_loss_threshold      # -1% максимальный убыток
        
    def analyze_signal_outcome(self, strategy, signal_index, signal_type, lookback_periods=10):
        """Анализирует результат сигнала через lookback_periods свечей"""
        
        if signal_index + lookback_periods >= len(strategy.candles):
            return None  # Недостаточно данных для анализа
        
        entry_candle = strategy.candles[signal_index]
        entry_price = entry_candle.close
        
        # Анализируем цены в следующие lookback_periods свечей
        future_prices = [c.close for c in strategy.candles[signal_index+1:signal_index+1+lookback_periods]]
        
        if signal_type == 'buy':
            # Для покупки ищем максимальную прибыль и максимальный убыток
            max_profit = max([(price - entry_price) / entry_price for price in future_prices])
            max_loss = min([(price - entry_price) / entry_price for price in future_prices])
        else:  # sell
            # Для продажи инвертируем
            max_profit = max([(entry_price - price) / entry_price for price in future_prices])
            max_loss = min([(entry_price - price) / entry_price for price in future_prices])
        
        # Определяем, был ли сигнал прибыльным
        if max_profit >= self.min_profit_threshold:
            return {
                'profitable': True,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'risk_reward': max_profit / abs(max_loss) if max_loss < 0 else float('inf')
            }
        elif max_loss <= self.max_loss_threshold:
            return {
                'profitable': False,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'risk_reward': max_profit / abs(max_loss) if max_loss < 0 else 0
            }
        else:
            # Нейтральный результат - не включаем в обучающую выборку
            return None
    
    def extract_features_at_signal(self, strategy, signal_index):
        """Извлекает признаки на момент сигнала"""
        
        if signal_index < self.lookback_candles:
            return None  # Недостаточно исторических данных
        
        # Получаем исторические данные до сигнала
        rsi_hist = strategy.rsi_values[signal_index-self.lookback_candles:signal_index]
        bb_hist = strategy.bb_values[signal_index-self.lookback_candles:signal_index]
        atr_hist = strategy.atr_values[signal_index-self.lookback_candles:signal_index]
        vol_ratio_hist = strategy.volatility_ratios[signal_index-self.lookback_candles:signal_index]
        
        candles_hist = strategy.candles[signal_index-self.lookback_candles:signal_index]
        prices_hist = [c.close for c in candles_hist]
        
        # Используем нейронную сеть для извлечения признаков
        neural_filter = NeuralSignalFilter()
        features = neural_filter.prepare_features(
            rsi_hist, bb_hist, atr_hist, vol_ratio_hist, prices_hist, 
            lookback=self.lookback_candles  # Используем наш lookback
        )
        
        return features
    
    def process_file(self, filename, max_signals_per_file=1000):
        """Обрабатывает один файл с историческими данными"""
        
        print(f"📊 Обрабатываем файл: {os.path.basename(filename)}")
        
        # Создаем стратегию с оптимизированными параметрами
        strategy = RSIStrategyBase(use_custom_rsi=True)
        
        # Загружаем и обрабатываем данные
        ticks_processed = 0
        try:
            with gzip.open(filename, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    strategy.on_tick(price, dt, volume)
                    ticks_processed += 1
            
            strategy.on_finish(price)
            
        except Exception as e:
            print(f"❌ Ошибка обработки файла {filename}: {e}")
            return []
        
        print(f"   ✅ Обработано тиков: {ticks_processed:,}")
        print(f"   📈 Создано свечей: {len(strategy.candles)}")
        print(f"   🔄 Всего сделок: {len(strategy.trades)}")
        
        # Анализируем все потенциальные сигналы
        training_samples = []
        
        for i in range(self.lookback_candles, len(strategy.candles) - 10):
            candle = strategy.candles[i]
            rsi = strategy.rsi_values[i] if i < len(strategy.rsi_values) else 50
            bb = strategy.bb_values[i] if i < len(strategy.bb_values) else (None, None, None)
            
            # Проверяем условия для сигналов
            signals_to_check = []
            
            # Сигнал покупки (используем параметры из config.py)
            if rsi < RSI_BUY_THRESHOLD:  # RSI < 30
                signals_to_check.append(('buy', i))
            
            # Сигнал продажи (используем параметры из config.py)
            if rsi > RSI_SELL_THRESHOLD:  # RSI > 70
                signals_to_check.append(('sell', i))
            
            # Анализируем каждый сигнал
            for signal_type, signal_idx in signals_to_check:
                # Извлекаем признаки
                features = self.extract_features_at_signal(strategy, signal_idx)
                if features is None:
                    continue
                
                # Анализируем результат сигнала
                outcome = self.analyze_signal_outcome(strategy, signal_idx, signal_type)
                if outcome is None:
                    continue
                
                # Добавляем в обучающую выборку
                training_samples.append({
                    'features': features,
                    'profitable': outcome['profitable'],
                    'signal_type': signal_type,
                    'max_profit': outcome['max_profit'],
                    'max_loss': outcome['max_loss'],
                    'risk_reward': outcome['risk_reward'],
                    'rsi': rsi,
                    'timestamp': candle.start_time.isoformat(),
                    'file': os.path.basename(filename)
                })
                
                # Ограничиваем количество сигналов с одного файла
                if len(training_samples) >= max_signals_per_file:
                    break
            
            if len(training_samples) >= max_signals_per_file:
                break
        
        profitable_count = sum(1 for s in training_samples if s['profitable'])
        print(f"   📈 Прибыльных сигналов: {profitable_count}")
        print(f"   📉 Убыточных сигналов: {len(training_samples) - profitable_count}")
        
        return training_samples
    
    def generate_training_data(self, data_pattern="data/BTCUSDT_*.csv.gz", max_files=None, 
                              output_file="training_data.json"):
        """Генерирует обучающие данные из всех исторических файлов"""
        
        # Находим все файлы с данными
        files = sorted(glob.glob(data_pattern))
        if max_files:
            files = files[:max_files]
        
        if not files:
            print(f"❌ Файлы не найдены по паттерну: {data_pattern}")
            return
        
        print(f"🎯 Найдено файлов: {len(files)}")
        print(f"📊 Генерируем обучающие данные...")
        
        all_training_data = []
        
        # Обрабатываем каждый файл
        for i, filename in enumerate(tqdm(files, desc="Обработка файлов")):
            try:
                file_data = self.process_file(filename)
                all_training_data.extend(file_data)
                
                print(f"[{i+1}/{len(files)}] {os.path.basename(filename)}: +{len(file_data)} сигналов")
                
            except Exception as e:
                print(f"❌ Ошибка обработки {filename}: {e}")
                continue
        
        # Статистика
        total_signals = len(all_training_data)
        profitable_signals = sum(1 for s in all_training_data if s['profitable'])
        
        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"📈 Всего сигналов: {total_signals:,}")
        print(f"✅ Прибыльных: {profitable_signals:,} ({profitable_signals/total_signals*100:.1f}%)")
        print(f"❌ Убыточных: {total_signals-profitable_signals:,} ({(1-profitable_signals/total_signals)*100:.1f}%)")
        
        if total_signals == 0:
            print("❌ Нет данных для сохранения!")
            return
        
        # Сохраняем данные
        print(f"\n💾 Сохраняем данные в {output_file}...")
        
        # Конвертируем numpy массивы в списки для JSON
        for sample in all_training_data:
            if isinstance(sample['features'], np.ndarray):
                sample['features'] = sample['features'].tolist()
        
        with open(output_file, 'w') as f:
            json.dump(all_training_data, f, indent=2)
        
        print(f"✅ Данные сохранены: {output_file}")
        print(f"📁 Размер файла: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        
        return all_training_data

def main():
    """Основная функция для генерации обучающих данных"""
    
    generator = TrainingDataGenerator()
    
    # Генерируем данные из всех доступных файлов
    training_data = generator.generate_training_data(
        data_pattern="data/BTCUSDT_2023-*.csv.gz",  # Используем 2023 год для обучения
        max_files=30,  # Ограичиваем для быстрого тестирования (~1 месяц данных)
        output_file="models/training_data.json"
    )
    
    if training_data:
        print(f"\n🎉 Генерация обучающих данных завершена!")
        print(f"📊 Готово к обучению нейронной сети")

if __name__ == "__main__":
    main()

