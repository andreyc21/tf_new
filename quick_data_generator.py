"""
Быстрый генератор обучающих данных для нейронной сети
Оптимизированная версия для ускоренного сбора данных
"""

import gzip
import csv
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json
from tqdm import tqdm
import talib

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

class FastTrainingDataGenerator:
    def __init__(self, min_profit_threshold=0.002, max_loss_threshold=-0.01):
        self.min_profit_threshold = min_profit_threshold  # 0.2% минимальная прибыль
        self.max_loss_threshold = max_loss_threshold      # -1% максимальный убыток
        
    def process_file_fast(self, filename, max_signals_per_file=500):
        """Быстрая обработка файла с минимальными вычислениями"""
        
        print(f"⚡ Быстрая обработка: {os.path.basename(filename)}")
        
        # Читаем весь файл в pandas сразу
        try:
            df = pd.read_csv(filename, compression='gzip')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp')
            
            # Ресемплим в 5-минутные свечи
            df_resampled = df.set_index('datetime').resample('5T').agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            }).dropna()
            
            # Упрощаем колонки
            df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            
            if len(df_resampled) < 50:
                return []
            
            # Быстро вычисляем индикаторы
            close_prices = df_resampled['close'].values
            high_prices = df_resampled['high'].values
            low_prices = df_resampled['low'].values
            
            # RSI (используем TA-Lib для скорости)
            rsi = talib.RSI(close_prices, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # ATR
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Быстрый поиск сигналов
            training_samples = []
            
            for i in range(20, len(close_prices) - 10):  # Оставляем место для анализа результата
                current_rsi = rsi[i]
                current_price = close_prices[i]
                
                if np.isnan(current_rsi):
                    continue
                
                # Определяем сигналы
                is_buy_signal = current_rsi < 30
                is_sell_signal = current_rsi > 70
                
                if not (is_buy_signal or is_sell_signal):
                    continue
                
                # Быстрый анализ результата (смотрим на 10 свечей вперед)
                future_prices = close_prices[i+1:i+11]
                if len(future_prices) < 10:
                    continue
                
                if is_buy_signal:
                    # Для покупки
                    max_profit = max([(price - current_price) / current_price for price in future_prices])
                    max_loss = min([(price - current_price) / current_price for price in future_prices])
                    signal_type = 'buy'
                else:
                    # Для продажи
                    max_profit = max([(current_price - price) / current_price for price in future_prices])
                    max_loss = min([(current_price - price) / current_price for price in future_prices])
                    signal_type = 'sell'
                
                # Определяем прибыльность (более строгие критерии)
                if max_profit >= self.min_profit_threshold and max_loss > self.max_loss_threshold:
                    profitable = True
                elif max_loss <= self.max_loss_threshold:
                    profitable = False
                elif max_profit < self.min_profit_threshold and max_loss > self.max_loss_threshold:
                    # Слабый сигнал - тоже считаем убыточным
                    profitable = False
                else:
                    continue  # Нейтральный результат
                
                # Создаем упрощенные признаки
                features = [
                    current_rsi,                           # Текущий RSI
                    np.mean(rsi[i-5:i]) if i >= 5 else current_rsi,  # RSI за 5 периодов
                    (current_price - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if not np.isnan(bb_upper[i]) else 0.5,  # Позиция в BB
                    atr[i] / current_price if not np.isnan(atr[i]) else 0.01,  # Нормализованная ATR
                    np.std(close_prices[i-10:i]) / current_price if i >= 10 else 0.01,  # Волатильность
                    (close_prices[i] - close_prices[i-5]) / close_prices[i-5] if i >= 5 else 0,  # Изменение за 5 периодов
                ]
                
                training_samples.append({
                    'features': features,
                    'profitable': profitable,
                    'signal_type': signal_type,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'risk_reward': max_profit / abs(max_loss) if max_loss < 0 else float('inf'),
                    'rsi': current_rsi,
                    'timestamp': df_resampled.index[i].isoformat(),
                    'file': os.path.basename(filename)
                })
                
                if len(training_samples) >= max_signals_per_file:
                    break
            
            profitable_count = sum(1 for s in training_samples if s['profitable'])
            print(f"   ✅ Свечей: {len(close_prices)}, Сигналов: {len(training_samples)} (👍{profitable_count} 👎{len(training_samples)-profitable_count})")
            
            return training_samples
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return []
    
    def generate_training_data(self, data_pattern="data/BTCUSDT_2023-*.csv.gz", max_files=10, 
                              output_file="models/training_data.json"):
        """Быстрая генерация обучающих данных"""
        
        files = sorted(glob.glob(data_pattern))
        if max_files:
            files = files[:max_files]
        
        if not files:
            print(f"❌ Файлы не найдены: {data_pattern}")
            return []
        
        print(f"🚀 БЫСТРАЯ ГЕНЕРАЦИЯ ДАННЫХ")
        print(f"📁 Файлов к обработке: {len(files)}")
        print(f"⚡ Ожидаемое время: {len(files) * 10} секунд")
        
        all_training_data = []
        
        for i, filename in enumerate(tqdm(files, desc="Обработка")):
            try:
                file_data = self.process_file_fast(filename)
                all_training_data.extend(file_data)
                
            except Exception as e:
                print(f"❌ Ошибка {filename}: {e}")
                continue
        
        # Статистика
        total_signals = len(all_training_data)
        profitable_signals = sum(1 for s in all_training_data if s['profitable'])
        
        print(f"\n📊 РЕЗУЛЬТАТ:")
        print(f"📈 Всего сигналов: {total_signals:,}")
        print(f"✅ Прибыльных: {profitable_signals:,} ({profitable_signals/total_signals*100:.1f}%)")
        print(f"❌ Убыточных: {total_signals-profitable_signals:,}")
        
        if total_signals == 0:
            print("❌ Нет данных!")
            return []
        
        # Сохраняем
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_training_data, f, indent=2)
        
        print(f"💾 Сохранено: {output_file}")
        print(f"📁 Размер: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
        
        return all_training_data

def main():
    """Основная функция"""
    
    generator = FastTrainingDataGenerator()
    
    # Быстрая генерация на свежих данных 2024-2025
    training_data = generator.generate_training_data(
        data_pattern="data/BTCUSDT_2025-*.csv.gz",  # Используем 2025 год!
        max_files=30,  # 30 файлов = месяц свежих данных
        output_file="models/training_data_2025.json"  # Новый файл
    )
    
    if training_data:
        print(f"\n🎉 Готово! Можно обучать нейросеть")

if __name__ == "__main__":
    main()
