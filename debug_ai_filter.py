"""
Отладка AI-фильтра - проверяем, почему он не фильтрует сигналы
"""

import gzip
import csv
import numpy as np
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
from neural_filter import NeuralSignalFilter
import os

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def debug_ai_filter():
    """Отладка AI-фильтра"""
    
    print("🔍 ОТЛАДКА AI-ФИЛЬТРА")
    print("=" * 50)
    
    # 1. Проверяем загрузку модели
    print("1️⃣ Проверка загрузки модели...")
    
    try:
        neural_filter = NeuralSignalFilter(
            model_path="models/neural_filter_fast.keras",
            scaler_path="models/scaler_fast.joblib"
        )
        
        if neural_filter.model is not None:
            print("✅ Модель загружена успешно")
        else:
            print("❌ Модель не загружена")
            return
            
        if neural_filter.scaler is not None:
            print("✅ Скалер загружен успешно")
        else:
            print("❌ Скалер не загружен")
            return
            
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # 2. Тестируем предсказания на тестовых данных
    print("\n2️⃣ Тестирование предсказаний...")
    
    # Создаем тестовые признаки (как в quick_data_generator.py)
    test_features = [
        [25.0, 28.0, 0.2, 0.002, 0.001, -0.005],  # Сигнал покупки (RSI низкий)
        [75.0, 72.0, 0.8, 0.002, 0.001, 0.005],   # Сигнал продажи (RSI высокий)
        [50.0, 48.0, 0.5, 0.002, 0.001, 0.001],   # Нейтральный
    ]
    
    for i, features in enumerate(test_features):
        confidence = neural_filter.predict(features)
        should_trade, conf = neural_filter.should_trade(features, threshold=0.6)
        print(f"   Тест {i+1}: confidence={confidence:.3f}, should_trade={should_trade}")
    
    # 3. Проверяем работу в стратегии
    print("\n3️⃣ Тестирование в стратегии...")
    
    # Берем небольшой файл для тестирования
    test_file = "data/BTCUSDT_2023-02-01.csv.gz"
    
    if not os.path.exists(test_file):
        print(f"❌ Тестовый файл не найден: {test_file}")
        return
    
    # Создаем стратегию с отладкой
    strategy = RSIStrategyBase(
        use_custom_rsi=True,
        use_neural_filter=True,
        neural_confidence_threshold=0.6
    )
    
    # Добавляем счетчики для отладки
    total_signals = 0
    filtered_signals = 0
    ai_decisions = []
    
    # Перехватываем метод для отладки
    original_on_tick = strategy.on_tick
    
    def debug_on_tick(price, dt, volume=0):
        nonlocal total_signals, filtered_signals, ai_decisions
        
        result = original_on_tick(price, dt, volume)
        
        # Проверяем, был ли сигнал
        if len(strategy.candles) > 20:  # Достаточно данных для анализа
            current_rsi = strategy.rsi_values[-1] if strategy.rsi_values else 50
            
            # Проверяем условия сигналов
            if current_rsi < 30 or current_rsi > 70:
                total_signals += 1
                
                # Проверяем, что сделал AI
                if strategy.use_neural_filter and strategy.neural_filter:
                    # Подготавливаем признаки как в стратегии
                    if (len(strategy.rsi_values) >= 20 and 
                        len(strategy.bb_values) >= 20 and 
                        len(strategy.atr_values) >= 20 and
                        len(strategy.volatility_ratios) >= 20 and
                        len(strategy.candles) >= 20):
                        
                        features = strategy.neural_filter.prepare_features(
                            strategy.rsi_values[-20:],
                            strategy.bb_values[-20:], 
                            strategy.atr_values[-20:],
                            strategy.volatility_ratios[-20:],
                            [c.close for c in strategy.candles[-20:]]
                        )
                        
                        if features is not None:
                            confidence = strategy.neural_filter.predict(features)
                            should_trade, conf = strategy.neural_filter.should_trade(features, threshold=0.6)
                            
                            ai_decisions.append({
                                'rsi': current_rsi,
                                'confidence': confidence,
                                'should_trade': should_trade,
                                'signal_type': 'buy' if current_rsi < 30 else 'sell'
                            })
                            
                            if not should_trade:
                                filtered_signals += 1
        
        return result
    
    strategy.on_tick = debug_on_tick
    
    # Обрабатываем файл
    print(f"   Обрабатываем файл: {os.path.basename(test_file)}")
    
    try:
        with gzip.open(test_file, 'rt') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i > 50000:  # Ограничиваем для быстрой отладки
                    break
                    
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                strategy.on_tick(price, dt, volume)
        
        strategy.on_finish(price)
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        return
    
    # 4. Анализируем результаты
    print(f"\n4️⃣ Результаты отладки:")
    print(f"   📊 Всего потенциальных сигналов: {total_signals}")
    print(f"   🚫 Отфильтровано AI: {filtered_signals}")
    print(f"   📈 Процент фильтрации: {filtered_signals/total_signals*100 if total_signals > 0 else 0:.1f}%")
    print(f"   🧠 AI решений принято: {len(ai_decisions)}")
    
    if ai_decisions:
        print(f"\n   📊 Статистика AI решений:")
        confidences = [d['confidence'] for d in ai_decisions]
        should_trade_count = sum(1 for d in ai_decisions if d['should_trade'])
        
        print(f"   💡 Средняя уверенность: {np.mean(confidences):.3f}")
        print(f"   📊 Диапазон уверенности: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        print(f"   ✅ Одобрено сигналов: {should_trade_count}/{len(ai_decisions)}")
        print(f"   🚫 Отклонено сигналов: {len(ai_decisions) - should_trade_count}/{len(ai_decisions)}")
        
        # Показываем примеры решений
        print(f"\n   🔍 Примеры решений AI:")
        for i, decision in enumerate(ai_decisions[:5]):
            print(f"   {i+1}. RSI={decision['rsi']:.1f}, confidence={decision['confidence']:.3f}, "
                  f"trade={decision['should_trade']}, type={decision['signal_type']}")

def main():
    debug_ai_filter()

if __name__ == "__main__":
    main()

