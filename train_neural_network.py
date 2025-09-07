"""
Скрипт для обучения нейронной сети на исторических данных
"""

import json
import os
import numpy as np
from neural_filter import NeuralSignalFilter
from data_generator import TrainingDataGenerator
import matplotlib.pyplot as plt

def load_training_data(filename="models/training_data.json"):
    """Загружает обучающие данные из файла"""
    
    if not os.path.exists(filename):
        print(f"❌ Файл с обучающими данными не найден: {filename}")
        print("💡 Сначала запустите: python data_generator.py")
        return None
    
    print(f"📂 Загружаем обучающие данные из {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Загружено {len(data)} примеров")
    
    # Статистика
    profitable_count = sum(1 for item in data if item['profitable'])
    print(f"📈 Прибыльных сигналов: {profitable_count} ({profitable_count/len(data)*100:.1f}%)")
    print(f"📉 Убыточных сигналов: {len(data)-profitable_count} ({(1-profitable_count/len(data))*100:.1f}%)")
    
    return data

def analyze_data_quality(training_data):
    """Анализирует качество обучающих данных"""
    
    print("\n🔍 АНАЛИЗ КАЧЕСТВА ДАННЫХ:")
    print("=" * 50)
    
    # Базовая статистика
    features_array = np.array([item['features'] for item in training_data])
    profits = [item['max_profit'] for item in training_data if item['profitable']]
    losses = [item['max_loss'] for item in training_data if not item['profitable']]
    
    print(f"📊 Размерность признаков: {features_array.shape[1]}")
    print(f"📈 Средняя прибыль успешных сигналов: {np.mean(profits)*100:.2f}%")
    print(f"📉 Средний убыток неуспешных сигналов: {np.mean(losses)*100:.2f}%")
    
    # Risk-reward анализ
    risk_rewards = [item['risk_reward'] for item in training_data if item['risk_reward'] != float('inf')]
    if risk_rewards:
        print(f"⚖️  Средний Risk/Reward: {np.mean(risk_rewards):.2f}")
    
    # Распределение по типам сигналов
    buy_signals = sum(1 for item in training_data if item['signal_type'] == 'buy')
    sell_signals = sum(1 for item in training_data if item['signal_type'] == 'sell')
    print(f"📊 Сигналов покупки: {buy_signals}")
    print(f"📊 Сигналов продажи: {sell_signals}")
    
    # Проверка на выбросы в признаках
    for i in range(features_array.shape[1]):
        feature_values = features_array[:, i]
        q1, q3 = np.percentile(feature_values, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((feature_values < q1 - 1.5*iqr) | (feature_values > q3 + 1.5*iqr))
        if outliers > len(feature_values) * 0.05:  # Больше 5% выбросов
            print(f"⚠️  Признак {i}: {outliers} выбросов ({outliers/len(feature_values)*100:.1f}%)")

def train_model(training_data, validation_split=0.2, epochs=100):
    """Обучает нейронную сеть"""
    
    print("\n🧠 ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ:")
    print("=" * 50)
    
    # Создаем и обучаем модель
    neural_filter = NeuralSignalFilter()
    
    history = neural_filter.train(
        training_data=training_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=64
    )
    
    return neural_filter, history

def plot_training_history(history):
    """Визуализирует процесс обучения"""
    
    plt.figure(figsize=(15, 5))
    
    # График точности
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # График функции потерь
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График precision
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 График обучения сохранен: models/training_history.png")

def test_model(neural_filter, training_data, n_samples=100):
    """Тестирует обученную модель на случайных примерах"""
    
    print("\n🧪 ТЕСТИРОВАНИЕ МОДЕЛИ:")
    print("=" * 50)
    
    # Берем случайные примеры
    test_indices = np.random.choice(len(training_data), min(n_samples, len(training_data)), replace=False)
    
    correct_predictions = 0
    high_confidence_correct = 0
    high_confidence_total = 0
    
    for idx in test_indices:
        sample = training_data[idx]
        features = np.array(sample['features'])
        actual_profitable = sample['profitable']
        
        # Получаем предсказание
        should_trade, confidence = neural_filter.should_trade(features, threshold=0.5)
        
        # Проверяем правильность
        if should_trade == actual_profitable:
            correct_predictions += 1
            
        # Анализируем высокоуверенные предсказания
        if confidence > 0.7 or confidence < 0.3:
            high_confidence_total += 1
            if should_trade == actual_profitable:
                high_confidence_correct += 1
    
    accuracy = correct_predictions / len(test_indices)
    high_conf_accuracy = high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0
    
    print(f"✅ Общая точность: {accuracy:.3f} ({correct_predictions}/{len(test_indices)})")
    print(f"🎯 Точность высокоуверенных предсказаний: {high_conf_accuracy:.3f} ({high_confidence_correct}/{high_confidence_total})")
    print(f"📊 Доля высокоуверенных предсказаний: {high_confidence_total/len(test_indices)*100:.1f}%")

def main():
    """Основная функция обучения"""
    
    print("🚀 ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ ДЛЯ ФИЛЬТРАЦИИ СИГНАЛОВ")
    print("=" * 60)
    
    # 1. Проверяем наличие обучающих данных
    training_data = load_training_data()
    if not training_data:
        print("\n🔄 Генерируем обучающие данные...")
        generator = TrainingDataGenerator()
        training_data = generator.generate_training_data(
            data_pattern="data/BTCUSDT_2023-*.csv.gz",
            max_files=30,  # Для тестирования
            output_file="models/training_data.json"
        )
        
        if not training_data:
            print("❌ Не удалось сгенерировать обучающие данные!")
            return
    
    # 2. Анализируем качество данных
    analyze_data_quality(training_data)
    
    # 3. Обучаем модель
    neural_filter, history = train_model(training_data, epochs=50)
    
    # 4. Визуализируем процесс обучения
    plot_training_history(history)
    
    # 5. Тестируем модель
    test_model(neural_filter, training_data)
    
    # 6. Анализируем важность признаков
    print("\n🔍 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ:")
    print("=" * 50)
    
    feature_importance = neural_filter.get_feature_importance(training_data)
    if feature_importance:
        print("📊 Топ-10 самых важных признаков:")
        for i, (feature_name, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature_name:<20}: {importance:.4f}")
    
    print(f"\n🎉 Обучение завершено!")
    print(f"💾 Модель сохранена: models/neural_filter.keras")
    print(f"💾 Скалер сохранен: models/scaler.joblib")
    print(f"\n🚀 Теперь можно использовать нейронную сеть в торговой стратегии!")

if __name__ == "__main__":
    # Создаем директорию для моделей
    os.makedirs("models", exist_ok=True)
    main()

