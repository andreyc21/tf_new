"""
Быстрое обучение нейронной сети на сгенерированных данных
"""

import json
import os
import numpy as np
from neural_filter import NeuralSignalFilter
import matplotlib.pyplot as plt

def load_fast_training_data(filename="models/training_data_2025.json"):
    """Загружает быстро сгенерированные обучающие данные"""
    
    if not os.path.exists(filename):
        print(f"❌ Файл не найден: {filename}")
        return None
    
    print(f"📂 Загружаем данные из {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Загружено {len(data)} примеров")
    
    # Статистика
    profitable_count = sum(1 for item in data if item['profitable'])
    print(f"📈 Прибыльных сигналов: {profitable_count} ({profitable_count/len(data)*100:.1f}%)")
    print(f"📉 Убыточных сигналов: {len(data)-profitable_count} ({(1-profitable_count/len(data))*100:.1f}%)")
    
    return data

def analyze_fast_data_quality(training_data):
    """Анализирует качество быстро сгенерированных данных"""
    
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
    
    # Анализ признаков
    feature_names = ['RSI', 'RSI_mean_5', 'BB_position', 'ATR_norm', 'Volatility', 'Price_change_5']
    print(f"\n📊 Статистика признаков:")
    for i, name in enumerate(feature_names):
        values = features_array[:, i]
        print(f"  {name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

def train_fast_model(training_data, validation_split=0.2, epochs=50):
    """Быстро обучает нейронную сеть"""
    
    print("\n🧠 БЫСТРОЕ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ:")
    print("=" * 50)
    
    # Создаем и обучаем модель на свежих данных
    neural_filter = NeuralSignalFilter(
        model_path="models/neural_filter_2025.keras",  # Новая модель
        scaler_path="models/scaler_2025.joblib"        # Новый скалер
    )
    
    history = neural_filter.train(
        training_data=training_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=32  # Меньший батч для быстрого обучения
    )
    
    return neural_filter, history

def plot_fast_training_history(history):
    """Визуализирует процесс обучения"""
    
    plt.figure(figsize=(15, 5))
    
    # График точности
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График функции потерь
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График precision
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision', color='blue')
    plt.plot(history.history['val_precision'], label='Validation Precision', color='red')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history_fast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 График обучения сохранен: models/training_history_fast.png")

def test_fast_model(neural_filter, training_data, n_samples=100):
    """Тестирует обученную модель"""
    
    print("\n🧪 ТЕСТИРОВАНИЕ МОДЕЛИ:")
    print("=" * 50)
    
    # Берем случайные примеры
    test_indices = np.random.choice(len(training_data), min(n_samples, len(training_data)), replace=False)
    
    correct_predictions = 0
    high_confidence_correct = 0
    high_confidence_total = 0
    
    confidence_scores = []
    actual_labels = []
    
    for idx in test_indices:
        sample = training_data[idx]
        features = np.array(sample['features'])
        actual_profitable = sample['profitable']
        
        # Получаем предсказание
        confidence = neural_filter.predict(features)
        should_trade = confidence >= 0.5
        
        confidence_scores.append(confidence)
        actual_labels.append(actual_profitable)
        
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
    
    # Анализ распределения уверенности
    confidence_scores = np.array(confidence_scores)
    print(f"📈 Средняя уверенность: {np.mean(confidence_scores):.3f}")
    print(f"📊 Стандартное отклонение: {np.std(confidence_scores):.3f}")

def main():
    """Основная функция быстрого обучения"""
    
    print("🚀 БЫСТРОЕ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("=" * 60)
    
    # 1. Загружаем данные
    training_data = load_fast_training_data()
    if not training_data:
        print("❌ Нет данных для обучения!")
        return
    
    # 2. Анализируем качество данных
    analyze_fast_data_quality(training_data)
    
    # 3. Обучаем модель
    neural_filter, history = train_fast_model(training_data, epochs=30)  # Меньше эпох для скорости
    
    # 4. Визуализируем процесс обучения
    plot_fast_training_history(history)
    
    # 5. Тестируем модель
    test_fast_model(neural_filter, training_data)
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"💾 Модель сохранена: models/neural_filter_fast.keras")
    print(f"💾 Скалер сохранен: models/scaler_fast.joblib")
    print(f"\n🚀 Теперь можно использовать AI-фильтр в торговой стратегии!")
    print(f"💡 Для активации в config.py установите:")
    print(f"   USE_NEURAL_FILTER = True")
    print(f"   NEURAL_CONFIDENCE_THRESHOLD = 0.6")

if __name__ == "__main__":
    # Создаем директорию для моделей
    os.makedirs("models", exist_ok=True)
    main()
