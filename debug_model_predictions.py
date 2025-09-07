"""
Диагностика AI-модели 2025 - почему всегда уверенность 0.500?
"""

import numpy as np
import joblib
from tensorflow import keras
from neural_filter import NeuralSignalFilter
import json

def debug_model_predictions():
    """Диагностика предсказаний модели"""
    
    print("🔍 ДИАГНОСТИКА AI-МОДЕЛИ 2025")
    print("=" * 50)
    
    # 1. Загружаем модель и скалер
    try:
        model = keras.models.load_model("models/neural_filter_2025.keras")
        scaler = joblib.load("models/scaler_2025.joblib")
        print("✅ Модель и скалер загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # 2. Проверяем архитектуру модели
    print(f"\n🏗️ АРХИТЕКТУРА МОДЕЛИ:")
    model.summary()
    
    # 3. Проверяем веса модели
    print(f"\n⚖️ ПРОВЕРКА ВЕСОВ:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()[0]  # Веса (без bias)
            print(f"Слой {i} ({layer.name}): shape={weights.shape}, mean={np.mean(weights):.6f}, std={np.std(weights):.6f}")
            
            # Проверяем на вырожденность
            if np.std(weights) < 1e-6:
                print(f"   ⚠️ ПРОБЛЕМА: Веса почти нулевые!")
    
    # 4. Загружаем тестовые данные
    print(f"\n📊 ТЕСТИРОВАНИЕ НА ОБУЧАЮЩИХ ДАННЫХ:")
    try:
        with open("models/training_data_2025.json", "r") as f:
            training_data = json.load(f)
        
        # Проверяем структуру данных
        print(f"Структура данных: {type(training_data)}")
        if isinstance(training_data, list):
            print(f"Список из {len(training_data)} элементов")
            print(f"Первый элемент: {training_data[0] if training_data else 'Пусто'}")
            
            # Извлекаем features и labels из списка словарей
            features_list = [item['features'] for item in training_data]
            labels_list = [1 if item['profitable'] else 0 for item in training_data]
        else:
            features_list = training_data["features"]
            labels_list = training_data["labels"]
        
        print(f"✅ Загружено {len(features_list)} примеров")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # 5. Тестируем первые 20 примеров
    print(f"\n🧪 ТЕСТ ПРЕДСКАЗАНИЙ (первые 20 примеров):")
    
    X_test = np.array(features_list[:20])
    y_test = np.array(labels_list[:20])
    
    # Нормализуем данные
    X_test_scaled = scaler.transform(X_test)
    
    # Получаем предсказания
    predictions = model.predict(X_test_scaled, verbose=0)
    
    print(f"📊 Статистика предсказаний:")
    print(f"   Минимальная: {np.min(predictions):.6f}")
    print(f"   Максимальная: {np.max(predictions):.6f}")
    print(f"   Средняя: {np.mean(predictions):.6f}")
    print(f"   Стандартное отклонение: {np.std(predictions):.6f}")
    
    if np.std(predictions) < 1e-6:
        print(f"   🚨 ПРОБЛЕМА: Все предсказания одинаковые!")
    
    # 6. Показываем детали
    print(f"\n📋 ДЕТАЛИ ПРЕДСКАЗАНИЙ:")
    for i in range(min(10, len(predictions))):
        pred_class = "Хороший" if predictions[i][0] > 0.5 else "Плохой"
        true_class = "Хороший" if y_test[i] == 1 else "Плохой"
        print(f"   {i+1}: pred={predictions[i][0]:.6f} ({pred_class}), true={true_class}")
    
    # 7. Проверяем скалер
    print(f"\n📏 ПРОВЕРКА СКАЛЕРА:")
    print(f"   Mean: {scaler.mean_}")
    print(f"   Scale: {scaler.scale_}")
    
    # Проверяем вырожденность скалера
    if np.any(scaler.scale_ < 1e-6):
        print(f"   🚨 ПРОБЛЕМА: Некоторые признаки имеют нулевую дисперсию!")
        zero_var_features = np.where(scaler.scale_ < 1e-6)[0]
        print(f"   Признаки с нулевой дисперсией: {zero_var_features}")
    
    # 8. Тестируем с искусственными данными
    print(f"\n🧪 ТЕСТ С ИСКУССТВЕННЫМИ ДАННЫМИ:")
    
    # Создаем разнообразные тестовые данные
    test_features = [
        [20, 25, 0.1, 0.002, 0.001, 0.01],    # Низкий RSI
        [80, 75, 0.9, 0.005, 0.003, -0.02],   # Высокий RSI
        [50, 50, 0.5, 0.003, 0.002, 0.0],     # Нейтральный
        [30, 35, 0.2, 0.001, 0.0005, 0.015],  # Граничный низкий
        [70, 65, 0.8, 0.004, 0.0025, -0.01],  # Граничный высокий
    ]
    
    X_artificial = np.array(test_features)
    X_artificial_scaled = scaler.transform(X_artificial)
    artificial_preds = model.predict(X_artificial_scaled, verbose=0)
    
    print(f"   Искусственные предсказания:")
    for i, pred in enumerate(artificial_preds):
        print(f"   {i+1}: {pred[0]:.6f}")
    
    if np.std(artificial_preds) < 1e-6:
        print(f"   🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА: Модель не реагирует на разные входы!")
        diagnose_model_issues(model, scaler, X_test_scaled, y_test)

def diagnose_model_issues(model, scaler, X_test, y_test):
    """Диагностирует конкретные проблемы модели"""
    
    print(f"\n🔧 ДИАГНОСТИКА ПРОБЛЕМ:")
    print("=" * 40)
    
    # 1. Проверяем активации слоев
    print(f"1. Проверка активаций слоев:")
    
    # Создаем модели для получения промежуточных активаций
    layer_outputs = [layer.output for layer in model.layers[1:]]  # Пропускаем input
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Получаем активации для одного примера
    sample_input = X_test[:1]
    activations = activation_model.predict(sample_input, verbose=0)
    
    for i, activation in enumerate(activations):
        layer_name = model.layers[i+1].name
        print(f"   Слой {i+1} ({layer_name}): mean={np.mean(activation):.6f}, std={np.std(activation):.6f}")
        
        # Проверяем на насыщение
        if hasattr(model.layers[i+1], 'activation'):
            if np.std(activation) < 1e-6:
                print(f"     ⚠️ Слой насыщен (нет вариативности)")
    
    # 2. Проверяем градиенты
    print(f"\n2. Проверка обучаемости:")
    
    # Компилируем модель заново для проверки
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Проверяем loss на тестовых данных
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test loss: {test_loss[0]:.6f}")
    print(f"   Test accuracy: {test_loss[1]:.6f}")
    
    # 3. Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if np.std(activations[-1]) < 1e-6:  # Последний слой
        print(f"   🔄 Переобучить модель с другими параметрами:")
        print(f"      - Увеличить learning rate")
        print(f"      - Изменить архитектуру")
        print(f"      - Проверить качество данных")
    
    # 4. Проверяем данные обучения
    print(f"\n3. Проверка качества обучающих данных:")
    
    try:
        with open("models/training_data_2025.json", "r") as f:
            training_data = json.load(f)
        
        features = np.array(training_data["features"])
        labels = np.array(training_data["labels"])
        
        print(f"   Размер данных: {features.shape}")
        print(f"   Баланс классов: {np.mean(labels):.3f} (должно быть ~0.5)")
        
        # Проверяем вариативность признаков
        print(f"   Вариативность признаков:")
        for i in range(features.shape[1]):
            feature_std = np.std(features[:, i])
            print(f"     Признак {i}: std={feature_std:.6f}")
            if feature_std < 1e-6:
                print(f"       ⚠️ Признак {i} константен!")
        
    except Exception as e:
        print(f"   ❌ Ошибка проверки данных: {e}")

def main():
    """Основная функция"""
    debug_model_predictions()

if __name__ == "__main__":
    main()
