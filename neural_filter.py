"""
Нейронная сеть для фильтрации торговых сигналов
Помогает избежать проигрышных входов на основе исторических данных
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class NeuralSignalFilter:
    def __init__(self, model_path="models/neural_filter.keras", scaler_path="models/scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
        # Создаем директорию для моделей
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Пытаемся загрузить существующую модель
        self.load_model()
    
    def prepare_features(self, rsi_values, bb_values, atr_values, volatility_ratios, 
                        prices, volumes=None, lookback=20):
        """Подготавливает признаки для нейронной сети"""
        
        # Проверяем минимальные требования к данным
        min_length = min(len(rsi_values), len(bb_values), len(atr_values), 
                        len(volatility_ratios), len(prices))
        
        if min_length < 10:  # Снижаем требования
            return None
            
        features = []
        
        # Используем доступные данные (адаптивно)
        actual_lookback = min(lookback, min_length)
        recent_rsi = rsi_values[-actual_lookback:]
        recent_bb = bb_values[-actual_lookback:]
        recent_atr = atr_values[-actual_lookback:]
        recent_vol_ratio = volatility_ratios[-actual_lookback:]
        recent_prices = prices[-actual_lookback:]
        
        # 1. RSI статистики
        features.extend([
            np.mean(recent_rsi),           # Средний RSI
            np.std(recent_rsi),            # Волатильность RSI
            recent_rsi[-1],                # Текущий RSI
            np.min(recent_rsi),            # Минимальный RSI
            np.max(recent_rsi),            # Максимальный RSI
        ])
        
        # 2. Bollinger Bands статистики
        bb_positions = []  # Позиция цены относительно BB
        for i, (ma, upper, lower) in enumerate(recent_bb):
            if ma and upper and lower:
                price = recent_prices[i]
                bb_pos = (price - lower) / (upper - lower) if upper != lower else 0.5
                bb_positions.append(bb_pos)
            else:
                bb_positions.append(0.5)
        
        features.extend([
            np.mean(bb_positions),         # Средняя позиция в BB
            np.std(bb_positions),          # Волатильность позиции в BB
            bb_positions[-1],              # Текущая позиция в BB
        ])
        
        # 3. Волатильность (ATR)
        features.extend([
            np.mean(recent_atr),           # Средняя ATR
            np.std(recent_atr),            # Волатильность ATR
            recent_atr[-1],                # Текущая ATR
            np.mean(recent_vol_ratio),     # Средний коэффициент волатильности
            recent_vol_ratio[-1],          # Текущий коэффициент волатильности
        ])
        
        # 4. Ценовая динамика
        price_changes = np.diff(recent_prices)
        features.extend([
            np.mean(price_changes),        # Средняя ценовая динамика
            np.std(price_changes),         # Волатильность цен
            price_changes[-1],             # Последнее изменение цены
            (recent_prices[-1] - recent_prices[0]) / recent_prices[0],  # Общее изменение за период
        ])
        
        # 5. Трендовые индикаторы (адаптивно)
        sma_short_len = min(5, len(recent_prices))
        sma_long_len = min(10, len(recent_prices))
        
        sma_short = np.mean(recent_prices[-sma_short_len:]) if sma_short_len > 0 else recent_prices[-1]
        sma_long = np.mean(recent_prices[-sma_long_len:]) if sma_long_len > 0 else recent_prices[-1]
        
        features.extend([
            (sma_short - sma_long) / sma_long if sma_long != 0 else 0,     # Соотношение SMA
            (recent_prices[-1] - sma_short) / sma_short if sma_short != 0 else 0,  # Цена vs SMA
        ])
        
        return np.array(features)
    
    def create_model(self, input_dim):
        """Создает архитектуру нейронной сети"""
        model = Sequential([
            # Входной слой с нормализацией
            Dense(128, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Скрытые слои
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # Выходной слой (бинарная классификация: хороший/плохой сигнал)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, training_data, validation_split=0.2, epochs=100, batch_size=32):
        """Обучает нейронную сеть на исторических данных"""
        
        print("🧠 Начинаем обучение нейронной сети...")
        
        # Подготавливаем данные
        X = np.array([item['features'] for item in training_data])
        y = np.array([item['profitable'] for item in training_data])
        
        print(f"📊 Размер dataset: {len(X)} примеров")
        print(f"📈 Прибыльных сигналов: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"📉 Убыточных сигналов: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
        
        # Нормализация признаков
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Создаем модель
        self.model = self.create_model(X_scaled.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_loss')
        ]
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка модели
        y_pred = (self.model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n📊 Результаты обучения:")
        print(f"✅ Точность на валидации: {accuracy:.3f}")
        print("\n📋 Детальный отчет:")
        print(classification_report(y_val, y_pred, target_names=['Плохой сигнал', 'Хороший сигнал']))
        
        # Сохраняем модель и скалер
        self.save_model()
        
        return history
    
    def predict(self, features):
        """Предсказывает качество сигнала (0-1, где 1 = хороший сигнал)"""
        if self.model is None or self.scaler is None:
            return 0.5  # Нейтральная оценка если модель не загружена
        
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            return float(prediction)
        except Exception as e:
            print(f"⚠️ Ошибка предсказания: {e}")
            return 0.5
    
    def should_trade(self, features, threshold=0.6):
        """Определяет, стоит ли торговать по данному сигналу"""
        confidence = self.predict(features)
        return confidence >= threshold, confidence
    
    def save_model(self):
        """Сохраняет обученную модель и скалер"""
        if self.model:
            self.model.save(self.model_path)
            print(f"💾 Модель сохранена: {self.model_path}")
        
        if self.scaler:
            joblib.dump(self.scaler, self.scaler_path)
            print(f"💾 Скалер сохранен: {self.scaler_path}")
    
    def load_model(self):
        """Загружает сохраненную модель и скалер"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"✅ Модель загружена: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"✅ Скалер загружен: {self.scaler_path}")
                
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            self.model = None
            self.scaler = None
    
    def get_feature_importance(self, training_data, n_samples=1000):
        """Анализирует важность признаков (упрощенный подход)"""
        if self.model is None:
            return None
        
        # Берем случайную выборку для анализа
        sample_data = np.random.choice(training_data, min(n_samples, len(training_data)), replace=False)
        
        X = np.array([item['features'] for item in sample_data])
        X_scaled = self.scaler.transform(X)
        
        # Вычисляем базовое предсказание
        base_pred = np.mean(self.model.predict(X_scaled, verbose=0))
        
        # Анализируем важность каждого признака
        feature_names = [
            'RSI_mean', 'RSI_std', 'RSI_current', 'RSI_min', 'RSI_max',
            'BB_pos_mean', 'BB_pos_std', 'BB_pos_current',
            'ATR_mean', 'ATR_std', 'ATR_current', 'Vol_ratio_mean', 'Vol_ratio_current',
            'Price_change_mean', 'Price_change_std', 'Price_change_last', 'Price_total_change',
            'SMA_ratio', 'Price_vs_SMA'
        ]
        
        importances = []
        for i in range(X_scaled.shape[1]):
            # Обнуляем признак
            X_modified = X_scaled.copy()
            X_modified[:, i] = 0
            
            # Вычисляем изменение предсказания
            modified_pred = np.mean(self.model.predict(X_modified, verbose=0))
            importance = abs(base_pred - modified_pred)
            importances.append(importance)
        
        # Сортируем по важности
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance

