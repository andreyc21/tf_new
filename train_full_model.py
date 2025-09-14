#!/usr/bin/env python3
"""
🧠 Полноценное обучение нейронной модели на всем периоде данных
"""

import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import json

# Импортируем наши модули
from rsi_strategy import RSIStrategyBase
from support_resistance import SupportResistanceFinder

class FullModelTrainer:
    """🧠 Тренер модели на полном датасете"""
    
    def __init__(self, data_directory="data", model_directory="models", 
                 lookback_window=50, prediction_horizon=5):
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Создаем директории
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Настройки модели
        self.model_config = {
            'lookback_window': lookback_window,
            'prediction_horizon': prediction_horizon,
            'features': [
                'rsi', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'atr', 'volatility_ratio', 'price_change_1', 'price_change_5',
                'volume_ratio', 'time_of_day', 'day_of_week',
                'nearest_support', 'nearest_resistance', 'support_strength', 
                'resistance_strength', 'support_distance', 'resistance_distance'
            ],
            'model_type': 'LSTM',
            'architecture': {
                'lstm_units': [128, 64, 32],
                'dropout': 0.3,
                'dense_units': [128, 64, 32],  # Увеличиваем размер для большего количества фичей
                'activation': 'relu',
                'output_activation': 'sigmoid'
            },
            'training': {
                'batch_size': 256,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 15,
                'reduce_lr_patience': 8,
                'min_lr': 1e-7
            }
        }
    
    def find_all_data_files(self):
        """📁 Находим все файлы данных"""
        print("🔍 Поиск файлов данных...")
        
        patterns = [
            os.path.join(self.data_directory, "*.csv"),
            os.path.join(self.data_directory, "*.csv.gz"),
            os.path.join(self.data_directory, "BTCUSDT_*.csv"),
            os.path.join(self.data_directory, "BTCUSDT_*.csv.gz")
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Убираем дубликаты и сортируем
        all_files = sorted(list(set(all_files)))
        
        print(f"📊 Найдено файлов: {len(all_files)}")
        for file in all_files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  📁 {os.path.basename(file)} ({size_mb:.1f} MB)")
        
        return all_files
    
    def extract_features_from_file(self, filename):
        """📊 Извлекаем фичи из одного файла"""
        print(f"📊 Обрабатываем: {os.path.basename(filename)}")
        
        try:
            # Загружаем данные через backtester
            from backtester import run_backtest_on_file
            
            # Запускаем минимальный бэктест чтобы получить обработанные данные
            strategy = run_backtest_on_file(filename, plot=False, verbose=False)
            
            if len(strategy.candles) < self.lookback_window + self.prediction_horizon + 50:
                print(f"⚠️ Недостаточно свечей в {filename}")
                return None
            
            print(f"  🕯️ Свечей: {len(strategy.candles)}")
            
            # Находим S&R уровни для всего файла
            print(f"  📊 Поиск S&R уровней...")
            sr_finder = SupportResistanceFinder(
                min_touches=2, 
                tolerance_pct=0.3,
                lookback=min(200, len(strategy.candles)),
                min_strength=0.1
            )
            sr_levels = sr_finder.find_support_resistance_levels(strategy.candles)
            print(f"  📊 Найдено S&R уровней: {len(sr_levels)}")
            
            # Извлекаем фичи
            features_data = []
            
            for i in range(self.lookback_window, len(strategy.candles) - self.prediction_horizon):
                # Текущие индикаторы
                rsi = strategy.rsi_values[i] if i < len(strategy.rsi_values) else 50.0
                bb_ma, bb_upper, bb_lower = strategy.bb_values[i] if i < len(strategy.bb_values) else (0, 0, 0)
                atr = strategy.atr_values[i] if i < len(strategy.atr_values) else 0.0
                volatility_ratio = strategy.volatility_ratios[i] if i < len(strategy.volatility_ratios) else 1.0
                
                candle = strategy.candles[i]
                
                # Дополнительные фичи
                bb_width = (bb_upper - bb_lower) / bb_ma if bb_ma > 0 else 0
                bb_position = (candle.close - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
                
                # Изменения цены
                price_change_1 = (candle.close - strategy.candles[i-1].close) / strategy.candles[i-1].close if i > 0 else 0
                price_change_5 = (candle.close - strategy.candles[max(0, i-5)].close) / strategy.candles[max(0, i-5)].close if i >= 5 else 0
                
                # Объем
                avg_volume = np.mean([c.volume for c in strategy.candles[max(0, i-20):i+1]])
                volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1.0
                
                # Временные фичи
                time_of_day = candle.start_time.hour + candle.start_time.minute / 60.0
                day_of_week = candle.start_time.weekday()
                
                # S&R фичи
                current_price = candle.close
                nearest_levels = sr_finder.get_nearest_levels(sr_levels, current_price, max_distance_pct=5.0)
                
                # Ближайшая поддержка
                if nearest_levels['support']:
                    nearest_support = nearest_levels['support'].price
                    support_strength = nearest_levels['support'].strength
                    support_distance = (current_price - nearest_support) / current_price  # Положительное если цена выше
                else:
                    nearest_support = current_price * 0.95  # Условная поддержка -5%
                    support_strength = 0.0
                    support_distance = 0.05
                
                # Ближайшее сопротивление
                if nearest_levels['resistance']:
                    nearest_resistance = nearest_levels['resistance'].price
                    resistance_strength = nearest_levels['resistance'].strength
                    resistance_distance = (nearest_resistance - current_price) / current_price  # Положительное если сопротивление выше
                else:
                    nearest_resistance = current_price * 1.05  # Условное сопротивление +5%
                    resistance_strength = 0.0
                    resistance_distance = 0.05
                
                # Целевая переменная - будет ли прибыльный сигнал через N свечей
                future_prices = [strategy.candles[j].close for j in range(i+1, min(i+1+self.prediction_horizon, len(strategy.candles)))]
                if future_prices:
                    max_future_price = max(future_prices)
                    min_future_price = min(future_prices)
                    
                    # Прибыльный сигнал если цена изменится больше чем на 0.1% в любую сторону
                    price_change_threshold = 0.001  # 0.1%
                    max_gain = (max_future_price - candle.close) / candle.close
                    max_loss = (candle.close - min_future_price) / candle.close
                    
                    target = 1.0 if (max_gain > price_change_threshold or max_loss > price_change_threshold) else 0.0
                else:
                    target = 0.0
                
                # Собираем фичи
                feature_row = [
                    rsi, bb_upper, bb_lower, bb_width, bb_position,
                    atr, volatility_ratio, price_change_1, price_change_5,
                    volume_ratio, time_of_day, day_of_week,
                    nearest_support, nearest_resistance, support_strength, 
                    resistance_strength, support_distance, resistance_distance,
                    target
                ]
                
                features_data.append(feature_row)
            
            if features_data:
                columns = self.model_config['features'] + ['target']
                df = pd.DataFrame(features_data, columns=columns)
                print(f"  ✅ Извлечено фич: {len(df)}")
                return df
            
        except Exception as e:
            print(f"❌ Ошибка в {filename}: {e}")
        
        return None
    
    def collect_all_features(self):
        """📊 Собираем фичи со всех файлов"""
        print("📊 СБОР ФИЧЕЙ СО ВСЕХ ФАЙЛОВ")
        print("=" * 50)
        
        files = self.find_all_data_files()
        if not files:
            raise ValueError("❌ Не найдено файлов данных")
        
        all_features = []
        processed_files = 0
        
        for filename in files:
            features_df = self.extract_features_from_file(filename)
            if features_df is not None:
                all_features.append(features_df)
                processed_files += 1
                
                # Промежуточная статистика
                if processed_files % 5 == 0:
                    total_rows = sum(len(df) for df in all_features)
                    print(f"📊 Обработано файлов: {processed_files}/{len(files)}, строк: {total_rows:,}")
        
        if not all_features:
            raise ValueError("❌ Не удалось извлечь фичи ни из одного файла")
        
        # Объединяем все данные
        print("🔄 Объединение данных...")
        combined_df = pd.concat(all_features, ignore_index=True)
        
        print(f"✅ ИТОГО:")
        print(f"  📁 Обработано файлов: {processed_files}/{len(files)}")
        print(f"  📊 Всего строк: {len(combined_df):,}")
        print(f"  📈 Положительных примеров: {combined_df['target'].sum():,.0f} ({combined_df['target'].mean()*100:.1f}%)")
        
        return combined_df
    
    def prepare_training_data(self, df):
        """🎯 Подготавливаем данные для обучения"""
        print("🎯 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        print("=" * 40)
        
        # Удаляем NaN и бесконечности
        print(f"📊 Строк до очистки: {len(df):,}")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"📊 Строк после очистки: {len(df):,}")
        
        # Разделяем фичи и цели
        feature_columns = self.model_config['features']
        X = df[feature_columns].values
        y = df['target'].values
        
        print(f"📊 Размерность X: {X.shape}")
        print(f"📊 Размерность y: {y.shape}")
        print(f"📊 Баланс классов: {y.mean():.3f} (1.0 = {y.sum():,.0f}, 0.0 = {(len(y)-y.sum()):,.0f})")
        
        # Нормализация фичей
        print("🔄 Нормализация фичей...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Сохраняем скейлер
        scaler_path = os.path.join(self.model_directory, 'feature_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"💾 Скейлер сохранен: {scaler_path}")
        
        # Разделение на train/validation/test
        print("🔄 Разделение данных...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Train: {X_train.shape[0]:,} ({y_train.mean():.3f})")
        print(f"📊 Validation: {X_val.shape[0]:,} ({y_val.mean():.3f})")
        print(f"📊 Test: {X_test.shape[0]:,} ({y_test.mean():.3f})")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
    
    def build_model(self, input_shape):
        """🏗️ Создаем архитектуру модели"""
        print("🏗️ СОЗДАНИЕ МОДЕЛИ")
        print("=" * 30)
        
        config = self.model_config['architecture']
        
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            
            # Dense layers для обработки фичей
            keras.layers.Dense(config['dense_units'][0], activation=config['activation']),
            keras.layers.Dropout(config['dropout']),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(config['dense_units'][1], activation=config['activation']),
            keras.layers.Dropout(config['dropout']),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(config['dense_units'][2], activation=config['activation']),
            keras.layers.Dropout(config['dropout']/2),
            
            # Выходной слой
            keras.layers.Dense(1, activation=config['output_activation'])
        ])
        
        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("📋 Архитектура модели:")
        model.summary()
        
        return model
    
    def train_model(self, model, train_data, val_data):
        """🎯 Обучаем модель"""
        print("🎯 ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 30)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        config = self.model_config['training']
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config['reduce_lr_patience'],
                min_lr=config['min_lr'],
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_directory, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Обучение
        print(f"🚀 Начинаем обучение на {config['epochs']} эпох...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_data):
        """📊 Оцениваем модель"""
        print("📊 ОЦЕНКА МОДЕЛИ")
        print("=" * 25)
        
        X_test, y_test = test_data
        
        # Предсказания
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Метрики
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"📊 Accuracy:  {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall:    {recall:.4f}")
        print(f"📊 F1 Score:  {f1:.4f}")
        print(f"📊 AUC:       {auc:.4f}")
        
        # Сохраняем метрики
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'test_samples': int(len(y_test))
        }
        
        return metrics
    
    def save_model_artifacts(self, model, metrics, scaler):
        """💾 Сохраняем все артефакты модели"""
        print("💾 СОХРАНЕНИЕ АРТЕФАКТОВ")
        print("=" * 35)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Основная модель
        model_path = os.path.join(self.model_directory, f'neural_filter_full_{timestamp}.keras')
        model.save(model_path)
        print(f"💾 Модель: {model_path}")
        
        # Конфигурация и метрики
        config_path = os.path.join(self.model_directory, f'model_config_{timestamp}.json')
        full_config = {
            'model_config': self.model_config,
            'training_metrics': metrics,
            'timestamp': timestamp,
            'tensorflow_version': tf.__version__,
            'model_path': model_path
        }
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        print(f"💾 Конфигурация: {config_path}")
        
        # Симлинки на последнюю модель
        latest_model = os.path.join(self.model_directory, 'neural_filter_latest.keras')
        latest_config = os.path.join(self.model_directory, 'model_config_latest.json')
        
        # Удаляем старые симлинки
        for path in [latest_model, latest_config]:
            if os.path.exists(path):
                os.remove(path)
        
        # Создаем новые
        os.symlink(os.path.basename(model_path), latest_model)
        os.symlink(os.path.basename(config_path), latest_config)
        
        print(f"🔗 Симлинки обновлены")
        
        return model_path, config_path
    
    def run_full_training(self):
        """🚀 Полный цикл обучения"""
        print("🧠 ПОЛНОЕ ОБУЧЕНИЕ НЕЙРОННОЙ МОДЕЛИ")
        print("=" * 60)
        print(f"🕐 Начало: {datetime.now()}")
        
        try:
            # 1. Сбор данных
            df = self.collect_all_features()
            
            # 2. Подготовка данных
            train_data, val_data, test_data, scaler = self.prepare_training_data(df)
            
            # 3. Создание модели
            input_shape = (train_data[0].shape[1],)
            model = self.build_model(input_shape)
            
            # 4. Обучение
            history = self.train_model(model, train_data, val_data)
            
            # 5. Оценка
            metrics = self.evaluate_model(model, test_data)
            
            # 6. Сохранение
            model_path, config_path = self.save_model_artifacts(model, metrics, scaler)
            
            print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"🕐 Время: {datetime.now()}")
            print(f"📊 Финальные метрики:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
            
            print(f"\n📁 Артефакты:")
            print(f"   🧠 Модель: {model_path}")
            print(f"   ⚙️ Конфигурация: {config_path}")
            print(f"   📊 Скейлер: models/feature_scaler.pkl")
            
            return model, metrics, history
            
        except Exception as e:
            print(f"❌ ОШИБКА ОБУЧЕНИЯ: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    # Создаем тренер
    trainer = FullModelTrainer(
        data_directory="data",
        model_directory="models",
        lookback_window=50,
        prediction_horizon=5
    )
    
    # Запускаем обучение
    model, metrics, history = trainer.run_full_training()
    
    print("\n✅ Готово! Модель обучена на всем доступном периоде.")
