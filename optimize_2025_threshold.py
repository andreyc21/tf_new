"""
Оптимизация порога уверенности для новой AI-модели 2025
"""

import gzip
import csv
import numpy as np
from datetime import datetime, timezone
from rsi_strategy import RSIStrategyBase
import os

def timestamp_to_dt(ts):
    """Конвертирует timestamp в datetime"""
    return datetime.fromtimestamp(int(ts) / 1000, timezone.utc)

def test_threshold_optimization():
    """Тестирует разные пороги уверенности для новой модели"""
    
    print("🎯 ОПТИМИЗАЦИЯ ПОРОГА УВЕРЕННОСТИ AI-МОДЕЛИ 2025")
    print("=" * 70)
    
    # Тестируем на одном файле для скорости
    test_file = "data/BTCUSDT_2025-02-01.csv.gz"
    
    if not os.path.exists(test_file):
        print(f"❌ Файл не найден: {test_file}")
        return
    
    # Тестируем разные пороги
    thresholds = [0.45, 0.47, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55, 0.60]
    
    print(f"📁 Тестовый файл: {os.path.basename(test_file)}")
    print(f"🎯 Тестируем пороги: {thresholds}")
    print()
    
    results = []
    
    # Сначала получаем базовый результат без AI
    print("📊 Получаем базовый результат...")
    baseline_strategy = RSIStrategyBase(
        use_custom_rsi=True,
        use_neural_filter=False
    )
    
    # Обрабатываем файл базовой стратегией
    try:
        with gzip.open(test_file, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row['price'])
                volume = float(row['volume'])
                dt = timestamp_to_dt(row['timestamp'])
                baseline_strategy.on_tick(price, dt, volume)
        
        baseline_strategy.on_finish(price)
        baseline_return = (baseline_strategy.equity - 1.0) * 100
        baseline_trades = len(baseline_strategy.trades)
        
        print(f"📈 Базовая стратегия: {baseline_return:+.2f}%, {baseline_trades} сделок")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка базового теста: {e}")
        return
    
    # Тестируем каждый порог
    for threshold in thresholds:
        print(f"🎯 Тестируем порог {threshold}...")
        
        ai_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=True,
            neural_confidence_threshold=threshold
        )
        
        # Счетчики для отладки
        ai_stats = {
            'potential_signals': 0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'avg_confidence': []
        }
        
        # Перехватываем AI решения
        original_on_tick = ai_strategy.on_tick
        
        def debug_on_tick(price, dt, volume=0):
            result = original_on_tick(price, dt, volume)
            
            # Отслеживаем AI решения
            if len(ai_strategy.candles) > 20 and len(ai_strategy.rsi_values) > 0:
                current_rsi = ai_strategy.rsi_values[-1]
                
                if (current_rsi < 30 or current_rsi > 70) and ai_strategy.position == 0:
                    ai_stats['potential_signals'] += 1
                    
                    if ai_strategy.use_neural_filter and ai_strategy.neural_filter:
                        try:
                            # Подготавливаем признаки
                            lookback = min(20, len(ai_strategy.rsi_values))
                            recent_rsi = ai_strategy.rsi_values[-lookback:]
                            recent_bb = ai_strategy.bb_values[-lookback:]
                            recent_atr = ai_strategy.atr_values[-lookback:]
                            recent_vol_ratio = ai_strategy.volatility_ratios[-lookback:]
                            recent_prices = [c.close for c in ai_strategy.candles[-lookback:]]
                            
                            features = ai_strategy.neural_filter.prepare_features(
                                recent_rsi, recent_bb, recent_atr, recent_vol_ratio, recent_prices
                            )
                            
                            if features is not None:
                                confidence = ai_strategy.neural_filter.predict(features)
                                ai_stats['avg_confidence'].append(confidence)
                                
                                neural_approved, neural_confidence = ai_strategy.neural_filter.should_trade(
                                    features, threshold
                                )
                                
                                if neural_approved:
                                    ai_stats['ai_approved'] += 1
                                else:
                                    ai_stats['ai_rejected'] += 1
                                    
                        except Exception as e:
                            pass  # Игнорируем ошибки для скорости
            
            return result
        
        ai_strategy.on_tick = debug_on_tick
        
        # Обрабатываем файл
        try:
            with gzip.open(test_file, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    ai_strategy.on_tick(price, dt, volume)
            
            ai_strategy.on_finish(price)
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
        
        # Сохраняем результаты
        ai_return = (ai_strategy.equity - 1.0) * 100
        ai_trades = len(ai_strategy.trades)
        
        result = {
            'threshold': threshold,
            'return': ai_return,
            'trades': ai_trades,
            'potential_signals': ai_stats['potential_signals'],
            'ai_approved': ai_stats['ai_approved'],
            'ai_rejected': ai_stats['ai_rejected'],
            'avg_confidence': np.mean(ai_stats['avg_confidence']) if ai_stats['avg_confidence'] else 0.5,
            'improvement': ai_return - baseline_return
        }
        
        results.append(result)
        
        # Выводим результат
        filter_rate = ai_stats['ai_rejected'] / max(ai_stats['potential_signals'], 1) * 100
        print(f"   💰 Доходность: {ai_return:+.2f}% (изменение: {result['improvement']:+.2f}%)")
        print(f"   🔄 Сделки: {ai_trades} из {baseline_trades} возможных")
        print(f"   🚫 Фильтрация: {filter_rate:.1f}% ({ai_stats['ai_rejected']}/{ai_stats['potential_signals']})")
        print(f"   🎯 Средняя уверенность: {result['avg_confidence']:.3f}")
        print()
    
    # Анализируем результаты
    analyze_threshold_results(results, baseline_return, baseline_trades)

def analyze_threshold_results(results, baseline_return, baseline_trades):
    """Анализирует результаты оптимизации порога"""
    
    print("\n📊 АНАЛИЗ ОПТИМИЗАЦИИ ПОРОГА:")
    print("=" * 70)
    
    # Сортируем по улучшению
    results_sorted = sorted(results, key=lambda x: x['improvement'], reverse=True)
    
    print("🏆 ТОП-3 ЛУЧШИХ ПОРОГА:")
    for i, result in enumerate(results_sorted[:3], 1):
        filter_rate = result['ai_rejected'] / max(result['potential_signals'], 1) * 100
        print(f"{i}. Порог {result['threshold']}: {result['improvement']:+.2f}% улучшение")
        print(f"   💰 Доходность: {result['return']:+.2f}%")
        print(f"   🔄 Сделки: {result['trades']} ({result['trades']/baseline_trades*100:.1f}% от базы)")
        print(f"   🚫 Фильтрация: {filter_rate:.1f}%")
        print()
    
    # Ищем оптимальный баланс
    print("🎯 РЕКОМЕНДАЦИИ:")
    
    best_improvement = results_sorted[0]
    
    # Ищем порог с хорошим балансом доходности и количества сделок
    balanced_results = [r for r in results if r['trades'] >= baseline_trades * 0.1 and r['improvement'] > 0]
    
    if balanced_results:
        best_balanced = max(balanced_results, key=lambda x: x['improvement'])
        print(f"✅ ОПТИМАЛЬНЫЙ ПОРОГ: {best_balanced['threshold']}")
        print(f"   💰 Улучшение доходности: {best_balanced['improvement']:+.2f}%")
        print(f"   🔄 Сохранено сделок: {best_balanced['trades']}/{baseline_trades}")
        print(f"   🎯 Средняя уверенность: {best_balanced['avg_confidence']:.3f}")
        
        # Обновляем конфиг
        update_config_with_optimal_threshold(best_balanced['threshold'])
        
    elif best_improvement['improvement'] > 0:
        print(f"⚠️ ЛУЧШИЙ ПОРОГ: {best_improvement['threshold']} (но мало сделок)")
        print(f"   💰 Максимальное улучшение: {best_improvement['improvement']:+.2f}%")
        print(f"   ⚠️ Сделок: {best_improvement['trades']} (очень мало)")
        
    else:
        print(f"❌ НИ ОДИН ПОРОГ НЕ УЛУЧШИЛ РЕЗУЛЬТАТЫ")
        print(f"💡 Возможно, нужно переобучить модель с другими параметрами")
    
    # Показываем распределение уверенности
    confidences = [r['avg_confidence'] for r in results]
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ МОДЕЛИ:")
    print(f"   📈 Минимальная: {min(confidences):.3f}")
    print(f"   📊 Средняя: {np.mean(confidences):.3f}")
    print(f"   📈 Максимальная: {max(confidences):.3f}")

def update_config_with_optimal_threshold(optimal_threshold):
    """Обновляет config.py с оптимальным порогом"""
    
    try:
        # Читаем текущий конфиг
        with open('/home/andreyc/devel/tf_new/config.py', 'r') as f:
            content = f.read()
        
        # Заменяем порог
        import re
        pattern = r'NEURAL_CONFIDENCE_THRESHOLD = [\d.]+' 
        replacement = f'NEURAL_CONFIDENCE_THRESHOLD = {optimal_threshold}'
        
        new_content = re.sub(pattern, replacement, content)
        
        # Записываем обратно
        with open('/home/andreyc/devel/tf_new/config.py', 'w') as f:
            f.write(new_content)
        
        print(f"\n✅ Конфиг обновлен: NEURAL_CONFIDENCE_THRESHOLD = {optimal_threshold}")
        
    except Exception as e:
        print(f"❌ Не удалось обновить конфиг: {e}")

def main():
    """Основная функция"""
    test_threshold_optimization()

if __name__ == "__main__":
    main()

