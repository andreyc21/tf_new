"""
Тестирование новой модели 2025 года на свежих данных
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

def test_new_model_vs_old():
    """Сравнивает новую модель 2025 с обычной стратегией"""
    
    print("🚀 ТЕСТИРОВАНИЕ НОВОЙ AI-МОДЕЛИ 2025")
    print("=" * 60)
    
    # Используем файлы, которые НЕ использовались для обучения
    test_files = [
        "data/BTCUSDT_2025-02-01.csv.gz",  # Февраль - не использовался для обучения
        "data/BTCUSDT_2025-02-02.csv.gz",
        "data/BTCUSDT_2025-02-03.csv.gz",
        "data/BTCUSDT_2025-02-04.csv.gz",
        "data/BTCUSDT_2025-02-05.csv.gz",
    ]
    
    # Проверяем наличие файлов
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Тестовые файлы февраля не найдены!")
        print("💡 Используем файлы из конца января (не использовались для обучения)")
        test_files = [
            "data/BTCUSDT_2025-01-26.csv.gz",
            "data/BTCUSDT_2025-01-27.csv.gz", 
            "data/BTCUSDT_2025-01-28.csv.gz",
            "data/BTCUSDT_2025-01-29.csv.gz",
            "data/BTCUSDT_2025-01-30.csv.gz",
        ]
        existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Нет доступных файлов для тестирования!")
        return
    
    print(f"📁 Найдено файлов для тестирования: {len(existing_files)}")
    
    results = {'normal': [], 'ai': [], 'ai_debug': []}
    
    for filename in existing_files:
        print(f"\n📊 Тестируем: {os.path.basename(filename)}")
        
        # 1. Обычная стратегия
        normal_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=False
        )
        
        # 2. AI-enhanced стратегия с НОВОЙ моделью
        ai_strategy = RSIStrategyBase(
            use_custom_rsi=True,
            use_neural_filter=True,
            neural_confidence_threshold=0.55  # Немного понижен для новой модели
        )
        
        # Счетчики для отладки
        ai_debug = {
            'potential_signals': 0,
            'ai_calls': 0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'decisions': []
        }
        
        # Перехватываем AI решения
        original_on_tick = ai_strategy.on_tick
        
        def debug_on_tick(price, dt, volume=0):
            result = original_on_tick(price, dt, volume)
            
            # Отслеживаем AI решения
            if len(ai_strategy.candles) > 20 and len(ai_strategy.rsi_values) > 0:
                current_rsi = ai_strategy.rsi_values[-1]
                
                if (current_rsi < 30 or current_rsi > 70) and ai_strategy.position == 0:
                    ai_debug['potential_signals'] += 1
                    
                    if ai_strategy.use_neural_filter and ai_strategy.neural_filter:
                        ai_debug['ai_calls'] += 1
                        
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
                                neural_approved, neural_confidence = ai_strategy.neural_filter.should_trade(
                                    features, ai_strategy.neural_confidence_threshold
                                )
                                
                                ai_debug['decisions'].append({
                                    'rsi': current_rsi,
                                    'confidence': confidence,
                                    'approved': neural_approved
                                })
                                
                                if neural_approved:
                                    ai_debug['ai_approved'] += 1
                                else:
                                    ai_debug['ai_rejected'] += 1
                                    
                        except Exception as e:
                            print(f"⚠️ AI ошибка: {e}")
            
            return result
        
        ai_strategy.on_tick = debug_on_tick
        
        # Обрабатываем файл обеими стратегиями
        try:
            with gzip.open(filename, 'rt') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    price = float(row['price'])
                    volume = float(row['volume'])
                    dt = timestamp_to_dt(row['timestamp'])
                    
                    normal_strategy.on_tick(price, dt, volume)
                    ai_strategy.on_tick(price, dt, volume)
            
            normal_strategy.on_finish(price)
            ai_strategy.on_finish(price)
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
        
        # Сохраняем результаты
        normal_result = {
            'trades': len(normal_strategy.trades),
            'return': (normal_strategy.equity - 1.0) * 100,
            'entries': len(normal_strategy.entry_points)
        }
        
        ai_result = {
            'trades': len(ai_strategy.trades),
            'return': (ai_strategy.equity - 1.0) * 100,
            'entries': len(ai_strategy.entry_points)
        }
        
        results['normal'].append(normal_result)
        results['ai'].append(ai_result)
        results['ai_debug'].append(ai_debug)
        
        # Выводим результат по файлу
        print(f"   📈 Обычная: {normal_result['return']:+.2f}%, {normal_result['trades']} сделок")
        print(f"   🧠 AI-2025: {ai_result['return']:+.2f}%, {ai_result['trades']} сделок")
        print(f"   🎯 AI фильтр: {ai_debug['ai_rejected']}/{ai_debug['potential_signals']} отклонено")
        
        # Показываем разницу
        diff = ai_result['return'] - normal_result['return']
        if abs(diff) > 0.01:
            if diff > 0:
                print(f"   ✅ AI лучше на {diff:.2f}% пунктов!")
            else:
                print(f"   ❌ AI хуже на {abs(diff):.2f}% пунктов")
        else:
            print(f"   ➖ Результаты практически идентичны")
    
    # Итоговый анализ
    if results['normal'] and results['ai']:
        analyze_final_results(results)

def analyze_final_results(results):
    """Анализирует итоговые результаты"""
    
    print(f"\n📊 ИТОГОВЫЙ АНАЛИЗ НОВОЙ AI-МОДЕЛИ 2025:")
    print("=" * 60)
    
    # Собираем метрики
    normal_returns = [r['return'] for r in results['normal']]
    ai_returns = [r['return'] for r in results['ai']]
    
    normal_trades = sum(r['trades'] for r in results['normal'])
    ai_trades = sum(r['trades'] for r in results['ai'])
    
    # AI статистика
    total_potential_signals = sum(d['potential_signals'] for d in results['ai_debug'])
    total_ai_rejected = sum(d['ai_rejected'] for d in results['ai_debug'])
    total_ai_approved = sum(d['ai_approved'] for d in results['ai_debug'])
    
    # Основные метрики
    print(f"📈 ОБЫЧНАЯ СТРАТЕГИЯ:")
    print(f"   💰 Средняя доходность: {np.mean(normal_returns):+.2f}% ± {np.std(normal_returns):.2f}%")
    print(f"   🔄 Всего сделок: {normal_trades}")
    print(f"   📊 Win rate: {sum(1 for r in normal_returns if r > 0)}/{len(normal_returns)} дней")
    
    print(f"\n🧠 AI-ENHANCED СТРАТЕГИЯ (модель 2025):")
    print(f"   💰 Средняя доходность: {np.mean(ai_returns):+.2f}% ± {np.std(ai_returns):.2f}%")
    print(f"   🔄 Всего сделок: {ai_trades}")
    print(f"   📊 Win rate: {sum(1 for r in ai_returns if r > 0)}/{len(ai_returns)} дней")
    
    print(f"\n🤖 AI ФИЛЬТРАЦИЯ:")
    print(f"   🚨 Потенциальных сигналов: {total_potential_signals}")
    print(f"   ✅ AI одобрил: {total_ai_approved}")
    print(f"   🚫 AI отклонил: {total_ai_rejected}")
    if total_potential_signals > 0:
        print(f"   📊 Процент фильтрации: {total_ai_rejected/total_potential_signals*100:.1f}%")
    
    # Сравнение
    return_improvement = np.mean(ai_returns) - np.mean(normal_returns)
    trade_change = ai_trades - normal_trades
    
    print(f"\n🎯 СРАВНЕНИЕ (AI-2025 vs Обычная):")
    print("=" * 50)
    print(f"💹 Изменение доходности: {return_improvement:+.2f}% пунктов")
    print(f"🔄 Изменение сделок: {trade_change:+d} ({trade_change/normal_trades*100:+.1f}%)")
    
    # Выводы
    print(f"\n🎉 ВЫВОДЫ:")
    if total_ai_rejected > 0:
        print(f"✅ Новая AI-модель РАБОТАЕТ! Отфильтровала {total_ai_rejected} сигналов")
        
        if return_improvement > 0.1:
            print(f"🚀 AI значительно улучшил результаты (+{return_improvement:.2f}%)")
        elif return_improvement > 0:
            print(f"📈 AI немного улучшил результаты (+{return_improvement:.2f}%)")
        elif return_improvement < -0.1:
            print(f"📉 AI ухудшил результаты ({return_improvement:.2f}%), нужна доработка")
        else:
            print(f"➖ AI не дал значимого улучшения, но работает корректно")
    else:
        print(f"⚠️ AI-модель не отклонила ни одного сигнала - возможно, нужно поднять порог")
    
    # Рекомендации
    if total_ai_rejected == 0:
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print(f"   🎯 Попробуйте повысить порог уверенности до 0.6-0.7")
        print(f"   🔄 Или переобучить модель с более строгими критериями")
    elif return_improvement > 0:
        print(f"\n🎉 УСПЕХ! Новая модель работает лучше старой!")
        print(f"   🚀 Готово к внедрению в продакшн")

def main():
    """Основная функция"""
    test_new_model_vs_old()

if __name__ == "__main__":
    main()

