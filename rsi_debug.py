#!/usr/bin/env python3.12
"""
Утилита для отладки и сравнения RSI значений
Помогает понять, почему RSI бота отличается от терминала
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
import json

# Попробуем импортировать pybit, если доступен
try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    print("⚠️ pybit не установлен. Только локальная проверка RSI.")

def compute_rsi_detailed(prices, period=14, verbose=True):
    """
    Детальный расчет RSI с выводом промежуточных значений
    """
    prices = np.array(prices)
    if len(prices) < period + 1:
        if verbose:
            print(f"❌ Недостаточно данных: {len(prices)} < {period + 1}")
        return None
    
    # Берем последние period+1 значений для расчета
    recent_prices = prices[-(period+1):]
    deltas = np.diff(recent_prices)
    
    if verbose:
        print(f"📊 RSI Расчет (период {period}):")
        print(f"   Последние {period+1} цен: {recent_prices}")
        print(f"   Изменения цен (deltas): {deltas}")
    
    # Разделяем на прибыли и убытки
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]
    
    if verbose:
        print(f"   Прибыли: {gains}")
        print(f"   Убытки: {losses}")
    
    # Средние значения
    avg_gain = gains.sum() / period if len(gains) > 0 else 0
    avg_loss = losses.sum() / period if len(losses) > 0 else 0
    
    if verbose:
        print(f"   Средняя прибыль: {avg_gain:.6f}")
        print(f"   Средний убыток: {avg_loss:.6f}")
    
    # RSI расчет
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    if verbose:
        print(f"   RS (Relative Strength): {avg_gain/avg_loss if avg_loss > 0 else 'inf'}")
        print(f"   RSI: {rsi:.2f}")
    
    return rsi

def get_bybit_klines(symbol="BTCUSDT", interval="5", limit=50):
    """Получает свечи с Bybit"""
    if not PYBIT_AVAILABLE:
        return None
        
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    testnet = os.environ.get('TESTNET', '1') == '1'
    
    if not (api_key and api_secret):
        print("❌ API ключи не настроены")
        return None
        
    try:
        http = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        print(f"📈 Получение свечей с Bybit ({'TESTNET' if testnet else 'MAINNET'}):")
        print(f"   Символ: {symbol}, Интервал: {interval}m, Лимит: {limit}")
        
        klines = http.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        # Преобразуем в удобный формат (от старых к новым)
        raw_candles = klines['result']['list'][::-1]
        candles = []
        
        for c in raw_candles:
            candle = {
                'timestamp': int(c[0]),
                'open': float(c[1]),
                'high': float(c[2]),
                'low': float(c[3]),
                'close': float(c[4]),
                'volume': float(c[5]),
                'datetime': datetime.fromtimestamp(int(c[0]) / 1000, timezone.utc)
            }
            candles.append(candle)
        
        print(f"✅ Получено {len(candles)} свечей")
        return candles
        
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")
        return None

def compare_rsi_methods(prices, period=14):
    """Сравнивает разные методы расчета RSI"""
    print(f"\n🔍 Сравнение методов расчета RSI (период {period}):")
    print("=" * 60)
    
    # Метод 1: Наш текущий метод
    from rsi_strategy import compute_rsi
    rsi1 = compute_rsi(prices, period)
    print(f"Метод бота (простое среднее): {rsi1:.4f}")
    
    # Метод 2: Детальный расчет
    rsi2 = compute_rsi_detailed(prices, period, verbose=False)
    print(f"Детальный расчет:             {rsi2:.4f}")
    
    # Метод 3: Экспоненциальное сглаживание (как в TradingView)
    rsi3 = compute_rsi_ema(prices, period)
    print(f"EMA метод (TradingView):       {rsi3:.4f}")
    
    print(f"\nРазличия:")
    if rsi2:
        print(f"   Бот vs Детальный: {abs(rsi1 - rsi2):.4f}")
        print(f"   Бот vs EMA:       {abs(rsi1 - rsi3):.4f}")
        print(f"   Детальный vs EMA: {abs(rsi2 - rsi3):.4f}")

def compute_rsi_ema(prices, period=14, alpha=None):
    """
    RSI с экспоненциальным сглаживанием (как в TradingView)
    """
    prices = np.array(prices)
    if len(prices) < period + 1:
        return 50.0
    
    if alpha is None:
        alpha = 2.0 / (period + 1)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Первое значение - простое среднее
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # EMA для остальных значений
    for i in range(period, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def analyze_candles(candles, period=14):
    """Анализирует свечи и показывает RSI"""
    if not candles or len(candles) < period + 1:
        print("❌ Недостаточно данных для анализа")
        return
    
    closes = [c['close'] for c in candles]
    
    print(f"\n📊 Анализ последних {len(candles)} свечей:")
    print("=" * 60)
    
    # Показываем последние 5 свечей
    print("Последние 5 свечей:")
    for i, candle in enumerate(candles[-5:], len(candles)-4):
        dt_str = candle['datetime'].strftime('%H:%M')
        print(f"   {i:2d}. {dt_str} | O:{candle['open']:8.1f} H:{candle['high']:8.1f} L:{candle['low']:8.1f} C:{candle['close']:8.1f}")
    
    # Сравниваем методы RSI
    compare_rsi_methods(closes, period)
    
    # Показываем изменения цен
    print(f"\nИзменения цен (последние 10):")
    for i in range(max(0, len(closes)-10), len(closes)-1):
        change = closes[i+1] - closes[i]
        print(f"   {closes[i]:8.1f} → {closes[i+1]:8.1f} ({change:+6.1f})")

def save_debug_data(candles, filename="rsi_debug_data.json"):
    """Сохраняет данные для дальнейшего анализа"""
    if not candles:
        return
        
    data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'candles': []
    }
    
    for candle in candles:
        data['candles'].append({
            'timestamp': candle['timestamp'],
            'datetime': candle['datetime'].isoformat(),
            'open': candle['open'],
            'high': candle['high'], 
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Данные сохранены в {filename}")

def main():
    print("🔍 RSI Debug Tool")
    print("=" * 50)
    
    # Получаем данные с Bybit
    candles = get_bybit_klines()
    
    if candles:
        # Анализируем полученные данные
        analyze_candles(candles)
        
        # Сохраняем для дальнейшего анализа
        save_debug_data(candles)
    else:
        # Тестовые данные для проверки алгоритма
        print("\n🧪 Тестирование на примере данных:")
        test_prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 
                      46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46.00, 46.03]
        
        print(f"Тестовые цены: {test_prices}")
        compare_rsi_methods(test_prices, 14)
    
    print("\n" + "=" * 50)
    print("Возможные причины расхождений:")
    print("1. 🕐 Разное время свечей (UTC vs локальное)")
    print("2. 📊 Разные методы расчета (простое vs EMA)")
    print("3. 🔢 Разное количество периодов")
    print("4. 📈 Разные источники данных")
    print("5. ⏰ Текущая vs закрытая свеча")

if __name__ == "__main__":
    main()
