#!/usr/bin/env python3
"""
📊 Алгоритмы определения уровней поддержки и сопротивления
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SRLevel:
    """📊 Уровень поддержки/сопротивления"""
    price: float
    level_type: str  # 'support' или 'resistance'
    strength: float  # Сила уровня (0-1)
    touches: int     # Количество касаний
    first_touch: datetime
    last_touch: datetime
    is_active: bool = True

class SupportResistanceFinder:
    """🔍 Поиск уровней поддержки и сопротивления"""
    
    def __init__(self, min_touches=2, tolerance_pct=0.1, lookback=100, min_strength=0.3):
        """
        Args:
            min_touches: Минимальное количество касаний для валидного уровня
            tolerance_pct: Допуск в % для группировки уровней
            lookback: Количество свечей для анализа
            min_strength: Минимальная сила уровня
        """
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct
        self.lookback = lookback
        self.min_strength = min_strength
        
    def find_pivot_points(self, candles, window=5):
        """🔍 Находим пивотные точки (локальные максимумы и минимумы)"""
        if len(candles) < window * 2 + 1:
            return [], []
        
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        times = [c.start_time for c in candles]
        
        pivot_highs = []  # Потенциальные уровни сопротивления
        pivot_lows = []   # Потенциальные уровни поддержки
        
        # Ищем локальные максимумы и минимумы
        for i in range(window, len(candles) - window):
            # Локальный максимум (resistance)
            if all(highs[i] >= highs[j] for j in range(i - window, i + window + 1)):
                pivot_highs.append((highs[i], times[i], i))
            
            # Локальный минимум (support)  
            if all(lows[i] <= lows[j] for j in range(i - window, i + window + 1)):
                pivot_lows.append((lows[i], times[i], i))
        
        return pivot_highs, pivot_lows
    
    def cluster_levels(self, pivot_points, current_price):
        """📊 Группируем близкие уровни в кластеры"""
        if not pivot_points:
            return []
        
        # Сортируем по цене
        sorted_points = sorted(pivot_points, key=lambda x: x[0])
        clusters = []
        current_cluster = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            price, time, idx = point
            cluster_center = np.mean([p[0] for p in current_cluster])
            
            # Если точка близко к кластеру, добавляем
            if abs(price - cluster_center) / current_price <= self.tolerance_pct / 100:
                current_cluster.append(point)
            else:
                # Сохраняем текущий кластер и начинаем новый
                if len(current_cluster) >= self.min_touches:
                    clusters.append(current_cluster)
                current_cluster = [point]
        
        # Не забываем последний кластер
        if len(current_cluster) >= self.min_touches:
            clusters.append(current_cluster)
        
        return clusters
    
    def calculate_level_strength(self, cluster, candles, current_price):
        """💪 Вычисляем силу уровня"""
        if not cluster:
            return 0.0
        
        # Базовые факторы силы
        touches = len(cluster)
        
        # 1. Количество касаний (больше = сильнее)
        touch_strength = min(touches / 5.0, 1.0)  # Нормализуем к 1.0
        
        # 2. Временной разброс (больший период = сильнее)
        times = [point[1] for point in cluster]
        time_span = (max(times) - min(times)).total_seconds() / 3600  # в часах
        time_strength = min(time_span / 24.0, 1.0)  # Нормализуем к суткам
        
        # 3. Недавность (более свежие уровни = сильнее)
        last_touch = max(times)
        hours_since = (candles[-1].start_time - last_touch).total_seconds() / 3600
        recency_strength = max(0, 1.0 - hours_since / 168)  # 168 часов = неделя
        
        # 4. Объем на уровне (если есть данные)
        avg_volume = np.mean([candles[point[2]].volume for point in cluster if point[2] < len(candles)])
        total_volume = sum(c.volume for c in candles[-50:])  # Последние 50 свечей
        volume_strength = min(avg_volume / (total_volume / 50), 2.0) / 2.0 if total_volume > 0 else 0.5
        
        # 5. Расстояние от текущей цены (ближе = актуальнее)
        level_price = np.mean([point[0] for point in cluster])
        distance_pct = abs(level_price - current_price) / current_price
        distance_strength = max(0, 1.0 - distance_pct * 10)  # Снижаем силу с расстоянием
        
        # Итоговая сила (взвешенная сумма)
        total_strength = (
            touch_strength * 0.3 +
            time_strength * 0.2 +
            recency_strength * 0.2 +
            volume_strength * 0.15 +
            distance_strength * 0.15
        )
        
        return min(total_strength, 1.0)
    
    def find_support_resistance_levels(self, candles):
        """🎯 Основной метод поиска уровней S&R"""
        if len(candles) < self.lookback:
            return []
        
        # Берем последние свечи для анализа
        recent_candles = candles[-self.lookback:]
        current_price = recent_candles[-1].close
        
        # 1. Находим пивотные точки
        pivot_highs, pivot_lows = self.find_pivot_points(recent_candles)
        
        levels = []
        
        # 2. Обрабатываем уровни сопротивления
        if pivot_highs:
            resistance_clusters = self.cluster_levels(pivot_highs, current_price)
            
            for cluster in resistance_clusters:
                strength = self.calculate_level_strength(cluster, recent_candles, current_price)
                
                if strength >= self.min_strength:
                    level_price = np.mean([point[0] for point in cluster])
                    times = [point[1] for point in cluster]
                    
                    level = SRLevel(
                        price=level_price,
                        level_type='resistance',
                        strength=strength,
                        touches=len(cluster),
                        first_touch=min(times),
                        last_touch=max(times)
                    )
                    levels.append(level)
        
        # 3. Обрабатываем уровни поддержки
        if pivot_lows:
            support_clusters = self.cluster_levels(pivot_lows, current_price)
            
            for cluster in support_clusters:
                strength = self.calculate_level_strength(cluster, recent_candles, current_price)
                
                if strength >= self.min_strength:
                    level_price = np.mean([point[0] for point in cluster])
                    times = [point[1] for point in cluster]
                    
                    level = SRLevel(
                        price=level_price,
                        level_type='support',
                        strength=strength,
                        touches=len(cluster),
                        first_touch=min(times),
                        last_touch=max(times)
                    )
                    levels.append(level)
        
        # 4. Сортируем по силе (сильнейшие первыми)
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels
    
    def get_nearest_levels(self, levels, current_price, max_distance_pct=5.0):
        """🎯 Получить ближайшие уровни к текущей цене"""
        nearest = {
            'support': None,
            'resistance': None,
            'all_nearby': []
        }
        
        for level in levels:
            distance_pct = abs(level.price - current_price) / current_price * 100
            
            if distance_pct <= max_distance_pct:
                nearest['all_nearby'].append(level)
                
                if level.level_type == 'support' and level.price < current_price:
                    if (nearest['support'] is None or 
                        abs(level.price - current_price) < abs(nearest['support'].price - current_price)):
                        nearest['support'] = level
                
                elif level.level_type == 'resistance' and level.price > current_price:
                    if (nearest['resistance'] is None or 
                        abs(level.price - current_price) < abs(nearest['resistance'].price - current_price)):
                        nearest['resistance'] = level
        
        return nearest

def test_support_resistance():
    """🧪 Тест алгоритма S&R"""
    from backtester import run_backtest_on_file
    
    print("🧪 ТЕСТ АЛГОРИТМА ПОДДЕРЖКИ/СОПРОТИВЛЕНИЯ")
    print("=" * 60)
    
    # Загружаем данные
    strategy = run_backtest_on_file("data/BTCUSDT_2025-01-01.csv.gz", plot=False, verbose=False)
    
    if len(strategy.candles) < 100:
        print("❌ Недостаточно данных для тестирования")
        return
    
    # Создаем анализатор
    sr_finder = SupportResistanceFinder(min_touches=2, tolerance_pct=0.1, lookback=200)
    
    # Находим уровни
    levels = sr_finder.find_support_resistance_levels(strategy.candles)
    
    print(f"📊 Найдено уровней S&R: {len(levels)}")
    
    if levels:
        current_price = strategy.candles[-1].close
        print(f"💰 Текущая цена: ${current_price:.2f}")
        print()
        
        # Показываем топ-5 уровней
        print("🏆 ТОП-5 СИЛЬНЕЙШИХ УРОВНЕЙ:")
        for i, level in enumerate(levels[:5], 1):
            distance_pct = (level.price - current_price) / current_price * 100
            emoji = "🟢" if level.level_type == 'support' else "🔴"
            
            print(f"{i}. {emoji} {level.level_type.upper()}: ${level.price:.2f}")
            print(f"   💪 Сила: {level.strength:.3f}")
            print(f"   👆 Касаний: {level.touches}")
            print(f"   📏 Расстояние: {distance_pct:+.2f}%")
            print(f"   🕐 Последнее касание: {level.last_touch.strftime('%H:%M')}")
            print()
        
        # Ближайшие уровни
        nearest = sr_finder.get_nearest_levels(levels, current_price)
        
        print("🎯 БЛИЖАЙШИЕ УРОВНИ:")
        if nearest['support']:
            s = nearest['support']
            dist = (current_price - s.price) / current_price * 100
            print(f"🟢 Поддержка: ${s.price:.2f} (-{dist:.2f}%, сила: {s.strength:.3f})")
        
        if nearest['resistance']:
            r = nearest['resistance']
            dist = (r.price - current_price) / current_price * 100
            print(f"🔴 Сопротивление: ${r.price:.2f} (+{dist:.2f}%, сила: {r.strength:.3f})")
        
        print(f"📍 Всего близких уровней: {len(nearest['all_nearby'])}")

if __name__ == '__main__':
    test_support_resistance()




