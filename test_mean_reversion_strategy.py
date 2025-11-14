#!/usr/bin/env python3
"""
Стратегия Mean Reversion на основе Bollinger Bands и режимов рынка:
- Основной принцип: курс возвращается после отклонения за стандартное отклонение
- Режимы рынка:
  1. Сильный тренд: не входим
  2. Устойчивый тренд/флет: входим только по тренду
  3. Не полный штиль: входим в обе стороны (mean reversion)
"""

import sys
import gzip
import csv
import glob
from pathlib import Path
from datetime import datetime
from collections import namedtuple
from typing import Optional, List
from enum import Enum

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Структуры данных
Candle = namedtuple('Candle', ['start_time', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume'])
Trade = namedtuple('Trade', ['entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'pnl_pct', 
                              'reason', 'entry_reason', 'expected_profit', 'expected_loss', 'risk_reward'])


class TrendStrength(Enum):
    """Сила тренда"""
    STRONG = "strong"      # Сильный тренд - не торгуем
    STABLE = "stable"      # Устойчивый тренд/флет - только по тренду
    CALM = "calm"          # Не полный штиль - в обе стороны


def classify_tick_side(tick, prev_price=None, last_side='buy'):
    """
    Классифицирует тик как buy или sell
    
    Если цена выросла относительно предыдущей - это buy (агрессор покупатель)
    Если цена упала - это sell (агрессор продавец)
    Если цена не изменилась - сохраняем последнее направление (tick rule)
    
    Args:
        tick: словарь с данными тика
        prev_price: предыдущая цена
        last_side: последнее определённое направление ('buy' или 'sell')
    
    Returns:
        'buy' или 'sell'
    """
    if prev_price is None:
        # Первый тик - считаем neutral (buy)
        return 'buy'
    
    price = tick['price']
    
    if price > prev_price:
        return 'buy'  # Покупатель поднял цену
    elif price < prev_price:
        return 'sell'  # Продавец опустил цену
    else:
        # Цена не изменилась - используем tick rule (сохраняем направление)
        return last_side


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """Расширяет glob-паттерны и возвращает список файлов"""
    files = []
    for pattern in patterns:
        # Если паттерн содержит glob символы, расширяем его
        if '*' in pattern or '?' in pattern or '[' in pattern:
            expanded = glob.glob(pattern)
            if not expanded:
                print(f"⚠️ Паттерн не дал результатов: {pattern}")
            files.extend(sorted(expanded))
        else:
            # Обычный файл
            files.append(pattern)
    return files


def load_ticks(filename: str):
    """Загружает тики из одного csv.gz файла"""
    ticks = []
    with gzip.open(filename, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_ms = int(row['timestamp'])
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
            ticks.append({
                'timestamp': timestamp_ms / 1000.0,
                'datetime': dt,
                'price': float(row['price']),
                'volume': float(row.get('volume', 0.0))
            })
    return ticks


def load_ticks_from_files(filenames: List[str]) -> List[dict]:
    """Загружает и объединяет тики из нескольких файлов"""
    all_ticks = []
    
    print(f"📥 Загрузка тиков из {len(filenames)} файл(ов)...")
    
    for filename in filenames:
        print(f"   📄 {filename}...", end=" ")
        ticks = load_ticks(filename)
        print(f"{len(ticks):,} тиков")
        all_ticks.extend(ticks)
    
    # Сортируем все тики по времени (на случай если файлы не в хронологическом порядке)
    print(f"🔄 Сортировка {len(all_ticks):,} тиков по времени...")
    all_ticks.sort(key=lambda t: t['timestamp'])
    
    if all_ticks:
        print(f"✅ Период: {all_ticks[0]['datetime']} → {all_ticks[-1]['datetime']}")
    
    return all_ticks


def compute_ema(prices, period):
    """Вычисляет EMA (начинается с SMA для корректной инициализации)"""
    if len(prices) < period:
        return None
    
    prices_arr = np.array(prices)
    alpha = 2.0 / (period + 1)
    
    # Инициализируем EMA как SMA первых period значений
    ema = np.mean(prices_arr[:period])
    
    # Применяем EMA для остальных значений
    if len(prices_arr) > period:
        for price in prices_arr[period:]:
            ema = alpha * price + (1 - alpha) * ema
    
    return ema


def compute_bollinger_bands(prices, period=20, num_std=2):
    """Вычисляет Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    prices_arr = np.array(prices[-period:])
    ma = np.mean(prices_arr)
    std = np.std(prices_arr)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def get_adaptive_trend_thresholds(volatility):
    """
    Вычисляет адаптивные пороги для определения силы тренда
    на основе текущей волатильности рынка.
    
    Args:
        volatility: Волатильность в % (например, 0.003 = 0.3%)
    
    Returns:
        dict: Пороги для strong, stable, calm режимов
    """
    # Базовый порог масштабируется с волатильностью
    base_threshold = volatility * 0.5
    
    return {
        'strong': base_threshold * 2.0,   # Сильный тренд: двойной базовый порог
        'stable': base_threshold * 1.0,   # Устойчивый тренд: базовый порог
        'calm': base_threshold * 0.5      # Спокойный: половина базового
    }


def detect_trend_strength(candles, ema_period=20, trend_period=10, use_adaptive_thresholds=True):
    """
    Определяет силу тренда и его направление
    
    Returns:
        (trend_direction, trend_strength)
        trend_direction: 1 = восходящий, -1 = нисходящий, 0 = флет
        trend_strength: TrendStrength enum
    """
    if len(candles) < max(ema_period, trend_period):
        return 0, TrendStrength.CALM
    
    closes = [c.close for c in candles]
    
    # EMA для определения направления
    ema = compute_ema(closes, ema_period)
    if ema is None:
        return 0, TrendStrength.CALM
    
    current_price = closes[-1]
    
    # Определяем направление тренда по наклону EMA за последние N свечей
    trend_dir = 0
    ema_slope = 0.0
    
    # ИСПРАВЛЕНИЕ: считаем наклон EMA за последние trend_period свечей, а не за 1 свечу
    if len(closes) >= ema_period + trend_period:
        # EMA на текущих данных (уже рассчитана выше как 'ema')
        current_ema = ema
        # EMA N свечей назад (где N = trend_period)
        past_ema = compute_ema(closes[:-trend_period], ema_period)
        
        if current_ema is not None and past_ema is not None:
            # Наклон за последние trend_period свечей
            ema_slope = (current_ema - past_ema) / past_ema  # Наклон EMA в %
            
            if ema_slope > 0.0001:  # Восходящий (>0.01%)
                trend_dir = 1
            elif ema_slope < -0.0001:  # Нисходящий (<-0.01%)
                trend_dir = -1
    
    # Определяем силу тренда по наклону EMA и волатильности
    if len(closes) >= trend_period and ema_slope != 0.0:
        # Вычисляем волатильность (ATR-like)
        high_low_ranges = []
        for i in range(max(1, len(candles) - trend_period), len(candles)):
            if i > 0:
                high_low = candles[i].high - candles[i].low
                high_close = abs(candles[i].high - candles[i-1].close)
                low_close = abs(candles[i].low - candles[i-1].close)
                true_range = max(high_low, high_close, low_close)
                high_low_ranges.append(true_range)
        
        if high_low_ranges:
            avg_volatility = np.mean(high_low_ranges) / current_price  # Волатильность в %
            
            # Определяем пороги: либо адаптивные, либо фиксированные
            if use_adaptive_thresholds:
                thresholds = get_adaptive_trend_thresholds(avg_volatility)
                strong_threshold = thresholds['strong']
                stable_threshold = thresholds['stable']
                # Для calm используем минимальный порог направления тренда
                min_trend_threshold = 0.0001  # 0.01% - минимум для определения направления
            else:
                # Фиксированные пороги (старая логика)
                strong_threshold = 0.002   # 0.2%
                stable_threshold = 0.0005  # 0.05%
                min_trend_threshold = 0.0001
            
            # Классификация силы тренда с адаптивными порогами
            # Сильный тренд: большой наклон EMA относительно волатильности
            if abs(ema_slope) > strong_threshold:
                trend_strength = TrendStrength.STRONG
            # Устойчивый тренд: умеренный наклон и низкая/средняя волатильность
            elif abs(ema_slope) > stable_threshold and avg_volatility < 0.003:
                trend_strength = TrendStrength.STABLE
            # Спокойный рынок: малый наклон или высокая волатильность
            else:
                trend_strength = TrendStrength.CALM
            
            # Отладочный вывод (можно включить для анализа)
            # print(f"[Trend] slope: {ema_slope:.6f}, vol: {avg_volatility:.6f}, "
            #       f"thresholds(strong: {strong_threshold:.6f}, stable: {stable_threshold:.6f}) "
            #       f"→ {trend_strength.value}")
            
            return trend_dir, trend_strength
    
    # По умолчанию - спокойный рынок
    return trend_dir, TrendStrength.CALM


class MeanReversionStrategy:
    def __init__(self, 
                 candle_minutes=5,
                 trend_tf_minutes=15,
                 bb_period=20,
                 bb_std=2,
                 ema_period=20,
                 trend_period=10,
                 entry_threshold=1.0,  # % от BB для входа (1.0 = точно на BB, >1.0 = за BB)
                 exit_threshold=0.5,   # % от BB средней для выхода
                 stop_loss_multiplier=0.5,  # Множитель для стоп-лосса (доля от ширины BB/2)
                 position_size=1.0,    # Доля капитала на сделку (1.0 = весь капитал)
                 use_risk_management=False,  # Включить риск-менеджмент
                 min_risk_reward_ratio=1.5,  # Минимальное соотношение Risk/Reward
                 use_adaptive_thresholds=True,  # Использовать адаптивные пороги тренда
                 initial_equity=1.0,
                 maker_fee=0.0001,
                 taker_fee=0.0005):
        self.candle_minutes = candle_minutes
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ema_period = ema_period
        self.trend_tf_minutes = trend_tf_minutes
        self.trend_period = trend_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_multiplier = stop_loss_multiplier
        self.position_size = position_size
        self.use_risk_management = use_risk_management
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.use_adaptive_thresholds = use_adaptive_thresholds
        
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # Данные свечей
        self.candles = []                  # рабочий ТФ для сигналов/выходов
        self.candles_trend = []            # старший ТФ для оценки тренда
        
        # Состояние
        self.position = 0  # 0 = нет позиции, 1 = лонг, -1 = шорт
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_reason = ""
        self.expected_profit = 0.0
        self.expected_loss = 0.0
        self.risk_reward = 0.0
        self.prev_price = None  # Для классификации buy/sell
        self.last_side = 'buy'  # Последнее определённое направление тика
        
        # Результаты
        self.trades = []
        self.equity_curve = [initial_equity]
        
        # История трендов для визуализации (timestamp, trend_dir, trend_strength)
        self.trend_history = []
        
        # Логирование изменений тренда
        self._last_trend_dir = None
        self._last_trend_strength = None
        self._current_trend_dir = 0
        self._current_trend_strength = TrendStrength.CALM
        
        # Статистика риск-менеджмента
        self.rejected_trades_rr = 0  # Сделки отклонённые из-за плохого R/R
    
    def process_tick(self, tick):
        """Обрабатывает один тик"""
        dt = tick['datetime']
        price = tick['price']
        volume = tick.get('volume', 0.0)
        
        # Классифицируем тик как buy или sell
        tick_side = classify_tick_side(tick, self.prev_price, self.last_side)
        buy_vol = volume if tick_side == 'buy' else 0.0
        sell_vol = volume if tick_side == 'sell' else 0.0
        
        # Сохраняем текущую цену и направление для следующего тика
        self.prev_price = price
        self.last_side = tick_side
        
        # Обновляем свечи рабочего ТФ
        candle_start = dt.replace(second=0, microsecond=0)
        candle_start = candle_start.replace(minute=(candle_start.minute // self.candle_minutes) * self.candle_minutes)
        
        if not self.candles or self.candles[-1].start_time != candle_start:
            # Новая свеча - open = первый тик новой свечи
            self.candles.append(Candle(
                start_time=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                buy_volume=buy_vol,
                sell_volume=sell_vol
            ))
        else:
            # Обновляем текущую свечу
            last = self.candles[-1]
            self.candles[-1] = last._replace(
                high=max(last.high, price),
                low=min(last.low, price),
                close=price,
                volume=last.volume + volume,
                buy_volume=last.buy_volume + buy_vol,
                sell_volume=last.sell_volume + sell_vol
            )
        
        # Обновляем свечи ТФ тренда (15m/60m)
        trend_candle_start = dt.replace(second=0, microsecond=0)
        trend_candle_start = trend_candle_start.replace(minute=(trend_candle_start.minute // self.trend_tf_minutes) * self.trend_tf_minutes)
        
        # Флаг новой свечи тренд-ТФ
        trend_candle_closed = False
        
        if not self.candles_trend or self.candles_trend[-1].start_time != trend_candle_start:
            # Новая свеча - open = первый тик новой свечи
            trend_candle_closed = len(self.candles_trend) > 0  # Предыдущая свеча закрылась
            
            self.candles_trend.append(Candle(
                start_time=trend_candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                buy_volume=buy_vol,
                sell_volume=sell_vol
            ))
        else:
            last_t = self.candles_trend[-1]
            self.candles_trend[-1] = last_t._replace(
                high=max(last_t.high, price),
                low=min(last_t.low, price),
                close=price,
                volume=last_t.volume + volume,
                buy_volume=last_t.buy_volume + buy_vol,
                sell_volume=last_t.sell_volume + sell_vol
            )

        # Обновляем/логируем тренд только при закрытии свечи тренд-ТФ
        if trend_candle_closed:
            self._update_trend(dt)

        # Проверяем сигналы на каждом тике (для точности)
        if len(self.candles) >= max(self.bb_period, self.ema_period):
            self._process_signals(price, dt)
        
        # Проверяем выходы на каждом тике
        if self.position != 0:
            self._check_exits(price, dt)
        
        self.equity_curve.append(self.equity)
    
    def _process_signals(self, price, dt):
        """Обрабатывает торговые сигналы"""
        if self.position != 0:
            return  # Уже в позиции
        
        if len(self.candles) < max(self.bb_period, self.ema_period):
            return
        
        # КРИТИЧЕСКИ ВАЖНО: проверяем, что тренд определён корректно
        # Тренд считается по старшему ТФ, нужно убедиться что данных достаточно
        if len(self.candles_trend) < max(self.ema_period, self.trend_period):
            return  # Недостаточно данных для определения тренда
        
        # Вычисляем индикаторы
        closes = [c.close for c in self.candles]
        # Добавляем текущую цену для точности расчетов в реальном времени
        closes_with_current = closes + [price]
        
        bb_middle, bb_upper, bb_lower = compute_bollinger_bands(
            closes_with_current, self.bb_period, self.bb_std
        )
        
        if bb_middle is None:
            return
        
        # Используем уже рассчитанный тренд
        trend_dir = self._current_trend_dir
        trend_strength = self._current_trend_strength
        
        # Вычисляем ширину BB и порог входа
        bb_width = bb_upper - bb_lower
        entry_offset = bb_width * (self.entry_threshold - 1.0)  # Дополнительное отклонение от BB
        
        # Условия входа
        long_signal = False
        short_signal = False
        
        # Проверяем выход за BB (верхняя или нижняя граница)
        # Если entry_threshold = 1.0, входим точно на BB
        # Если entry_threshold > 1.0, входим когда цена зашла за BB на дополнительный offset
        upper_entry_level = bb_upper + entry_offset
        lower_entry_level = bb_lower - entry_offset
        
        if price >= upper_entry_level:
            # Цена выше верхней BB (перекупленность)
            if trend_strength == TrendStrength.STRONG:
                # Сильный тренд - не входим (слишком рискованно против сильного движения)
                pass
            elif trend_strength == TrendStrength.STABLE:
                # Устойчивый тренд - входим по тренду при временных отклонениях
                # Цена выше BB (отклонение вверх) + ВОСХОДЯЩИЙ тренд → шорт (возврат к тренду)
                # Логика: в восходящем тренде перекупленность - это временное отклонение
                if trend_dir == 1:
                    short_signal = True
            elif trend_strength == TrendStrength.CALM:
                # Не полный штиль - mean reversion в обе стороны
                # Цена выше BB → шорт (возврат к средней)
                short_signal = True
                #short_signal = False
        
        if price <= lower_entry_level:
            # Цена ниже нижней BB (перепроданность)
            if trend_strength == TrendStrength.STRONG:
                # Сильный тренд - не входим (слишком рискованно против сильного движения)
                pass
            elif trend_strength == TrendStrength.STABLE:
                # Устойчивый тренд - входим по тренду при временных отклонениях
                # Цена ниже BB (отклонение вниз) + НИСХОДЯЩИЙ тренд → лонг (возврат к тренду)
                # Логика: в нисходящем тренде перепроданность - это временное отклонение
                if trend_dir == -1:
                    long_signal = True
            elif trend_strength == TrendStrength.CALM:
                # Не полный штиль - mean reversion в обе стороны
                # Цена ниже BB → лонг (возврат к средней)
                long_signal = True
                #long_signal = False
        
        # Риск-менеджмент: проверяем соотношение Risk/Reward перед входом
        if self.use_risk_management and (long_signal or short_signal):
            # Вычисляем предполагаемый TP и SL
            exit_distance = (bb_upper - bb_middle) * self.exit_threshold
            bb_half_width = (bb_upper - bb_middle)
            stop_loss_distance = bb_half_width * self.stop_loss_multiplier
            
            if long_signal:
                # Лонг: вход когда цена ниже BB, ожидаем возврат вверх к средней
                tp_price = bb_middle + exit_distance  # TP ВЫШЕ средней BB
                sl_price = bb_lower - stop_loss_distance  # SL ниже нижней BB
                
                potential_profit = tp_price - price
                potential_loss = price - sl_price
                
                # Проверяем R/R соотношение
                if potential_loss > 0:
                    risk_reward = potential_profit / potential_loss
                    if risk_reward < self.min_risk_reward_ratio:
                        print(f"❌ ЛОНГ отклонён: R/R={risk_reward:.2f} < {self.min_risk_reward_ratio:.2f} | TP={tp_price:.2f}, SL={sl_price:.2f}")
                        self.rejected_trades_rr += 1
                        long_signal = False
                else:
                    # SL выше входа - некорректная ситуация
                    print(f"❌ ЛОНГ отклонён: SL={sl_price:.2f} >= вход={price:.2f}")
                    self.rejected_trades_rr += 1
                    long_signal = False
            
            elif short_signal:
                # Шорт: вход когда цена выше BB, ожидаем возврат вниз к средней
                tp_price = bb_middle - exit_distance  # TP НИЖЕ средней BB
                sl_price = bb_upper + stop_loss_distance  # SL выше верхней BB
                
                potential_profit = price - tp_price
                potential_loss = sl_price - price
                
                # Проверяем R/R соотношение
                if potential_loss > 0:
                    risk_reward = potential_profit / potential_loss
                    if risk_reward < self.min_risk_reward_ratio:
                        print(f"❌ ШОРТ отклонён: R/R={risk_reward:.2f} < {self.min_risk_reward_ratio:.2f} | TP={tp_price:.2f}, SL={sl_price:.2f}")
                        self.rejected_trades_rr += 1
                        short_signal = False
                else:
                    # SL ниже входа - некорректная ситуация
                    print(f"❌ ШОРТ отклонён: SL={sl_price:.2f} <= вход={price:.2f}")
                    self.rejected_trades_rr += 1
                    short_signal = False
        
        # Открываем позиции (только если прошли риск-менеджмент)
        if long_signal or short_signal:
            # Вычисляем предполагаемые TP/SL для логирования
            exit_distance = (bb_upper - bb_middle) * self.exit_threshold
            bb_half_width = (bb_upper - bb_middle)
            stop_loss_distance = bb_half_width * self.stop_loss_multiplier
            
            if long_signal:
                # Лонг: вход ниже BB, ожидаем возврат вверх
                tp_price = bb_middle + exit_distance  # TP выше средней
                sl_price = bb_lower - stop_loss_distance  # SL ниже нижней
                expected_profit = tp_price - price
                expected_loss = price - sl_price
                rr = expected_profit / expected_loss if expected_loss > 0 else 0
                
                entry_reason = f"Mean reversion лонг | Режим: {trend_strength.value} | Тренд: {'▲' if trend_dir == 1 else '▼' if trend_dir == -1 else '—'}"
                self._open_position(1, price, dt, entry_reason, expected_profit, expected_loss, rr)
            
            elif short_signal:
                # Шорт: вход выше BB, ожидаем возврат вниз
                tp_price = bb_middle - exit_distance  # TP ниже средней
                sl_price = bb_upper + stop_loss_distance  # SL выше верхней
                expected_profit = price - tp_price
                expected_loss = sl_price - price
                rr = expected_profit / expected_loss if expected_loss > 0 else 0
                
                entry_reason = f"Mean reversion шорт | Режим: {trend_strength.value} | Тренд: {'▲' if trend_dir == 1 else '▼' if trend_dir == -1 else '—'}"
                self._open_position(-1, price, dt, entry_reason, expected_profit, expected_loss, rr)

    def _update_trend(self, dt):
        """Обновляет и логирует тренд на основе свечей старшего ТФ."""
        if len(self.candles_trend) < max(self.ema_period, self.trend_period):
            # Недостаточно данных — сбрасываем к спокойному режиму
            self._current_trend_dir = 0
            self._current_trend_strength = TrendStrength.CALM
            return
        trend_dir, trend_strength = detect_trend_strength(
            self.candles_trend, self.ema_period, self.trend_period, self.use_adaptive_thresholds
        )
        # Логируем изменения направления тренда и/или силы
        if self._last_trend_dir is None or self._last_trend_strength is None:
            self._last_trend_dir = trend_dir
            self._last_trend_strength = trend_strength
            # Сохраняем в историю
            self.trend_history.append((dt, trend_dir, trend_strength))
        else:
            dir_changed = (trend_dir != self._last_trend_dir)
            strength_changed = (trend_strength != self._last_trend_strength)
            if dir_changed or strength_changed:
                dir_symbol_old = '▲' if self._last_trend_dir == 1 else '▼' if self._last_trend_dir == -1 else '—'
                dir_symbol_new = '▲' if trend_dir == 1 else '▼' if trend_dir == -1 else '—'
                msg_parts = []
                if dir_changed:
                    msg_parts.append(f"направление {dir_symbol_old} -> {dir_symbol_new}")
                if strength_changed:
                    msg_parts.append(f"сила {self._last_trend_strength.value} -> {trend_strength.value}")
                print(f"ℹ️ Тренд изменился в {dt}: " + ", ".join(msg_parts))
                self._last_trend_dir = trend_dir
                self._last_trend_strength = trend_strength
                # Сохраняем в историю
                self.trend_history.append((dt, trend_dir, trend_strength))
        self._current_trend_dir = trend_dir
        self._current_trend_strength = trend_strength
    
    def _check_exits(self, price, dt):
        """Проверяет условия выхода из позиции"""
        if self.position == 0:
            return
        
        if len(self.candles) < self.bb_period:
            return
        
        closes = [c.close for c in self.candles]
        # Добавляем текущую цену для точности расчетов в реальном времени
        closes_with_current = closes + [price]
        
        bb_middle, bb_upper, bb_lower = compute_bollinger_bands(
            closes_with_current, self.bb_period, self.bb_std
        )
        
        if bb_middle is None:
            return
        
        # Вычисляем порог выхода как расстояние от средней BB
        # exit_threshold указывает долю от расстояния между средней и границей BB (0.5 = 50%)
        exit_distance = (bb_upper - bb_middle) * self.exit_threshold
        
        # Вычисляем стоп-лосс как расстояние за границей BB
        # stop_loss_multiplier указывает, на сколько дальше границы BB ставить SL
        bb_half_width = (bb_upper - bb_middle)
        stop_loss_distance = bb_half_width * self.stop_loss_multiplier
        
        if self.position == 1:  # Лонг
            # Тейк-профит: возврат к средней BB + exit_distance (вверх от средней)
            if price >= bb_middle + exit_distance:
                self._close_position(price, dt, "TP: возврат к BB средней")
            # Стоп-лосс: пробой нижней BB с запасом (вниз)
            elif price <= bb_lower - stop_loss_distance:
                self._close_position(price, dt, "SL: пробой BB нижней")
        
        elif self.position == -1:  # Шорт
            # Тейк-профит: возврат к средней BB - exit_distance (вниз от средней)
            if price <= bb_middle - exit_distance:
                self._close_position(price, dt, "TP: возврат к BB средней")
            # Стоп-лосс: пробой верхней BB с запасом (вверх)
            elif price >= bb_upper + stop_loss_distance:
                self._close_position(price, dt, "SL: пробой BB верхней")
    
    def _open_position(self, direction, price, dt, reason, expected_profit=0, expected_loss=0, risk_reward=0):
        """Открывает позицию"""
        # Защита от двойного входа
        if self.position != 0:
            print(f"⚠️ ОШИБКА: попытка открыть позицию при наличии открытой (position={self.position})")
            return
        
        self.position = direction
        self.entry_price = price
        self.entry_time = dt
        self.entry_reason = reason
        self.expected_profit = expected_profit
        self.expected_loss = expected_loss
        self.risk_reward = risk_reward
        
        # Выводим информацию о входе с ожиданиями
        print(f"🟢 ВХОД {('ЛОНГ' if direction == 1 else 'ШОРТ')}: {price:.2f} в {dt}")
        print(f"   {reason}")
        print(f"   Ожидаемая прибыль: {expected_profit:+.2f} | Ожидаемый убыток: {expected_loss:+.2f} | R/R: {risk_reward:.2f}")
    
    def _close_position(self, price, dt, reason):
        """Закрывает позицию"""
        if self.position == 0:
            return
        
        # Рассчитываем процент прибыли/убытка (без комиссий)
        if self.position == 1:
            pnl_pct_gross = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct_gross = (self.entry_price - price) / self.entry_price
        
        # Учитываем комиссии
        pnl_pct_net = pnl_pct_gross - self.taker_fee * 2  # Вход и выход
        
        # Применяем к части капитала (position_size)
        pnl = self.equity * pnl_pct_net * self.position_size
        self.equity += pnl
        
        trade = Trade(
            entry_time=self.entry_time,
            entry_price=self.entry_price,
            exit_time=dt,
            exit_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct_net * 100,  # Чистый % с учетом комиссий
            reason=reason,
            entry_reason=self.entry_reason,
            expected_profit=self.expected_profit,
            expected_loss=self.expected_loss,
            risk_reward=self.risk_reward
        )
        self.trades.append(trade)
        
        print(f"🔴 ВЫХОД: {price:.2f} в {dt} | PnL: {pnl:+.6f} ({pnl_pct_net*100:+.2f}%) | {reason}")
        
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_reason = ""
        self.expected_profit = 0.0
        self.expected_loss = 0.0
        self.risk_reward = 0.0


def run_backtest(filenames, candle_minutes=5, entry_threshold=1.0, exit_threshold=0.5, 
                 stop_loss_multiplier=0.5, position_size=1.0, trend_tf_minutes=15,
                 use_risk_management=False, min_risk_reward_ratio=1.5, use_adaptive_thresholds=True):
    """Запускает бэктест стратегии на одном или нескольких файлах"""
    
    # Если передан один файл (строка), оборачиваем в список
    if isinstance(filenames, str):
        filenames = [filenames]
    
    # Загружаем тики из всех файлов
    ticks = load_ticks_from_files(filenames)
    print(f"✅ Всего загружено: {len(ticks):,} тиков\n")
    
    strategy = MeanReversionStrategy(
        candle_minutes=candle_minutes,
        trend_tf_minutes=trend_tf_minutes,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_multiplier=stop_loss_multiplier,
        position_size=position_size,
        use_risk_management=use_risk_management,
        min_risk_reward_ratio=min_risk_reward_ratio,
        use_adaptive_thresholds=use_adaptive_thresholds
    )
    
    print(f"\n🚀 Запуск бэктеста (таймфрейм: {candle_minutes} минут)...")
    print(f"   📊 Тренд: таймфрейм {trend_tf_minutes}м, адаптивные пороги: {'✅ ВКЛ' if use_adaptive_thresholds else '❌ ВЫКЛ'}")
    if use_risk_management:
        print(f"   💰 Риск-менеджмент: ВКЛ (мин. R/R: {min_risk_reward_ratio:.1f})")
    print()
    for i, tick in enumerate(ticks):
        strategy.process_tick(tick)
        if (i + 1) % 50000 == 0:
            print(f"   Обработано {i + 1:,} / {len(ticks):,} тиков")
    
    # Закрываем последнюю позицию если есть
    if strategy.position != 0 and ticks:
        strategy._close_position(ticks[-1]['price'], ticks[-1]['datetime'], "Конец данных")
    
    # Результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Начальный капитал: {strategy.initial_equity:.6f}")
    print(f"   Итоговый капитал: {strategy.equity:.6f}")
    print(f"   Доходность: {(strategy.equity / strategy.initial_equity - 1) * 100:+.2f}%")
    print(f"   Всего сделок: {len(strategy.trades)}")
    
    if strategy.trades:
        wins = [t for t in strategy.trades if t.pnl > 0]
        losses = [t for t in strategy.trades if t.pnl < 0]
        print(f"   Прибыльных: {len(wins)} ({len(wins)/len(strategy.trades)*100:.1f}%)")
        print(f"   Убыточных: {len(losses)} ({len(losses)/len(strategy.trades)*100:.1f}%)")
        if wins:
            avg_win = np.mean([t.pnl for t in wins])
            print(f"   Средняя прибыль: {avg_win:.6f}")
        if losses:
            avg_loss = np.mean([t.pnl for t in losses])
            print(f"   Средний убыток: {avg_loss:.6f}")
        
        # Статистика по причинам выхода
        tp_count = len([t for t in strategy.trades if 'TP' in t.reason])
        sl_count = len([t for t in strategy.trades if 'SL' in t.reason])
        print(f"   Тейк-профитов: {tp_count}")
        print(f"   Стоп-лоссов: {sl_count}")
        
        # Статистика по режимам рынка (используем entry_reason, а не reason выхода)
        strong_trades = [t for t in strategy.trades if 'strong' in t.entry_reason.lower()]
        stable_trades = [t for t in strategy.trades if 'stable' in t.entry_reason.lower()]
        calm_trades = [t for t in strategy.trades if 'calm' in t.entry_reason.lower()]
        print(f"\n   По режимам рынка:")
        print(f"   - Сильный тренд: {len(strong_trades)} сделок")
        print(f"   - Устойчивый тренд: {len(stable_trades)} сделок")
        print(f"   - Не полный штиль: {len(calm_trades)} сделок")
        
        # Статистика по R/R
        if strategy.trades:
            avg_rr = np.mean([t.risk_reward for t in strategy.trades if t.risk_reward > 0])
            print(f"\n   Среднее R/R сделок: {avg_rr:.2f}")
        
        # Статистика риск-менеджмента
        if strategy.use_risk_management:
            print(f"\n   Риск-менеджмент:")
            print(f"   - Отклонено сделок (плохой R/R): {strategy.rejected_trades_rr}")
            print(f"   - Минимальный R/R: {strategy.min_risk_reward_ratio:.2f}")
    
    return strategy


def visualize_backtest(strategy, show_in_browser=True):
    """
    Визуализирует результаты бэктеста в Plotly
    
    Args:
        strategy: объект MeanReversionStrategy после выполнения бэктеста
        show_in_browser: если True, открывает график в браузере, иначе сохраняет в файл
    """
    if not strategy.candles:
        print("⚠️ Нет данных для визуализации")
        return
    
    print("\n📊 Создание визуализации...")
    
    # Подготовка данных свечей
    times = [c.start_time for c in strategy.candles]
    opens = [c.open for c in strategy.candles]
    highs = [c.high for c in strategy.candles]
    lows = [c.low for c in strategy.candles]
    closes = [c.close for c in strategy.candles]
    
    # Вычисляем Bollinger Bands для всех свечей
    bb_upper_list = []
    bb_middle_list = []
    bb_lower_list = []
    
    for i in range(len(closes)):
        if i >= strategy.bb_period - 1:
            window = closes[max(0, i - strategy.bb_period + 1):i + 1]
            bb_middle, bb_upper, bb_lower = compute_bollinger_bands(
                window, strategy.bb_period, strategy.bb_std
            )
            bb_upper_list.append(bb_upper)
            bb_middle_list.append(bb_middle)
            bb_lower_list.append(bb_lower)
        else:
            bb_upper_list.append(None)
            bb_middle_list.append(None)
            bb_lower_list.append(None)
    
    # Подготовка объёмов (разделённых на buy/sell)
    volumes = [c.volume for c in strategy.candles]
    buy_volumes = [c.buy_volume for c in strategy.candles]
    sell_volumes = [c.sell_volume for c in strategy.candles]
    
    # Вычисляем дисбаланс объёмов (Buy - Sell)
    volume_imbalance = [buy - sell for buy, sell in zip(buy_volumes, sell_volumes)]
    
    # Создаём subplots: 1) Цена + BB, 2) Тренд, 3) Объёмы, 4) Дисбаланс
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.15, 0.25, 0.2],
        subplot_titles=('Цена и сделки', 'Режим рынка', 'Объём (Buy/Sell)', 'Дисбаланс объёма')
    )
    
    # 1. График цены (свечи)
    fig.add_trace(
        go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name='Цена',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # 2. Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bb_upper_list,
            name='BB Upper',
            line=dict(color='rgba(100, 100, 250, 0.5)', width=1, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bb_middle_list,
            name='BB Middle',
            line=dict(color='rgba(100, 100, 250, 0.7)', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=times,
            y=bb_lower_list,
            name='BB Lower',
            line=dict(color='rgba(100, 100, 250, 0.5)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(100, 100, 250, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 3. Точки входа и выхода
    if strategy.trades:
        # Разделяем на лонги и шорты
        long_entries = [(t.entry_time, t.entry_price) for t in strategy.trades if t.entry_price < t.exit_price or 'лонг' in t.entry_reason.lower()]
        short_entries = [(t.entry_time, t.entry_price) for t in strategy.trades if t.entry_price > t.exit_price or 'шорт' in t.entry_reason.lower()]
        
        # Точки выхода с прибылью и убытком
        profit_exits = [(t.exit_time, t.exit_price) for t in strategy.trades if t.pnl > 0]
        loss_exits = [(t.exit_time, t.exit_price) for t in strategy.trades if t.pnl <= 0]
        
        # Лонг входы
        if long_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t[0] for t in long_entries],
                    y=[t[1] for t in long_entries],
                    mode='markers',
                    name='Вход LONG',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(color='darkgreen', width=1)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Шорт входы
        if short_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t[0] for t in short_entries],
                    y=[t[1] for t in short_entries],
                    mode='markers',
                    name='Вход SHORT',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Прибыльные выходы
        if profit_exits:
            fig.add_trace(
                go.Scatter(
                    x=[t[0] for t in profit_exits],
                    y=[t[1] for t in profit_exits],
                    mode='markers',
                    name='Выход TP',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='lime',
                        line=dict(color='darkgreen', width=1)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Убыточные выходы
        if loss_exits:
            fig.add_trace(
                go.Scatter(
                    x=[t[0] for t in loss_exits],
                    y=[t[1] for t in loss_exits],
                    mode='markers',
                    name='Выход SL',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='orange',
                        line=dict(color='darkred', width=1)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # 4. График режима рынка (тренд)
    if strategy.trend_history:
        # Создаём области для каждого режима
        trend_times = [t[0] for t in strategy.trend_history]
        trend_dirs = [t[1] for t in strategy.trend_history]
        trend_strengths = [t[2] for t in strategy.trend_history]
        
        # Преобразуем в числовые значения для визуализации
        # Направление: -1 (вниз), 0 (флет), +1 (вверх)
        # Сила: 0 (calm), 1 (stable), 2 (strong)
        strength_values = []
        colors = []
        
        for dir_val, strength in zip(trend_dirs, trend_strengths):
            if strength == TrendStrength.STRONG:
                strength_val = 2
                color = 'red' if dir_val != 0 else 'gray'
            elif strength == TrendStrength.STABLE:
                strength_val = 1
                color = 'orange' if dir_val != 0 else 'gray'
            else:  # CALM
                strength_val = 0
                color = 'green'
            
            strength_values.append(strength_val * dir_val if dir_val != 0 else strength_val)
            colors.append(color)
        
        # Добавляем линию с цветовой кодировкой
        for i in range(len(trend_times) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[trend_times[i], trend_times[i+1]],
                    y=[strength_values[i], strength_values[i]],
                    mode='lines',
                    line=dict(color=colors[i], width=3),
                    showlegend=False,
                    hovertemplate=f'Тренд: {trend_strengths[i].value}<br>Направление: {"▲" if trend_dirs[i]==1 else "▼" if trend_dirs[i]==-1 else "—"}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Добавляем легенду (отдельные маркеры)
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', 
                      marker=dict(size=10, color='red'),
                      name='Сильный тренд', showlegend=True),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='orange'),
                      name='Устойчивый', showlegend=True),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers',
                      marker=dict(size=10, color='green'),
                      name='Спокойный', showlegend=True),
            row=2, col=1
        )
    
    # 5. График объёмов (разделение на buy и sell)
    # Buy объёмы - положительные (зелёные)
    fig.add_trace(
        go.Bar(
            x=times,
            y=buy_volumes,
            name='Buy Volume',
            marker=dict(color='green', opacity=0.7),
            showlegend=True,
            hovertemplate='Buy: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Sell объёмы - отрицательные (красные)
    sell_volumes_negative = [-v for v in sell_volumes]
    fig.add_trace(
        go.Bar(
            x=times,
            y=sell_volumes_negative,
            name='Sell Volume',
            marker=dict(color='red', opacity=0.7),
            showlegend=True,
            hovertemplate='Sell: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. График дисбаланса объёмов (Buy - Sell)
    # Положительный дисбаланс = давление покупателей (зелёный)
    # Отрицательный дисбаланс = давление продавцов (красный)
    imbalance_colors = ['green' if v >= 0 else 'red' for v in volume_imbalance]
    
    fig.add_trace(
        go.Bar(
            x=times,
            y=volume_imbalance,
            name='Дисбаланс',
            marker=dict(color=imbalance_colors, opacity=0.7),
            showlegend=True,
            hovertemplate='Дисбаланс: %{y:.2f}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Добавляем нулевую линию для дисбаланса
    fig.add_hline(
        y=0,
        line=dict(color='gray', dash='dash', width=1),
        row=4, col=1
    )
    
    # Настройка layout
    fig.update_layout(
        title=f'Backtest Results | ROI: {(strategy.equity/strategy.initial_equity - 1)*100:+.2f}% | Trades: {len(strategy.trades)}',
        hovermode='x unified',
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hoversubplots="axis",
        spikedistance=-1
    )
    
    # Настройка осей
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(title_text='Время', row=4, col=1)
    
    fig.update_yaxes(title_text='Цена', row=1, col=1)
    fig.update_yaxes(title_text='Режим', row=2, col=1)
    fig.update_yaxes(title_text='Объём', row=3, col=1)
    fig.update_yaxes(title_text='Дисбаланс', row=4, col=1)
    
    # Привязываем оси X2, X3, X4 к X1 для синхронизации зума/прокрутки
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)
    fig.update_xaxes(matches='x', row=4, col=1)
    
    # Синхронизация вертикальной линии между графиками
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1
    )
    
    if show_in_browser:
        print("🌐 Открываю график в браузере...")
        fig.show()
    else:
        output_file = 'backtest_results.html'
        fig.write_html(output_file)
        print(f"💾 График сохранён в {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Тест стратегии Mean Reversion с режимами рынка',
        epilog="""
Примеры использования:
  # Один файл:
  %(prog)s data/BTCUSDT_2025-01-01.csv.gz
  
  # Несколько файлов:
  %(prog)s data/BTCUSDT_2025-01-01.csv.gz data/BTCUSDT_2025-01-02.csv.gz
  
  # Glob-паттерн:
  %(prog)s "data/BTCUSDT_2025-*.csv.gz"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('files', nargs='+', 
                       help='Путь к файлу(ам) с тиками (csv.gz). Можно указать несколько файлов или glob-паттерн')
    parser.add_argument('--candle-minutes', type=int, default=5,
                       help='Таймфрейм свечей в минутах (по умолчанию 5)')
    parser.add_argument('--trend-tf', type=int, default=5, choices=[15, 60],
                       help='Таймфрейм для определения тренда (мин): 15 или 60 (по умолчанию 15)')
    parser.add_argument('--entry-threshold', type=float, default=1.0,
                       help='Порог входа (доля от BB, 1.0 = точно на BB, >1.0 = за BB, по умолчанию 1.0)')
    parser.add_argument('--exit-threshold', type=float, default=0.5,
                       help='Порог выхода (доля от BB средней, по умолчанию 0.5)')
    parser.add_argument('--stop-loss-mult', type=float, default=0.5,
                       help='Множитель стоп-лосса (доля от ширины BB/2, по умолчанию 0.5)')
    parser.add_argument('--position-size', type=float, default=1.0,
                       help='Доля капитала на сделку (0.0-1.0, по умолчанию 1.0)')
    parser.add_argument('--use-risk-mgmt', action='store_true',
                       help='Включить риск-менеджмент (проверка R/R перед входом)')
    parser.add_argument('--min-rr', type=float, default=1.5,
                       help='Минимальное соотношение Risk/Reward (по умолчанию 1.5)')
    parser.add_argument('--adaptive-thresholds', action='store_true', default=True,
                       help='Использовать адаптивные пороги тренда (по умолчанию вкл)')
    parser.add_argument('--no-adaptive-thresholds', dest='adaptive_thresholds', action='store_false',
                       help='Отключить адаптивные пороги (использовать фиксированные)')
    parser.add_argument('--visualize', action='store_true',
                       help='Показать график результатов в браузере')
    parser.add_argument('--save-chart', type=str, metavar='FILE',
                       help='Сохранить график в HTML файл (вместо показа в браузере)')
    
    args = parser.parse_args()
    
    # Расширяем glob-паттерны и проверяем существование файлов
    files = expand_file_patterns(args.files)
    
    if not files:
        print(f"❌ Не найдено файлов, соответствующих паттерну(ам): {args.files}")
        sys.exit(1)
    
    # Проверяем существование всех файлов
    missing_files = [f for f in files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Файлы не найдены:")
        for f in missing_files:
            print(f"   - {f}")
        sys.exit(1)
    
    strategy = run_backtest(
        files, 
        candle_minutes=args.candle_minutes,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss_multiplier=args.stop_loss_mult,
        position_size=args.position_size,
        trend_tf_minutes=args.trend_tf,
        use_risk_management=args.use_risk_mgmt,
        min_risk_reward_ratio=args.min_rr,
        use_adaptive_thresholds=args.adaptive_thresholds
    )
    
    print("\n✅ Бэктест завершен")
    
    # Визуализация результатов
    if args.visualize or args.save_chart:
        if args.save_chart:
            # Сохраняем в файл
            visualize_backtest(strategy, show_in_browser=False)
            # Переименовываем файл если указано другое имя
            if args.save_chart != 'backtest_results.html':
                import shutil
                shutil.move('backtest_results.html', args.save_chart)
                print(f"💾 График сохранён в {args.save_chart}")
        else:
            # Показываем в браузере
            visualize_backtest(strategy, show_in_browser=True)


if __name__ == '__main__':
    main()

