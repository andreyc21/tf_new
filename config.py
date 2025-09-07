# Конфигурация торговой стратегии

# 🏆 ОПТИМИЗИРОВАННАЯ СТРАТЕГИЯ
# Основано на результатах бэктестов за 2 месяца:
# - Custom RSI: +8.79% доходность, 507 сделок, Sharpe 0.55
# - TA-Lib RSI: -1.52% доходность, 181 сделка, Sharpe -5.48M

# RSI настройки
USE_CUSTOM_RSI = True    # 🏆 True = наша выигрышная SMA-based реализация
                         #    False = стандартная TA-Lib Wilder's реализация

USE_DUAL_RSI = False     # True = использовать оба RSI для сравнения сигналов
                         # False = использовать только один тип RSI

# Торговые параметры
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 30   # Покупка при RSI < 30
RSI_SELL_THRESHOLD = 70  # Продажа при RSI > 70

# Bollinger Bands (всегда через TA-Lib - быстрее, результат тот же)
BB_PERIOD = 20
BB_STD_DEV = 2.0

# Временные рамки
CANDLE_MINUTES = 5

# Размер позиции
POSITION_SIZE = 0.01

# Торговая пара
SYMBOL = 'BTCUSDT'

# Настройки переподключения
RECONNECT_DELAY = 30
MAX_RECONNECT_ATTEMPTS = 10

# 🧠 Нейронная сеть (AI-фильтр)
USE_NEURAL_FILTER = True         # True = использовать нейронный фильтр для сигналов
NEURAL_CONFIDENCE_THRESHOLD = 0.51  # Оптимальная уверенность для входа

# Пути к свежей обученной модели (2025)
NEURAL_MODEL_PATH = "models/neural_filter_2025.keras"
NEURAL_SCALER_PATH = "models/scaler_2025.joblib"
