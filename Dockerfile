# Простой однослойный образ с pandas-ta
FROM python:3.13-slim

# Устанавливаем только необходимые системные пакеты
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Создаем пользователя для безопасности
RUN groupadd -r trading && useradd -r -g trading -s /bin/false trading

# Создаем рабочую директорию
WORKDIR /app

# Копируем код приложения
COPY --chown=trading:trading . .

# Убеждаемся, что config.py доступен
COPY --chown=trading:trading config.py /app/config.py

# Создаем директории для логов и данных
RUN mkdir -p /app/logs /app/data && \
    chown -R trading:trading /app

# Переключаемся на пользователя trading
USER trading

# Настраиваем переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEVELOPMENT=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Expose порт для health check
EXPOSE 8080

# Команда запуска
CMD ["python", "bybit_bot.py"]
