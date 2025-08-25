#!/bin/bash

# Скрипт сборки Docker образа для Bybit Trading Bot
set -e

# Конфигурация
REGISTRY="ghcr.io"
REPO_OWNER=${GITHUB_REPOSITORY_OWNER:-$(git config --get remote.origin.url | sed -n 's#.*/\([^/]*\)/.*#\1#p')}
REPO_NAME=${GITHUB_REPOSITORY##*/}
if [ -z "$REPO_NAME" ]; then
    REPO_NAME=$(basename "$(git rev-parse --show-toplevel)" 2>/dev/null || echo "bybit-bot")
fi
IMAGE_NAME="$REPO_OWNER/$REPO_NAME"
VERSION=${1:-$(date +%Y%m%d-%H%M%S)}
FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$VERSION"
LATEST_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:latest"

echo "🐳 Сборка Docker образа: $FULL_IMAGE_NAME"

# Проверяем наличие Dockerfile
if [[ ! -f "Dockerfile" ]]; then
    echo "❌ Dockerfile не найден!"
    exit 1
fi

# Проверяем наличие requirements.txt
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ requirements.txt не найден!"
    exit 1
fi

# Сборка образа
echo "🔨 Сборка образа..."
docker build \
    --tag "$FULL_IMAGE_NAME" \
    --tag "$LATEST_IMAGE_NAME" \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
    --build-arg VERSION="$VERSION" \
    .

echo "✅ Образ собран успешно!"

# Показываем размер образа
echo "📊 Размер образа:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

# Проверяем аутентификацию в GitHub Registry
echo "🔐 Проверяем аутентификацию в GitHub Container Registry..."
if ! docker info | grep -q "ghcr.io"; then
    echo "⚠️  Не авторизованы в ghcr.io"
    echo "💡 Для аутентификации выполните:"
    echo "   echo \$GITHUB_TOKEN | docker login ghcr.io -u \$GITHUB_USERNAME --password-stdin"
    echo "   или"
    echo "   docker login ghcr.io"
fi

# Опционально: пушим в registry
read -p "🚀 Загрузить образ в GitHub Container Registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Загружаем образ в GitHub Container Registry..."
    
    # Пробуем загрузить
    if docker push "$FULL_IMAGE_NAME" && docker push "$LATEST_IMAGE_NAME"; then
        echo "✅ Образ загружен в registry!"
        
        echo ""
        echo "📋 Для обновления Kubernetes деплоя выполните:"
        echo "kubectl set image deployment/bybit-bot bybit-bot=$FULL_IMAGE_NAME -n trading"
        echo ""
        echo "🔗 Образ доступен по адресу:"
        echo "   $FULL_IMAGE_NAME"
        echo "   $LATEST_IMAGE_NAME"
    else
        echo "❌ Ошибка загрузки образа!"
        echo "💡 Убедитесь, что вы авторизованы в GitHub Container Registry"
    fi
fi

echo "🎉 Готово! Образ: $FULL_IMAGE_NAME"
