#!/bin/bash

# Скрипт деплоя Bybit Trading Bot в Kubernetes
set -e

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Конфигурация
NAMESPACE="trading"
APP_NAME="bybit-bot"

info "🚀 Деплой Bybit Trading Bot в Kubernetes"

# Проверяем подключение к кластеру
if ! kubectl cluster-info &>/dev/null; then
    error "❌ Нет подключения к Kubernetes кластеру!"
    exit 1
fi

info "✅ Подключение к кластеру установлено"

# Переходим в корневую директорию проекта
cd "$(dirname "$0")/.."

# Создаем namespace
info "📦 Создаем namespace '$NAMESPACE'..."
kubectl apply -f k8s/namespace.yaml

# Проверяем секреты
info "🔐 Проверяем секреты..."
if kubectl get secret bybit-bot-secrets -n $NAMESPACE &>/dev/null; then
    warning "⚠️  Секреты уже существуют. Обновляем..."
else
    info "📝 Создаем новые секреты..."
fi

# Предупреждение о секретах
if grep -q "your_api_key_here" k8s/secret.yaml; then
    warning "⚠️  В файле k8s/secret.yaml остались значения по умолчанию!"
    warning "⚠️  Отредактируйте файл перед деплоем или используйте kubectl create secret"
    
    read -p "Продолжить деплой? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "❌ Деплой отменен"
        exit 1
    fi
fi

# Применяем конфигурацию
info "🔧 Применяем ConfigMap..."
kubectl apply -f k8s/configmap.yaml

info "🔐 Применяем секреты..."
kubectl apply -f k8s/secret.yaml

info "💾 Создаем PersistentVolumeClaims..."
kubectl apply -f k8s/pvc.yaml

info "🚀 Деплоим приложение..."
kubectl apply -f k8s/deployment.yaml

info "🌐 Создаем сервисы..."
kubectl apply -f k8s/service.yaml

# Ждем готовности деплоя
info "⏳ Ожидаем готовности деплоя..."
kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=300s

# Проверяем статус
info "📊 Статус деплоя:"
kubectl get pods -n $NAMESPACE -l app=$APP_NAME

# Показываем логи
info "📋 Последние логи:"
kubectl logs -n $NAMESPACE -l app=$APP_NAME --tail=10

success "🎉 Деплой завершен успешно!"

info "📋 Полезные команды:"
info "   kubectl get pods -n $NAMESPACE"
info "   kubectl logs -f -n $NAMESPACE -l app=$APP_NAME"
info "   kubectl exec -it -n $NAMESPACE deployment/$APP_NAME -- /bin/bash"
info "   kubectl port-forward -n $NAMESPACE svc/bybit-bot-service 8080:8080"

# Health check
info "🏥 Проверяем health endpoint..."
if kubectl port-forward -n $NAMESPACE svc/bybit-bot-service 8080:8080 &>/dev/null &
    PF_PID=$!
    sleep 3
    
    if curl -s http://localhost:8080/health | jq . &>/dev/null; then
        success "✅ Health check прошел успешно!"
    else
        warning "⚠️  Health check недоступен (это нормально если бот еще запускается)"
    fi
    
    kill $PF_PID 2>/dev/null || true
fi
