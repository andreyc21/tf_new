#!/bin/bash

# Основной скрипт деплоя Bybit Trading Bot
set -e

# Цвета
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Показываем доступные команды
show_help() {
    echo "🚀 Bybit Trading Bot - Деплой"
    echo ""
    echo "Использование: ./deploy.sh [COMMAND]"
    echo ""
    echo "Команды:"
    echo "  docker         Запуск через Docker Compose (разработка)"
    echo "  build [TAG]    Сборка Docker образа"
    echo "  k8s            Деплой в Kubernetes (продакшен)"
    echo "  help           Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  ./deploy.sh docker          # Разработка"
    echo "  ./deploy.sh build v1.0.0    # Сборка образа"
    echo "  ./deploy.sh k8s             # Продакшен"
}

case "${1:-help}" in
    docker)
        info "🐳 Запуск через Docker Compose..."
        
        if [[ ! -f ".env" ]]; then
            if [[ -f "env.example" ]]; then
                info "📝 Создаем .env файл из примера..."
                cp env.example .env
                echo "⚠️  Отредактируйте .env файл с вашими API ключами!"
            else
                echo "❌ Файл env.example не найден!"
                exit 1
            fi
        fi
        
        docker-compose up -d
        success "✅ Бот запущен!"
        info "📋 Полезные команды:"
        info "   docker-compose logs -f bybit-bot  # Логи"
        info "   curl http://localhost:8080/health # Health check"
        info "   open http://localhost:8081        # Веб-интерфейс логов"
        ;;
        
    build)
        info "🔨 Сборка Docker образа..."
        ./deploy/build.sh ${2:-latest}
        ;;
        
    k8s)
        info "☸️  Деплой в Kubernetes..."
        ./deploy/deploy-k8s.sh
        ;;
        
    help|--help|-h)
        show_help
        ;;
        
    *)
        echo "❌ Неизвестная команда: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
