# 🤖 GitHub Actions для Bybit Trading Bot

Автоматизированная сборка, тестирование и публикация Docker образов.

## 📋 Workflows

### 1. 🔧 CI Pipeline (`ci.yml`)
**Триггеры:** Push в main/develop, Pull Requests

**Что делает:**
- ✅ Проверка синтаксиса Python (flake8)
- ✅ Проверка форматирования кода (black)
- ✅ Тестирование сборки Docker образа
- ✅ Проверка импорта TA-Lib и стратегий

### 2. 🐳 Docker Build (`docker-build.yml`)
**Триггеры:** Push в main/develop, теги v*, Pull Requests

**Что делает:**
- 🏗️ Сборка Docker образа для Linux (amd64, arm64)
- 📦 Публикация в GitHub Container Registry (ghcr.io)
- 🏷️ Автоматическое тегирование:
  - `latest` (из main ветки)
  - `develop` (из develop ветки)
  - `v1.0.0`, `1.0`, `1` (из тегов)
- 💾 Кэширование слоев для ускорения сборки

### 3. 🚀 Release (`release.yml`)
**Триггеры:** Push тегов вида `v*.*.*`

**Что делает:**
- 📝 Генерация changelog из коммитов
- 🎉 Создание GitHub Release
- 📦 Автоматическая публикация образа
- 📋 Инструкции по деплою

## 🏷️ Стратегия тегирования

```bash
# Релиз версии
git tag v1.2.3
git push origin v1.2.3

# Создаст образы:
# - ghcr.io/username/repo:v1.2.3
# - ghcr.io/username/repo:1.2
# - ghcr.io/username/repo:1
# - ghcr.io/username/repo:latest (если main ветка)
```

## 🔐 Настройка

1. **Personal Access Token:**
   - Права: `write:packages`, `read:packages`, `repo`
   - Используется автоматически через `GITHUB_TOKEN`

2. **Приватность пакетов:**
   - По умолчанию: приватные
   - Можно сделать публичными в настройках пакета

3. **Multi-platform сборка:**
   - Linux AMD64 (x86_64)
   - Linux ARM64 (Apple Silicon, Raspberry Pi)

## 📊 Мониторинг

- **Actions**: `https://github.com/USERNAME/REPO/actions`
- **Packages**: `https://github.com/USERNAME/REPO/pkgs/container/REPO`
- **Releases**: `https://github.com/USERNAME/REPO/releases`

## 🔄 Workflow статусы

| Workflow | Badge |
|----------|-------|
| CI | ![CI](https://github.com/USERNAME/REPO/workflows/CI/badge.svg) |
| Docker Build | ![Docker](https://github.com/USERNAME/REPO/workflows/Build%20and%20Push%20Docker%20Image/badge.svg) |
| Release | ![Release](https://github.com/USERNAME/REPO/workflows/Release/badge.svg) |

## 🛠️ Локальная разработка

```bash
# Тестирование CI локально (с act)
act -j lint
act -j test-docker

# Ручная сборка и публикация
./deploy.sh build v1.0.0-dev
```

## 🚀 Деплой

После успешной сборки образ автоматически доступен:

```bash
# Kubernetes
kubectl set image deployment/bybit-bot \
  bybit-bot=ghcr.io/username/repo:v1.0.0 \
  -n trading

# Docker Compose
docker-compose pull && docker-compose up -d
```

