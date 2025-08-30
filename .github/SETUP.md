# 🚀 GitHub Container Registry Setup

Инструкция по настройке автоматической сборки и публикации Docker образов в GitHub Container Registry.

## 📋 Что нужно сделать

### 1. 🔧 Включить GitHub Container Registry

1. Перейдите в **Settings** → **Developer settings** → **Personal access tokens**
2. Создайте новый **Classic token** с правами:
   - `write:packages` (для публикации образов)
   - `read:packages` (для скачивания образов)
   - `repo` (для доступа к репозиторию)

### 2. 🐳 Локальная аутентификация

```bash
# Сохраните токен в переменную
export GITHUB_TOKEN=your_personal_access_token
export GITHUB_USERNAME=your_github_username

# Войдите в GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

### 3. 📝 Настройка репозитория

1. **Обновите image в deployment.yaml:**
   ```yaml
   image: ghcr.io/YOUR_USERNAME/REPO_NAME:latest
   ```

2. **Сделайте пакет публичным** (опционально):
   - Перейдите в **Packages** на странице репозитория
   - Выберите пакет → **Package settings**
   - **Change visibility** → **Public**

## 🚀 Автоматическая сборка

GitHub Actions автоматически собирает и публикует образы при:

### 📦 Push в ветки:
- `main` → `ghcr.io/username/repo:latest`
- `develop` → `ghcr.io/username/repo:develop`

### 🏷️ Создание тегов:
```bash
git tag v1.0.0
git push origin v1.0.0
```
Создаст образы:
- `ghcr.io/username/repo:v1.0.0`
- `ghcr.io/username/repo:1.0`
- `ghcr.io/username/repo:1`

### 🔀 Pull Requests:
- Создает временный образ для тестирования

## 📊 Мониторинг сборки

1. **GitHub Actions**: `.github/workflows/docker-build.yml`
2. **Packages**: `https://github.com/USERNAME/REPO/pkgs/container/REPO`
3. **Releases**: автоматически создаются для тегов

## 🛠️ Локальная сборка

```bash
# Сборка и публикация
./deploy.sh build v1.0.0

# Только сборка
docker build -t bybit-bot:local .
```

## ☸️ Деплой в Kubernetes

```bash
# Автоматический деплой (использует latest)
./deploy.sh k8s

# Деплой конкретной версии
kubectl set image deployment/bybit-bot \
  bybit-bot=ghcr.io/username/repo:v1.0.0 \
  -n trading
```

## 🔒 Приватные образы

Если пакет приватный, создайте Secret в Kubernetes:

```bash
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN \
  --namespace=trading
```

И добавьте в deployment.yaml:
```yaml
spec:
  template:
    spec:
      imagePullSecrets:
      - name: ghcr-secret
```

## 🐛 Troubleshooting

### ❌ "unauthorized: unauthenticated"
- Проверьте права токена (`write:packages`)
- Убедитесь, что залогинены: `docker login ghcr.io`

### ❌ "package does not exist"
- Первый раз пуш должен быть из GitHub Actions
- Или создайте пакет вручную через веб-интерфейс

### ❌ Kubernetes не может скачать образ
- Проверьте, что пакет публичный
- Или настройте `imagePullSecrets`

## 📈 Мониторинг

- **GitHub Actions**: история сборок
- **Packages**: статистика скачиваний  
- **Container Registry**: все версии образов
- **Releases**: changelog и метаданные

