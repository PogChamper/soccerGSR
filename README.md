# SoccerGSR ML Service

ML-сервис для анализа футбольных трансляций с использованием компьютерного зрения.

## Функционал

- **POST /forward** - Инференс видео через YOLO детектор (детектирование игроков, вратарей, судей, мяча)
- **GET /history** - История всех запросов из базы данных
- **DELETE /history** - Удаление истории (требует X-Admin-Token)
- **GET /stats** - Статистика запросов (время обработки, характеристики входных данных)

### Доп. функции:
- JWT авторизация (/auth/register, /auth/login, /auth/me)
- Миграции базы данных через Alembic

## Вклад участников
**OlegNotHehe** (Олег Рокин) - реализация основного inference скрипта

**PogChamper** (Олег Байшев) - остальная часть работ по сервису

## Установка и запуск

### 1. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка окружения (опционально)

```bash
cp env.example .env
# Отредактируйте .env при необходимости
```

### 4. Применение миграций

```bash
alembic upgrade head
```

### 5. Запуск сервера

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Модель скачается автоматически при первом запуске!**

Или:

```bash
python -m app.main
```

## Модель

Модель YOLOv5lu (ONNX) **скачивается автоматически** с Google Drive при первом запуске сервера.

- **Размер:** ~200 MB
- **Расположение:** `models/best.onnx`
- **Google Drive ID:** `1OA6l1GEb6ki5Dq2zSmJgH4AcEkbYvisq`

### Ручное скачивание (если нужно)

```bash
# Через CLI утилиту
python -m app.utils.model_loader

# Или с указанием пути
python -m app.utils.model_loader --output ./models/best.onnx
```

### Отключение авто-скачивания

```bash
# В .env файле
MODEL_AUTO_DOWNLOAD=false
MODEL_PATH=/path/to/your/model.onnx
```

## API Документация

После запуска доступна по адресам:
- Swagger UI: http://localhost:8000/docs

## Использование

### POST /forward

Отправка видео на обработку (сохранение в файл):

```bash
curl -X POST "http://localhost:8000/forward" \
  -F "image=@video.mp4" \
  -H "X-Return-Format: stream" \
  -o output.mp4
```

Для получения JSON с base64-encoded видео:
```bash
curl -X POST "http://localhost:8000/forward" -F "image=@video.mp4"
```

Ответ в формате JSON:
```json
{
  "status": "success",
  "video": "<base64-encoded-mp4>",
  "metadata": {
    "filename": "video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "processing_time": 12.5,
    "total_detections": 1500
  }
}
```

### GET /history

```bash
curl "http://localhost:8000/history?limit=10&offset=0"
```

### DELETE /history

```bash
curl -X DELETE "http://localhost:8000/history" \
  -H "X-Admin-Token: admin-delete-token-change-in-production"
```

### GET /stats

```bash
curl "http://localhost:8000/stats"
```

## Авторизация

### Регистрация

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123"}'
```

### Логин

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -d "username=user1&password=password123"
```

## Классы детектирования

| ID | Класс | Цвет |
|----|-------|------|
| 0 | player | 🟢 Зеленый |
| 1 | goalkeeper | 🟡 Желтый |
| 2 | referee | 🔴 Красный |
| 3 | ball | 🟠 Оранжевый |

## Структура проекта

```
soccer-app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Настройки
│   ├── api/
│   │   ├── forward.py       # POST /forward
│   │   ├── history.py       # GET/DELETE /history
│   │   ├── stats.py         # GET /stats
│   │   └── auth.py          # JWT авторизация
│   ├── models/
│   │   ├── database.py      # SQLAlchemy модели
│   │   └── schemas.py       # Pydantic схемы
│   ├── services/
│   │   ├── detector.py      # ONNX детектор
│   │   ├── video_processor.py
│   │   └── history_service.py
│   └── utils/
│       ├── visualizer.py    # Отрисовка bbox
│       └── model_loader.py  # Загрузка модели с GDrive
├── models/                  # Создаётся автоматически
│   └── best.onnx           # Скачивается при запуске
├── alembic/
│   ├── env.py
│   └── versions/
├── alembic.ini
├── requirements.txt
└── README.md
```

## Команда

- **PogChamper** (Олег Байшев)
- **OlegNotHehe** (Олег Рокин)
- **FoshchanVArvar** (Виктор Фощан) - не участвовал в чекпоинтах, начиная с обучения бейзлйн моделей

## Куратор

**Mark Blumenau** (Марк Блуменау)
