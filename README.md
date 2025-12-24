# soccerGSR: Game State Reconstruction in Soccer

Реконструкция игровых состояний футбольного матча

![Status](https://img.shields.io/badge/status-in%20progress-green)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

Проект команды студентов 1-го курса магистратуры "Искусственный интеллект" ВШЭ (ФКН). Цель проекта — разработка системы для анализа футбольных трансляций с помощью компьютерного зрения.

---

## Цель проекта

Основной целью проекта является создание и запуск end-to-end системы, которая принимает на вход отрывок футбольного матча и выполняет на нем **детектирование и трекинг игроков**. Финальное решение упаковано в интерактивный веб-сервис для наглядной демонстрации работы моделей.

---

## ML Service (FastAPI)

ML-сервис для анализа футбольных трансляций с использованием компьютерного зрения.

### Функционал

- **POST /forward** - Инференс видео через YOLO детектор (детектирование игроков, вратарей, судей, мяча)
- **GET /history** - История всех запросов из базы данных
- **DELETE /history** - Удаление истории (требует X-Admin-Token)
- **GET /stats** - Статистика запросов (время обработки, характеристики входных данных)

#### Доп. функции:
- JWT авторизация (/auth/register, /auth/login, /auth/me)
- Миграции базы данных через Alembic

### Вклад участников
- **OlegNotHehe** (Олег Рокин) - реализация основного inference скрипта
- **PogChamper** (Олег Байшев) - остальная часть работ по сервису

---

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

---

## Модель

Модель YOLOv5lu (ONNX) **скачивается автоматически** с Google Drive при первом запуске сервера.

- **Размер:** ~200 MB
- **Расположение:** `models/best.onnx`
- **Google Drive ID:** `1OA6l1GEb6ki5Dq2zSmJgH4AcEkbYvisq`

### Ручное скачивание (если нужно)

```bash
python -m app.utils.model_loader
```

### Отключение авто-скачивания

```bash
# В .env файле
MODEL_AUTO_DOWNLOAD=false
MODEL_PATH=/path/to/your/model.onnx
```

---

## API Документация

После запуска доступна по адресам:
- Swagger UI: http://localhost:8000/docs

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

---

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

---

## Классы детектирования

| ID | Класс | Цвет |
|----|-------|------|
| 0 | player | 🟢 Зеленый |
| 1 | goalkeeper | 🟡 Желтый |
| 2 | referee | 🔴 Красный |
| 3 | ball | 🟠 Оранжевый |

---

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

---

## Технологический стек

* **Язык:** Python 3.9+
* **Основные библиотеки:** PyTorch, OpenCV, NumPy, FastAPI
* **Модели:** YOLOv5lu (детектирование)
* **Сервис:** FastAPI + Uvicorn
* **База данных:** SQLite + SQLAlchemy + Alembic
* **Инструменты:** Git, ONNX Runtime

---

## План работы на год

| Чекпойнт | Дедлайн | Цель этапа | Ключевые задачи |
| :--- | :--- | :--- | :--- |
| **1. Установочный** | Конец сентября 2025 | Формализация проекта | - Создание репозитория<br>- Финализация темы и плана<br>- Оформление `README.md` |
| **2. EDA** | 20-е числа октября 2025 | Исследование данных | - Поиск и анализ датасетов (SoccerNet, SportsMOT)<br>- Написание скриптов для EDA<br>- Выбор основного датасета для работы |
| **3-4. Бейзлайн**| Конец ноября 2025 -<br>Середина января 2026 | Создание первого рабочего прототипа | - Обучение baseline-детектора (YOLOv8)<br>- Реализация простого трекера (Kalman Filter)<br>- Сборка пайплайна `Detection + Tracking` |
| **5. Сервис** | Середина февраля 2026 | Упаковка решения в демо | - Разработка UI на Streamlit/Gradio<br>- Интеграция бейзлайн-моделей в веб-сервис<br>- Демонстрация работы на тестовых видео |
| **6-7. Улучшение DL части**| Конец марта -<br>Середина мая 2026 | Повышение качества трекинга | - Изучение и внедрение SOTA-трекера (BoT-SORT)<br>- Обучение/адаптация Re-ID модели<br>- Сравнение метрик с бейзлайном |
| **Защита** | Июнь 2026 | Финализация и презентация проекта | - Подготовка итогового отчета<br>- Создание презентации<br>- Демонстрация лучшей версии сервиса |

---

## Команда

* **PogChamper** (Олег Байшев) - Researcher
* **OlegNotHehe** (Олег Рокин) - Researcher
* **FoshchanVArvar** (Виктор Фощан) - Researcher (не участвовал в чекпоинтах, начиная с обучения бейзлайн моделей)

## 👨Куратор

* **Mark Blumenau** (Марк Блуменау)
