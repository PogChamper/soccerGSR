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

## ML Service (FastAPI, async, GSR)

End-to-end Game State Reconstruction для футбольных трансляций. Каждый ML-этап
работает на GPU через `onnxruntime-gpu`. Один GPU — один воркер за раз.

### Полный пайплайн на клип

| Stage | Что делает | Модель |
|-------|------------|--------|
| Pass1 | per-frame: detection → DINOv3 batch-embed → tracking (motion + ReID) → jersey gate+OCR → team embedding sample → field keypoints/lines | YOLOv5lu, DINOv3 ViT-S+/16, BoT-SORT (vendored, motion+IoU+CMC+ReID), ShuffleNetV2 (visibility), ConvNeXt-Tiny (OCR), HRNet kp/lines |
| Aggregate | per-track: jersey number from logits-mean, team via 2-component GMM on per-track mean DINOv3 embeddings (outfield outliers → referee), independent GK GMM; per-clip: PnLCalib homography per frame, foot-point projection to pitch coords | scikit-learn GaussianMixture, PnLCalib FramebyFrameCalib |
| Pass2 | render mp4: bbox + #track_id + J<jersey> + цвет команды + мини-карта поля с проекциями | OpenCV |

### REST API (async, основной)

- **POST /forward** — принимает mp4, возвращает `202 Accepted` + `job_id`. Воркер обрабатывает в фоне.
- **GET /jobs/{job_id}** — статус (`queued|running|done|error`), стадия (`pass1|aggregate|pass2`), progress %.
- **GET /jobs/{job_id}/video** — готовый аннотированный mp4 (404 пока не done).
- **GET /jobs/{job_id}/gsr.json** — структура ClipState: `meta`, `frames` (homography per frame), `observations` (bbox + track_id + visibility_p + team_id + pitch_xy), `tracks` (cls_name, jersey_number, team_label, ...).
- **GET /jobs?limit=&offset=&status=** — список последних задач.
- **POST /forward/sync** — legacy: внутри ставит async-задачу и блокируется до её завершения. Совместимо со старым API.

### Доп. эндпоинты

- **GET /history**, **DELETE /history** (JWT админа), **GET /stats**
- JWT авторизация: `/auth/register`, `/auth/login`, `/auth/me`
- Alembic миграции для `users`, `request_history`, `jobs`

### Вклад участников
- **OlegNotHehe** (Олег Рокин) - реализация основного inference скрипта
- **PogChamper** (Олег Байшев) - остальная часть работ по сервису

---

## Установка и запуск

### Системные требования

- Linux / WSL2 (Ubuntu 24.04 проверено)
- NVIDIA GPU + driver с CUDA 12.x compat (тестировано на RTX 4070 Ti SUPER 16GB)
- Python 3.12 (3.11 тоже подходит)

### 1. venv

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Зависимости

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

`requirements.txt` поднимает onnxruntime-gpu 1.20.1 (CUDA 12), transformers 4.57
(только для офлайн-экспорта DINOv3), torch 2.5.1 + torchvision 0.20.1 (нужны
для heatmap-decode в PnLCalib и для ONNX-экспортов), scikit-learn (GMM/KMeans),
shapely, lap, loguru. **boxmot не устанавливается** — вместо него используется
уменьшенный, torch-free форк под `app/vendor/boxmot/` (см.
`app/vendor/boxmot/NOTICE.md`, лицензия AGPL-3.0).

### 3. WSL2 nuance

`app/utils/cuda_env.bootstrap()` сам добавляет `/usr/lib/wsl/lib` в
`LD_LIBRARY_PATH` и зовёт `ort.preload_dlls(cuda=True, cudnn=True)` ДО
создания первой сессии. Без этого `onnxruntime-gpu >= 1.19` падает с
`CUDA failure 100`. Бутстрэп идемпотентен и вызывается из `lifespan`.

### 4. Миграции БД

```bash
alembic upgrade head
```

### 5. Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

При старте автоматически:
1. Бутстрапит CUDA для WSL.
2. Через `app/utils/models_registry` подтягивает все модели с Google Drive
   (gdown) и проверяет их sha256.
3. Создаёт ORT-сессии **до** любого `import torch` (важно: torch инициализирует
   свой cudnn frontend, после которого ORT не может создать новую CUDA-сессию
   на cudnn 9.1).
4. Стартует фоновый воркер обработки задач.

---

## Модели

Все DL-модели — **ONNX**. Реестр и автозагрузка — `app/utils/models_registry.py`:
каждая модель скачивается с Google Drive (gdown) при отсутствии и
верифицируется по sha256. Скрипты экспорта нужны только для пере-экспорта
весов из исходных чекпойнтов.

| Модель | Файл | Размер | Пере-экспорт |
|--------|------|--------|--------------|
| DEIMv2 детектор (основной) | `models/deimv2_m_896.onnx` | 76 MB | DEIMv2 `export_l_model.py` из `best_stg2.pth` |
| YOLO детектор (legacy) | `models/best.onnx` | 213 MB | — |
| Visibility gate | `models/visibility_gate.onnx` | ~5 MB | jersey-visibility-project (ShuffleNetV2) |
| Jersey OCR | `models/jersey_ocr.onnx` (+ `.data`) | ~110+110 MB | jersey-ocr-project (ConvNeXt-Tiny) |
| HRNet keypoints | `models/hrnet_kp.onnx` | 264 MB | `python scripts/export_hrnet_onnx.py kp` |
| HRNet lines | `models/hrnet_lines.onnx` | 264 MB | `python scripts/export_hrnet_onnx.py lines` |
| DINOv3 embedder | `models/dinov3_vits16plus.onnx` | 115 MB | `python scripts/export_dinov3_onnx.py` (gated на HF, см. ниже) |

### Экспорт PnLCalib HRNet → ONNX

```bash
python scripts/export_hrnet_onnx.py both
```

Скрипт сам:
1. Скачивает `SV_kp` и `SV_lines` (~265 MB каждый) с
   `github.com/mguti97/PnLCalib/releases/v1.0.0`.
2. Загружает через vendor-копию `app/vendor/pnlcalib/model/cls_hrnet*.py`.
3. Экспортирует opset 17, фиксированный input `(1, 3, 540, 960)`,
   `dynamic_axes` по batch.

### Экспорт DINOv3 → ONNX

```bash
python scripts/export_dinov3_onnx.py
```

Скачивает `facebook/dinov3-vits16plus-pretrain-lvd1689m` (28.7M params,
embedding 384, gated — нужно принять [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
и быть залогиненным `huggingface-cli login`), экспортирует только
`pooler_output` через `torch.onnx.export` opset 17, dynamic batch axis,
input fixed at `(B, 3, 224, 224)`. Размер ONNX ~110 MB, латентность на
RTX 4070 Ti SUPER:
- batch 1 → ~6 ms
- batch 8 → ~10 ms (1.3 ms/img)
- batch 22 → ~20 ms (~0.9 ms/img) — типичный кадр (~22 игрока)

DINOv3-эмбеддинг в одном проходе питает обоих потребителей: BoT-SORT
получает их через `update(..., embeddings=...)` для ReID-aware ассоциации,
а `TeamClassifier` — для GMM-кластеризации команд.

---

## API Документация

Swagger: http://localhost:8000/docs

### Async (рекомендуется): POST /forward

```bash
# 1. submit
JOB=$(curl -s -X POST "http://localhost:8000/forward" \
  -F "image=@123.mp4" | jq -r '.job_id')

# 2. poll
watch -n 1 "curl -s http://localhost:8000/jobs/$JOB | jq"

# 3. download
curl -s "http://localhost:8000/jobs/$JOB/video"     -o gsr.mp4
curl -s "http://localhost:8000/jobs/$JOB/gsr.json"  -o gsr.json
```

`gsr.json` содержит:
```json
{
  "meta": {"filename":"...","width":1920,"height":1080,"fps":30,"frame_count":673,...},
  "frames": [{"frame_idx":0,"H_world2img":[[...]],"H_img2world":[[...]],"cam_params":{...}}, ...],
  "observations": [
    {"frame_idx":0,"track_id":2,"cls_id":0,"bbox_xyxy":[...],"team_id":1,"pitch_xy":[12.3,4.5], ...},
    ...
  ],
  "tracks": {
    "2": {"track_id":2,"cls_name":"player","team_label":"team_b","jersey_number":10,"jersey_confidence":0.99, ...},
    ...
  }
}
```

### Legacy sync: POST /forward/sync

Совместимо со старым API: блокируется до конца, возвращает либо base64 mp4
(JSON), либо stream:

```bash
curl -X POST "http://localhost:8000/forward/sync" \
  -F "image=@video.mp4" \
  -H "X-Return-Format: stream" -o output.mp4
```

### Прочее

```bash
curl "http://localhost:8000/jobs?limit=10&status=done"
curl "http://localhost:8000/history?limit=10&offset=0"
curl -X DELETE "http://localhost:8000/history" -H "Authorization: Bearer <admin-jwt>"
curl "http://localhost:8000/stats"
curl "http://localhost:8000/health"
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

### Создание админа

Регистрация через API никогда не даёт права админа. Создать или повысить
пользователя до админа (нужен для `DELETE /history`):

```bash
PYTHONPATH=. python scripts/create_admin.py admin <password>
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
│   ├── main.py                          # FastAPI lifespan: cuda → models → ORT → torch → worker
│   ├── config.py
│   ├── api/
│   │   ├── jobs.py                      # POST /forward (async), GET /jobs/{id}/...
│   │   ├── forward.py                   # POST /forward/sync (legacy wrapper)
│   │   ├── history.py
│   │   ├── stats.py
│   │   └── auth.py
│   ├── models/
│   │   ├── database.py                  # users, request_history, jobs
│   │   └── schemas.py
│   ├── services/
│   │   ├── detector.py                  # YOLO ONNX
│   │   ├── tracker.py                   # wrapper над app/vendor/boxmot (BoT-SORT, без torch)
│   │   ├── jersey.py                    # visibility gate ONNX + ConvNeXt OCR ONNX + per-track aggregation
│   │   ├── embedder.py                  # DINOv3 ViT-S+/16 ONNX (общий ReID + team)
│   │   ├── team_classifier.py           # GMM на DINO эмбеддингах + outlier→referee
│   │   ├── keypoints.py                 # PnLCalib HRNet kp+lines via ORT
│   │   ├── calibration.py               # FramebyFrameCalib wrapper, projection, foot-point → pitch
│   │   ├── minimap.py                   # 2D pitch + player markers overlay
│   │   ├── clip_state.py                # ClipMeta / FrameInfo / FrameObservation / TrackInfo / ClipState
│   │   ├── video_processor.py           # pass1 + aggregate + pass2 orchestrator
│   │   ├── job_worker.py                # asyncio queue + thread executor for GPU jobs
│   │   └── history_service.py
│   ├── utils/
│   │   ├── cuda_env.py                  # WSL2 LD_LIBRARY_PATH + ort.preload_dlls bootstrap
│   │   ├── models_registry.py           # central model spec/download registry (Google Drive + sha256)
│   │   └── visualizer.py                # bbox + track + jersey + team color
│   └── vendor/
│       └── pnlcalib/                    # vendored from github.com/mguti97/PnLCalib (model + utils + config)
├── scripts/
│   └── export_hrnet_onnx.py             # SV_kp/SV_lines .pt → ONNX 540×960
├── models/
│   ├── best.onnx                        # YOLO детектор
│   ├── visibility_gate.onnx             # ShuffleNetV2
│   ├── jersey_ocr.onnx (+ .data)        # ConvNeXt-Tiny tens+units
│   ├── hrnet_kp.onnx                    # PnLCalib SV_kp
│   ├── hrnet_lines.onnx                 # PnLCalib SV_lines
│   └── SV_*.pt                          # source weights kept for re-export
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 2024_..._001_initial_migration.py
│       └── 2026_..._002_add_jobs_table.py
├── alembic.ini
├── requirements.txt
└── README.md
```

---

## Технологический стек

* **Язык:** Python 3.12 (3.11 ок)
* **Inference:** onnxruntime-gpu 1.20 (CUDA 12, cudnn 9). `torch 2.5.1` остаётся как зависимость только для heatmap-decode внутри PnLCalib и для офлайн ONNX-экспорта; на пути инференса tracker'а torch не подгружается
* **Computer Vision:** OpenCV (headless), NumPy
* **ML модели:**
  * YOLOv5lu — детекция (player / goalkeeper / referee / ball)
  * BoT-SORT (vendored, app/vendor/boxmot, AGPL-3.0; numpy + scipy + opencv) — motion + appearance tracking, эмбеддинги от DINOv3
  * DINOv3 ViT-S+/16 — общий appearance embedder для ReID и team clustering
  * ShuffleNetV2 — visibility gate (jersey-visibility-project)
  * ConvNeXt-Tiny two-head — jersey OCR (jersey-ocr-project)
  * HRNet-W48 (×2) — field keypoints + lines (PnLCalib SV_kp / SV_lines)
* **Калибровка:** PnLCalib FramebyFrameCalib (per-frame, классический солвер)
* **Кластеризация команд:** scikit-learn `GaussianMixture` (k=2 outfield, k=2 GK отдельно) на 384-d L2-нормированных DINOv3 эмбеддингах; outfield-треки с лог-вероятностью ниже 5-го перцентиля автоматически промоутятся в `referee`
* **Сервис:** FastAPI + Uvicorn, async job worker (1 GPU = 1 воркер)
* **БД:** SQLite + SQLAlchemy 2 + Alembic

---

## Лицензии и атрибуция

* **BoT-SORT** — трекинг построен на коде [BoxMOT](https://github.com/mikel-brostrom/boxmot)
  (Mikel Broström, **AGPL-3.0**). В `app/vendor/boxmot/` лежит урезанный
  torch-free форк (только BoT-SORT, без ReID-бэкбонов); полный список
  изменений и текст лицензии — в `app/vendor/boxmot/NOTICE.md` и
  `app/vendor/boxmot/LICENSE-AGPL`.
* **PnLCalib** — калибровка камеры основана на
  [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib) (**GPL-2.0**);
  в `app/vendor/pnlcalib/` вендорены определения HRNet, декодер хитмап и
  оптимизация калибровки. Веса `SV_kp`/`SV_lines` — из релизов PnLCalib.
* **DINOv3** — эмбеддер использует веса
  [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
  под [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) (Meta).

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

## Куратор

* **Mark Blumenau** (Марк Блуменау)
