# Полный пайплайн обработки данных

## Шаг 1: Базовая загрузка всех датасетов

### 1.1. Загрузить все датасеты из основного конфига

```bash
# Загрузить все датасеты (gsr, sportsmot, socceryolo)
python ingest_hub.py
```

**Что загружается** (из `conf/config_ingest.yaml`):
- `gsr_main` - SoccerNet GSR данные (gold, 4 класса)
- `sportsmot_main` - SportsMOT в MOT формате (raw, только pedestrian)
- `yolo_small` - SoccerYolo данные (gold, 4 класса)

**Проверка:**
```bash
# Открыть FiftyOne App
python view_hub.py

# Или в Python
python -c "import fiftyone as fo; hub = fo.load_dataset('football_hub'); print(f'Total samples: {len(hub)}'); print(f'Dataset tags: {hub.distinct(\"dataset_tag\")}')"
```

---

## Шаг 2: Инференс ONNX модели на SportsMOT

### 2.1. Запустить инференс на всех данных SportsMOT

```bash
# Инференс на всех данных sportsmot (все splits)
python infer_model.py infer=sportsmot_inference
```

**Что происходит:**
- Модель обрабатывает все samples с `dataset_tag=sportsmot`
- Создается поле `yolo_predictions` с predictions
- Классы: player, goalkeeper, referee, ball

**Конфиг:** `conf/infer/sportsmot_inference.yaml`
```yaml
name: sportsmot_inference
model:
  type: onnx
  path: /mnt/c/users/user/xanylabeling_data/trainer/ultralytics/runs/detect/footbal_test_train2/weights/best.onnx
  conf_threshold: 0.25
  nms_threshold: 0.45
  img_size: 1280
  class_map:
    0: player
    1: goalkeeper
    2: referee
    3: ball
pred_field: yolo_predictions
filter:
  dataset_tags:
    - sportsmot
  splits: []  # Все splits
  status: null
max_samples: null
```

**Проверка:**
```bash
# Открыть в App и проверить поле yolo_predictions
python view_hub.py
```

---

## Шаг 3: Применить эвристику (опционально, но рекомендуется)

### 3.1. Применить эвристическое улучшение

```bash
# Применить эвристику для автоматического улучшения разметки
python refine_sportsmot.py refine=sportsmot_heuristic
```

**Что происходит:**
- Сопоставление GT (pedestrian) с predictions по IoU > 0.8
- Автоматическая замена классов:
  - pedestrian → player
  - pedestrian → goalkeeper (max 2)
  - pedestrian → referee
- Добавление топ-1 ball с максимальным confidence
- Добавление топ-2 referee с максимальным confidence
- Создается поле `refined_detections`

**Конфиг:** `conf/refine/sportsmot_heuristic.yaml`
```yaml
name: sportsmot_heuristic_refinement
gt_field: detections
pred_field: yolo_predictions
output_field: refined_detections
iou_threshold: 0.8
max_goalkeepers: 2
max_referees: 2
filter:
  dataset_tags:
    - sportsmot
  splits:
    - train
    - valid
    - test
  status: null
limit: null
```

---

## Шаг 4: Выгрузка в CVAT для проверки

### 4.1. Подготовка CVAT

**Убедитесь, что CVAT запущен:**

```bash
# Для локального CVAT
curl http://localhost:8080
# Должен вернуть ответ

# Если CVAT не запущен, запустите:
cd /path/to/cvat
docker compose up -d
```

**Установите credentials:**

```bash
# Для локального CVAT
export FIFTYONE_CVAT_URL=http://localhost:8080
export FIFTYONE_CVAT_USERNAME=admin
export FIFTYONE_CVAT_PASSWORD=your_password

# Или для облачного app.cvat.ai
export FIFTYONE_CVAT_URL=https://app.cvat.ai
export FIFTYONE_CVAT_USERNAME=your_cvat_username
export FIFTYONE_CVAT_PASSWORD=your_cvat_password
```

### 4.2. Отправить данные в CVAT

**Вариант A: С эвристикой (рекомендуется)**

```bash
# Отправить refined_detections (после эвристики) в CVAT
python send_to_cvat.py cvat=sportsmot_refined
```

**Вариант B: Без эвристики (только predictions)**

```bash
# Отправить yolo_predictions напрямую в CVAT
python send_to_cvat.py cvat=sportsmot_predictions
```

**Что происходит:**
- Samples с `dataset_tag=sportsmot` фильтруются
- Берется каждый 25-й кадр (`frame_interval: 25`)
- Все splits (train, valid, test)
- Загружаются в CVAT проект `sportsmot_refined_verification`
- Автоматически открывается браузер с CVAT

**Конфиг (с эвристикой):** `conf/cvat/sportsmot_refined.yaml`
```yaml
name: sportsmot_refined_annotations
anno_key: sportsmot_refined_v1
label_field: refined_detections  # После эвристики
cvat_url: http://localhost:8080
classes:
  - player
  - goalkeeper
  - referee
  - ball
filter:
  dataset_tags:
    - sportsmot
  splits:
    - train
    - valid
    - test
subsample_configs:
  sportsmot:
    enabled: true
    frame_interval: 25  # Каждый 25-й кадр
task_size: 50
segment_size: 25
project_name: sportsmot_refined_verification
add_tag_on_import: cvat_verified_v1
```

---

## Шаг 5: Разметка в CVAT

### 5.1. В веб-интерфейсе CVAT

1. Откройте проект `sportsmot_refined_verification`
2. Проверьте и исправьте аннотации:
   - Правильность классов (player/goalkeeper/referee/ball)
   - Корректность bounding boxes
   - Пропущенные объекты
   - Лишние detections
3. Сохраните изменения

**Если использовали эвристику:**
- Большая часть работы уже сделана
- Нужно только проверить и исправить ошибки
- Значительно быстрее чем полная разметка с нуля

---

## Шаг 6: Загрузка обратно из CVAT

### 6.1. Импортировать проверенные аннотации

```bash
# Импортировать аннотации обратно в FiftyOne
python send_to_cvat.py cvat=sportsmot_refined cvat.load_annotations=true
```

**Что происходит:**
- Аннотации загружаются в поле `refined_detections`
- К samples добавляется тег `cvat_verified_v1`
- Теперь можно фильтровать по этому тегу

**С дополнительными опциями:**

```bash
# Импорт + удаление задач из CVAT
python send_to_cvat.py cvat=sportsmot_refined \
  cvat.load_annotations=true \
  cvat.cleanup=true

# Импорт в другое поле (не перезаписывать refined_detections)
python send_to_cvat.py cvat=sportsmot_refined \
  cvat.load_annotations=true \
  cvat.dest_field=cvat_corrected_detections

# Полная очистка (импорт + удаление задач + удаление записи)
python send_to_cvat.py cvat=sportsmot_refined \
  cvat.load_annotations=true \
  cvat.cleanup=true \
  cvat.delete_run=true
```

**Проверка:**

```bash
# Проверить тег
python -c "import fiftyone as fo; hub = fo.load_dataset('football_hub'); view = hub.match_tags('cvat_verified_v1'); print(f'Samples с тегом: {len(view)}')"

# Открыть в App
python view_hub.py
# В App: фильтр по тегу 'cvat_verified_v1'
```

---

## Шаг 7: Экспорт переделанного SportsMOT

### 7.1. Экспорт всех CVAT-verified samples

```bash
# Экспортировать только проверенные в CVAT samples в YOLO и COCO
python export_hub.py export=sportsmot_cvat_verified
```

**Результат:**
- `exports/sportsmot_cvat_verified/yolo/` - YOLO формат
- `exports/sportsmot_cvat_verified/coco/` - COCO формат
- Только samples с тегом `cvat_verified_v1`
- Каждый 25-й кадр
- Все splits (train, val, test)
- Все классы (player, goalkeeper, referee, ball)

**Конфиг:** `conf/export/sportsmot_cvat_verified.yaml`
```yaml
name: sportsmot_cvat_verified
label_field: refined_detections
dataset_tags:
  - sportsmot
sample_tags:
  - cvat_verified_v1  # Только проверенные
classes:
  - player
  - goalkeeper
  - referee
  - ball
splits:
  - train
  - valid
  - test
subsample_configs:
  sportsmot:
    enabled: true
    frame_interval: 25
formats:
  - yolo
  - coco
output_dir: exports
```

**Структура экспорта:**

```
exports/sportsmot_cvat_verified/
├── yolo/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
└── coco/
    ├── images/
    │   ├── train/
    │   ├── val/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
```

### 7.2. Варианты экспорта

```bash
# Только YOLO
python export_hub.py export=sportsmot_cvat_verified export.formats=[yolo]

# Только COCO
python export_hub.py export=sportsmot_cvat_verified export.formats=[coco]

# В другую директорию
python export_hub.py export=sportsmot_cvat_verified export.output_dir=my_exports
```

---

## Шаг 8: Добавление переделанного SportsMOT обратно

### 8.1. Создать новый конфиг для загрузки

Создайте `conf/config_ingest_v2.yaml`:

```yaml
# Hydra Configuration for Data Ingestion v2
# Только socceryolo + переделанный sportsmot

defaults:
  - _self_

hub:
  name: football_hub
  persistent: true

datasets:
  
  # Переделанный SportsMOT из YOLO формата
  sportsmot_refined:
    enabled: true
    type: yolo  # Теперь YOLO формат
    tag: sportsmot
    path: /home/dxdxxd/projects/dataIntegratorSoccer/exports/sportsmot_cvat_verified/yolo
    splits:
      - train
      - val
      - test
    status: gold  # Теперь gold (4 класса)
    class_mapping:
      "0": player
      "1": goalkeeper
      "2": referee
      "3": ball
  
  # SoccerYolo данные (без изменений)
  yolo_small:
    enabled: true
    type: yolo
    tag: socceryolo
    path: /mnt/d/datasets/socceryolo
    splits:
      - train
      - valid
      - test
    status: gold
    class_mapping:
      "0": ball
      "1": goalkeeper
      "2": player
      "3": referee
```

**Важно:**
- Проверьте `class_mapping` в `exports/sportsmot_cvat_verified/yolo/data.yaml`
- Индексы должны совпадать с тем, что экспортировал `export_hub.py`

### 8.2. Удалить старый SportsMOT и загрузить новый

```bash
# 1. Удалить старый sportsmot (MOT формат)
python ingest_hub.py +delete_datasets=[sportsmot]

# 2. Загрузить новый sportsmot (YOLO формат) из config_ingest_v2
python ingest_hub.py --config-name=config_ingest_v2 +datasets_to_process=[sportsmot]

# Или одной командой (удалить + загрузить)
python ingest_hub.py --config-name=config_ingest_v2 \
  +delete_datasets=[sportsmot] \
  +datasets_to_process=[sportsmot]
```

### 8.3. Полная перезагрузка датасета

Если нужно загрузить только socceryolo + переделанный sportsmot:

```bash
# 1. Удалить весь hub
python view_hub.py --delete

# 2. Загрузить только socceryolo + sportsmot_refined
python ingest_hub.py --config-name=config_ingest_v2
```

**Проверка:**

```bash
# Проверить датасеты
python -c "import fiftyone as fo; hub = fo.load_dataset('football_hub'); print(f'Total: {len(hub)}'); print(f'Tags: {hub.distinct(\"dataset_tag\")}'); print(f'Splits: {hub.distinct(\"split\")}')"

# Открыть в App
python view_hub.py
```

---

## Ссылки на конфиги

- **Инференс:** `conf/infer/sportsmot_inference.yaml`
- **Эвристика:** `conf/refine/sportsmot_heuristic.yaml`
- **CVAT (с эвристикой):** `conf/cvat/sportsmot_refined.yaml`
- **CVAT (без эвристики):** `conf/cvat/sportsmot_predictions.yaml`
- **Экспорт:** `conf/export/sportsmot_cvat_verified.yaml`
- **Загрузка v2:** `conf/config_ingest_v2.yaml`

---