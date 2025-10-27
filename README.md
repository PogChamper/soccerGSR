**SoccerNet (GSR):**

https://disk.360.yandex.ru/d/CJDIWx1TFLYpxA
Содержимое положить в директорию data внутри основной директории


**SportsMOT**

https://disk.360.yandex.ru/d/5Ta8nKRVVrr5Ug
(здесь уже оставлены только футбольные клипы, а также вдобавок к MOT формату добавлен COCO формат)

**socceryolo (маленький, но хорошо размеченный даатсет с roboflow)** - размечены все 4 класса, готов для использования в обучении

https://disk.360.yandex.ru/d/3h6sT_o1ZYhT1A

**SportsMOT в gold формате, размечены все 4 класса, готов для использования в обучении**

https://disk.360.yandex.ru/d/egZgpzlGALA9KA


**Инструкции по новому функционалу тут: PIPELINE_FULL.md**

**Веса модели yolo11x (1280), которые использовал для исправления разметки sportsmot: https://disk.360.yandex.ru/d/OJHvU7A19dqbTg**


## Установка

### Создать окружение и установить зависимости

```bash
conda create -n data_tool python=3.12
conda activate data_tool
pip install -r requirements.txt
```

-----

## 1\. Загрузка данных

`ingest_hub.py` - загружает данные в FiftyOne.

  * **GSR (`type: gsr`)**: SoccerNet Game State Reconstruction (JSON-аннотации, трекинг, форма).
  * **MOT Frames (`type: mot_frames`)**: Стандартный MOT (папки с кадрами + `gt.txt`).
  * **YOLO (`type: yolo`)**: Стандартный YOLO (картинки + `.txt` лейблы).

### Основные команды

```bash
# Загрузить все датасеты из в conf/config_ingest.yaml
python ingest_hub.py

# Загрузить только конкретные датасеты (по тегам из конфига)
python ingest_hub.py +datasets_to_process=[gsr,socceryolo]

# Удалить датасеты из FiftyOne
python ingest_hub.py +delete_datasets=[socceryolo]

# Полностью пересоздать датасет (удалить и загрузить заново)
python ingest_hub.py +delete_datasets=[supertest] +datasets_to_process=[supertest]
```

-----

## 2\. Экспорт

`export_hub.py` - когда данные в FiftyOne, их можно выгрузить в нужном формате для обучения.

Примеры в `conf/export`.

```bash
# Выгрузить только семплы, где есть класс "мяч", и лейблы только для него
python export_hub.py export=ball_only
```

```bash
# Выгрузить полные датасеты, где размечены все 4 класса(=gold)
python export_hub.py export=detection_full +export.status=gold
```

```bash
# Выгрузить только sportsmot (допустим, хотим прогнать на нем обученную
# на 4 класса модель, чтобы полуавтоматически доразметить)
python export_hub.py export=detection_full export.dataset_tags=[sportsmot]
```

-----

## 3\. Инференс модели

`infer_model.py` - запуск ONNX/Ultralytics модели на данных из хаба.

```bash
# Инференс на всех данных sportsmot
python infer_model.py infer=sportsmot_inference

# На конкретном split
python infer_model.py infer=sportsmot_inference \
  infer.filter.splits=[test]

# Ограничить количество samples
python infer_model.py infer=sportsmot_inference +max_samples=100
```

**Что происходит:**
- Модель обрабатывает samples с `dataset_tag=sportsmot`
- Создается поле `yolo_predictions` с результатами
- Классы: player, goalkeeper, referee, ball

**Конфиг:** `conf/infer/sportsmot_inference.yaml`

-----

## 4\. Оценка модели (метрики)

`evaluate_model.py` - подсчет метрик качества модели.

```bash
# COCO метрики на всем sportsmot (все splits)
python evaluate_model.py evaluate=coco_protocol \
  +evaluate.filter.dataset_tags=[sportsmot] \
  +evaluate.filter.splits=[]

# Только на test split
python evaluate_model.py evaluate=coco_protocol \
  +evaluate.filter.dataset_tags=[sportsmot] \
  +evaluate.filter.splits=[test]
```

**Что получите:**
- mAP
- Precision, Recall, F1 по каждому классу
- Confusion matrix (PNG в `outputs/evaluations/`)
- PR кривые (PNG в `outputs/evaluations/`)

-----

## 5\. Просмотр в FiftyOne

```bash
# Открыть FiftyOne App
python view_hub.py

# Полностью удалить хаб
python view_hub.py --delete
```

-----
