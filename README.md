**SoccerNet (GSR):**

https://disk.360.yandex.ru/d/CJDIWx1TFLYpxA
Содержимое положить в директорию data внутри основной директории


**SportsMOT**

https://disk.360.yandex.ru/d/5Ta8nKRVVrr5Ug
(здесь уже оставлены только футбольные клипы, а также вдобавок к MOT формату добавлен COCO формат)

**socceryolo (маленький, но хорошо размеченный даатсет с roboflow)**

https://disk.360.yandex.ru/d/3h6sT_o1ZYhT1A


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

## 3\. Просмотр в FiftyOne

```bash
python view_hub.py
```

Полностью удалить хаб из FiftyOne:

```bash
python view_hub.py --delete
```
