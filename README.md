# ⚽ soccerGSR: Game State Reconstruction in Soccer

## Exploratory Data Analysis (EDA)

### 🔧 Установка окружения

#### Conda 
```bash
# Создать окружение из файла
conda env create -f environment.yml

# Активировать окружение
conda activate soccer_eda

# Установить дополнительные pip зависимости
pip install -r requirements.txt
```

### Датасеты

Для работы необходим датасет **SoccerNet Game State Reconstruction**.

**Скачать датасет:**
https://disk.360.yandex.ru/d/CJDIWx1TFLYpxA
Содержимое положить в директорию data внутри основной директории

Планируется также задействовать датасет [**SportsMOT**](https://www.kaggle.com/datasets/ayushspai/sportsmot).

**Скачать датасет:**
https://disk.360.yandex.ru/d/5Ta8nKRVVrr5Ug
(здесь уже оставлены только футбольные клипы, а также вдобавок к MOT формату добавлен COCO формат)

### Множественные форматы для работы с датасетом SoccerNet
```
python convert_labels.py --format coco yolo --copy-images
```

#### Что делает скрипт?

- **COCO формат** - для object detection моделей (YOLO, Faster R-CNN, DETR)
  - Создает JSON файлы с аннотациями
  - Копирует изображения в структуру `images/train|valid|test/`
  - Поддерживает 3 класса: player, goalkeeper, referee

- **YOLO формат** - для семейства YOLO моделей
  - Создает `.txt` файлы с нормализованными bbox
  - Генерирует `data.yaml` конфигурацию
  - Организует структуру для обучения

- **MOT формат** - для алгоритмов трекинга (SORT, DeepSORT, BoT-SORT)
  - Создает `gt.txt` файлы с треками
  - Включает `seqinfo.ini` с метаданными
  - Готов для бенчмарка MOT Challenge

Подробный анализ датасета SoccerNet находится в ноутбуке **`notebooks/eda.ipynb`**

Аналогичный анализ датасета SportsMOT находится в ноутбуке **`notebooks/edaSportsMOT.ipynb`**