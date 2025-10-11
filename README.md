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

### Датасет

Для работы необходим датасет **SoccerNet Game State Reconstruction**.

**Скачать датасет:**

### Множественные форматы
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

Подробный анализ датасета находится в ноутбуке **`notebooks/eda.ipynb`**
