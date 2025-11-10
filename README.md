# Eval моделей YOLO

```bash
pip install -r requirements.txt
```

```bash
python3 evaluate_faster_coco.py \
    --experiments \
        "yolov8m_640:/путь/к/runs/detect/val" \
        "yolov8m_1280:/путь/к/runs/detect/val2" \
    --gt-images /путь/к/test/images \
    --gt-labels /путь/к/test/labels \
    --output /путь/для/результатов \
    --classes player goalkeeper referee ball
```

## Структура данных

### Eval yolo в Python

```results = model.val(data='main_data.yaml', split="test", save_txt=True, save_conf=True)```

### Предсказания модели (YOLO формат)
```
/путь/к/runs/detect/val/
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Формат файла предсказаний:
```
class_id x_center y_center width height confidence
0 0.5 0.5 0.3 0.4 0.95
1 0.2 0.3 0.1 0.2 0.87
```

### Ground truth (YOLO формат)
```
/путь/к/test/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```

Формат файла разметки:
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

## Результаты

После выполнения в папке `--output` появятся:

```
результаты/
├── yolov8m_640_gt.json              # Ground truth в COCO формате
├── yolov8m_640_dt.json              # Предсказания в COCO формате
├── yolov8m_640_results.json         # Метрики
├── yolov8m_640_confusion_matrix.png # Матрица ошибок
├── yolov8m_640_confusion_matrix.npy # Сырые данные матрицы
└── comparison.json                  # Сравнение всех экспериментов
```
