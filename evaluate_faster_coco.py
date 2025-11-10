#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x = x_center - width / 2
    y = y_center - height / 2
    
    return [x, y, width, height]


def create_coco_gt(images_dir: Path, labels_dir: Path, class_names: List[str]) -> Dict:
    coco_gt = {"images": [], "annotations": [], "categories": []}
    
    for idx, name in enumerate(class_names):
        coco_gt["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "none"
        })
    
    ann_id = 1
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    for img_id, img_path in enumerate(tqdm(image_files, desc="Converting GT")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}")
            continue
            
        height, width = img.shape[:2]
        
        coco_gt["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })
        
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)
                
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1
    
    return coco_gt


def create_coco_dt(images_dir: Path, pred_labels_dir: Path, 
                   img_id_map: Dict[str, int]) -> List[Dict]:
    coco_dt = []
    
    for img_name, img_id in tqdm(img_id_map.items(), desc="Converting predictions"):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        height, width = img.shape[:2]
        pred_path = pred_labels_dir / f"{Path(img_name).stem}.txt"
        
        if not pred_path.exists():
            continue
            
        with open(pred_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                    
                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]
                score = float(parts[5])
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)
                
                coco_dt.append({
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "score": score
                })
    
    return coco_dt


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_confusion_matrix(cocoGt, cocoDt, iou_threshold: float = 0.5) -> np.ndarray:
    imgIds = cocoGt.getImgIds()
    catIds = cocoGt.getCatIds()
    n_classes = len(catIds)
    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int32)
    
    for imgId in imgIds:
        gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[imgId]))
        dts = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=[imgId]))
        dts = sorted(dts, key=lambda x: x.get('score', 0), reverse=True)
        
        gt_matched = [False] * len(gts)
        
        for dt in dts:
            dt_cat = dt['category_id']
            dt_bbox = dt['bbox']
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue
                iou = compute_iou(dt_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_cat = gts[best_gt_idx]['category_id']
                gt_matched[best_gt_idx] = True
                confusion_matrix[gt_cat, dt_cat] += 1
            else:
                confusion_matrix[n_classes, dt_cat] += 1
        
        for gt_idx, gt in enumerate(gts):
            if not gt_matched[gt_idx]:
                confusion_matrix[gt['category_id'], n_classes] += 1
    
    return confusion_matrix


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         exp_name: str, output_path: Path, iou_threshold: float = 0.5):
    cm_normalized = cm.astype('float')
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)
    
    all_labels = class_names + ['background']
    tick_marks = np.arange(len(all_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_yticklabels(all_labels)
    
    thresh = 0.5
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            val = cm_normalized[i, j]
            count = cm[i, j]
            ax.text(j, i, f'{val:.2f}\n({count})',
                   ha="center", va="center",
                   color="white" if val > thresh else "black",
                   fontsize=8)
    
    ax.set_title(f'Normalized Confusion Matrix - {exp_name}\nIoU Threshold={iou_threshold}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_experiment(exp_name: str, pred_dir: Path, gt_images_dir: Path,
                       gt_labels_dir: Path, class_names: List[str], 
                       output_dir: Path) -> Optional[Dict]:
    print(f"\n{'='*80}")
    print(f"Evaluating: {exp_name}")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating COCO ground truth...")
    coco_gt_dict = create_coco_gt(gt_images_dir, gt_labels_dir, class_names)
    
    gt_json_path = output_dir / f"{exp_name}_gt.json"
    with open(gt_json_path, 'w') as f:
        json.dump(coco_gt_dict, f)
    print(f"GT saved to: {gt_json_path}")
    
    img_id_map = {img['file_name']: img['id'] for img in coco_gt_dict['images']}
    
    print("Creating COCO predictions...")
    pred_labels_dir = pred_dir / "labels"
    coco_dt_list = create_coco_dt(gt_images_dir, pred_labels_dir, img_id_map)
    
    dt_json_path = output_dir / f"{exp_name}_dt.json"
    with open(dt_json_path, 'w') as f:
        json.dump(coco_dt_list, f)
    print(f"DT saved to: {dt_json_path}")
    
    print("\nRunning faster-coco-eval...")
    try:
        from faster_coco_eval import COCO, COCOeval_faster
        
        cocoGt = COCO(coco_gt_dict)
        cocoDt = cocoGt.loadRes(coco_dt_list)
        
        cocoEval = COCOeval_faster(cocoGt, cocoDt, "bbox", extra_calc=True)
        cocoEval.params.maxDets = [100, 300, 1000]
        
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        print("\nGenerating confusion matrix...")
        try:
            iou_threshold = 0.5
            cm = compute_confusion_matrix(cocoGt, cocoDt, iou_threshold)
            matrix_path = output_dir / f"{exp_name}_confusion_matrix.png"
            plot_confusion_matrix(cm, class_names, exp_name, matrix_path, iou_threshold)
            print(f"Confusion matrix saved to: {matrix_path}")
            np.save(output_dir / f"{exp_name}_confusion_matrix.npy", cm)
        except Exception as e:
            print(f"Warning: Could not generate confusion matrix: {e}")
        
        results = {
            "experiment": exp_name,
            "stats": cocoEval.stats.tolist(),
            "stats_as_dict": cocoEval.stats_as_dict,
            "extended_metrics": cocoEval.extended_metrics
        }
        
        results_json_path = output_dir / f"{exp_name}_results.json"
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_json_path}")
        
        return results
        
    except ImportError:
        print("ERROR: faster-coco-eval not installed")
        print("Install with: pip install faster-coco-eval")
        return None


def print_comparison(all_results: Dict[str, Dict]):
    print("\n" + "="*80)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("="*80)
    
    print(f"\n{'Experiment':<20} {'mAP@50:95':<12} {'mAP@50':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*80)
    
    for exp_name, results in all_results.items():
        metrics = results['extended_metrics']
        stats_dict = results['stats_as_dict']
        
        print(f"{exp_name:<20} {stats_dict['AP_all']:<12.4f} {stats_dict['AP_50']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO predictions using faster-coco-eval'
    )
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='Experiment names and prediction directories in format "name:path"')
    parser.add_argument('--gt-images', type=str, required=True,
                       help='Path to ground truth images directory')
    parser.add_argument('--gt-labels', type=str, required=True,
                       help='Path to ground truth labels directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['player', 'goalkeeper', 'referee', 'ball'],
                       help='Class names')
    
    args = parser.parse_args()
    
    if args.experiments:
        experiments = {}
        for exp_str in args.experiments:
            name, path = exp_str.split(':')
            experiments[name] = path
    else:
        experiments = {
            "yolov8m_640": "/srv/home/o.baishev/projects/soc/runs/detect/val",
            "yolov8m_1280": "/srv/home/o.baishev/projects/soc/runs/detect/val2",
            "yolov8l_1280": "/srv/home/o.baishev/projects/soc/runs/detect/val3",
            "yolov8x_1280": "/srv/home/o.baishev/projects/soc/runs/detect/val4",
        }
    
    gt_images_dir = Path(args.gt_images)
    gt_labels_dir = Path(args.gt_labels)
    output_dir = Path(args.output)
    class_names = args.classes
    
    all_results = {}
    
    for exp_name, pred_dir in experiments.items():
        pred_path = Path(pred_dir)
        if not pred_path.exists():
            print(f"Warning: {pred_dir} does not exist, skipping...")
            continue
            
        results = evaluate_experiment(
            exp_name=exp_name,
            pred_dir=pred_path,
            gt_images_dir=gt_images_dir,
            gt_labels_dir=gt_labels_dir,
            class_names=class_names,
            output_dir=output_dir
        )
        
        if results:
            all_results[exp_name] = results
    
    if all_results:
        print_comparison(all_results)
        
        comparison_path = output_dir / "comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull comparison saved to: {comparison_path}")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"\nAll results saved in: {output_dir}/")


if __name__ == "__main__":
    main()
