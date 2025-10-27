#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Tuple, Optional
from collections import Counter

import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes in [x, y, w, h] format."""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


class SportsMOTRefiner:
    """Heuristic refinement for SportsMOT annotations."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name not in fo.list_datasets():
            raise ValueError(f"Hub '{self.hub_name}' not found")
        
        self.hub = fo.load_dataset(self.hub_name)
        print(f"Loaded hub: {self.hub_name} ({len(self.hub)} samples)")
    
    def refine(self, refine_cfg: DictConfig):
        print(f"\n{'=' * 80}")
        print(f"Refinement: {refine_cfg.name}")
        print(f"{'=' * 80}")
        
        view = self._build_filtered_view(refine_cfg)
        print(f"Samples to process: {len(view)}")
        
        if len(view) == 0:
            print("No samples to process")
            return
        
        gt_field = refine_cfg.gt_field
        pred_field = refine_cfg.pred_field
        output_field = refine_cfg.output_field
        iou_threshold = refine_cfg.get('iou_threshold', 0.8)
        
        max_goalkeepers = refine_cfg.get('max_goalkeepers', 2)
        max_referees = refine_cfg.get('max_referees', 2)
        
        print(f"\nConfiguration:")
        print(f"  GT field: {gt_field}")
        print(f"  Predictions field: {pred_field}")
        print(f"  Output field: {output_field}")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"  Max goalkeepers: {max_goalkeepers}")
        print(f"  Max referees: {max_referees}")
        
        stats = {
            'total_samples': 0,
            'pedestrian_to_player': 0,
            'pedestrian_to_goalkeeper': 0,
            'pedestrian_to_referee': 0,
            'balls_added': 0,
            'referees_added': 0,
        }
        
        print(f"\nProcessing samples...")
        for sample in view.iter_samples(progress=True):
            refined_detections = self._refine_sample(
                sample,
                gt_field,
                pred_field,
                iou_threshold,
                max_goalkeepers,
                max_referees,
                stats
            )
            
            sample[output_field] = fo.Detections(detections=refined_detections)
            sample.save()
            stats['total_samples'] += 1
        
        self._print_statistics(stats)
        
        print(f"\n{'=' * 80}")
        print(f"Refinement complete! Saved to field '{output_field}'")
        print(f"{'=' * 80}")
    
    def _refine_sample(
        self,
        sample: fo.Sample,
        gt_field: str,
        pred_field: str,
        iou_threshold: float,
        max_goalkeepers: int,
        max_referees: int,
        stats: Dict
    ) -> List[fo.Detection]:
        refined = []
        
        gt_detections = sample[gt_field].detections if sample[gt_field] else []
        pred_detections = sample[pred_field].detections if sample[pred_field] else []
        
        used_pred_indices = set()
        goalkeeper_count = 0
        
        for gt_det in gt_detections:
            if gt_det.label.lower() not in ['pedestrian', 'person']:
                refined.append(gt_det)
                continue
            
            gt_box = gt_det.bounding_box
            best_iou = 0.0
            best_pred_idx = -1
            best_pred_label = None
            
            for pred_idx, pred_det in enumerate(pred_detections):
                if pred_idx in used_pred_indices:
                    continue
                
                pred_label = pred_det.label.lower()
                if pred_label not in ['player', 'goalkeeper', 'referee']:
                    continue
                
                pred_box = pred_det.bounding_box
                iou = compute_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
                    best_pred_label = pred_label
            
            if best_iou >= iou_threshold and best_pred_label:
                used_pred_indices.add(best_pred_idx)
                
                if best_pred_label == 'goalkeeper':
                    if goalkeeper_count >= max_goalkeepers:
                        best_pred_label = 'player'
                    else:
                        goalkeeper_count += 1
                        stats['pedestrian_to_goalkeeper'] += 1
                
                new_det = gt_det.copy()
                new_det.label = best_pred_label
                refined.append(new_det)
                
                if best_pred_label == 'player':
                    stats['pedestrian_to_player'] += 1
                elif best_pred_label == 'referee':
                    stats['pedestrian_to_referee'] += 1
            else:
                new_det = gt_det.copy()
                new_det.label = 'player'
                refined.append(new_det)
        
        ball_detections = [
            det for det in pred_detections
            if det.label.lower() == 'ball'
        ]
        
        if ball_detections:
            ball_detections.sort(key=lambda x: x.confidence, reverse=True)
            best_ball = ball_detections[0].copy()
            refined.append(best_ball)
            stats['balls_added'] += 1
        
        referee_detections = [
            (idx, det) for idx, det in enumerate(pred_detections)
            if det.label.lower() == 'referee' and idx not in used_pred_indices
        ]
        
        referee_detections.sort(key=lambda x: x[1].confidence, reverse=True)
        
        referee_count = 0
        for pred_idx, ref_det in referee_detections[:max_referees]:
            ref_box = ref_det.bounding_box
            matched_gt = False
            
            for gt_det in gt_detections:
                if gt_det.label.lower() not in ['pedestrian', 'person']:
                    continue
                
                gt_box = gt_det.bounding_box
                iou = compute_iou(ref_box, gt_box)
                
                if iou >= iou_threshold:
                    matched_gt = True
                    break
            
            if not matched_gt:
                new_ref = ref_det.copy()
                refined.append(new_ref)
                referee_count += 1
                stats['referees_added'] += 1
        
        return refined
    
    def _build_filtered_view(self, refine_cfg: DictConfig) -> fo.DatasetView:
        view = self.hub
        
        if refine_cfg.get('filter') and refine_cfg.filter.get('dataset_tags'):
            tags = list(refine_cfg.filter.dataset_tags)
            if len(tags) > 0:
                from functools import reduce
                import operator
                dataset_filter = [F("dataset_tag") == tag for tag in tags]
                if len(dataset_filter) == 1:
                    view = view.match(dataset_filter[0])
                else:
                    combined = reduce(operator.or_, dataset_filter)
                    view = view.match(combined)
                print(f"Filtered by dataset tags: {tags}")
        
        if refine_cfg.get('filter') and refine_cfg.filter.get('splits'):
            splits = list(refine_cfg.filter.splits)
            if len(splits) > 0:
                view = view.match(F("split").is_in(splits))
                print(f"Filtered by splits: {splits}")
        
        if refine_cfg.get('filter') and refine_cfg.filter.get('status'):
            status = refine_cfg.filter.status
            view = view.match(F("status") == status)
            print(f"Filtered by status: {status}")
        
        if refine_cfg.get('limit'):
            view = view.limit(refine_cfg.limit)
            print(f"Limited to {refine_cfg.limit} samples")
        
        return view
    
    def _print_statistics(self, stats: Dict):
        print(f"\n{'=' * 80}")
        print(f"Refinement Statistics:")
        print(f"{'=' * 80}")
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"\nClass conversions:")
        print(f"  pedestrian → player:      {stats['pedestrian_to_player']}")
        print(f"  pedestrian → goalkeeper:  {stats['pedestrian_to_goalkeeper']}")
        print(f"  pedestrian → referee:     {stats['pedestrian_to_referee']}")
        print(f"\nNew detections added:")
        print(f"  Balls added:              {stats['balls_added']}")
        print(f"  Referees added:           {stats['referees_added']}")
        
        total_conversions = (
            stats['pedestrian_to_player'] +
            stats['pedestrian_to_goalkeeper'] +
            stats['pedestrian_to_referee']
        )
        print(f"\nTotal conversions: {total_conversions}")


@hydra.main(version_base=None, config_path="conf", config_name="config_refine")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    refiner = SportsMOTRefiner(cfg)
    refiner.refine(cfg.refine)


if __name__ == "__main__":
    main()

