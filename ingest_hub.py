#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import fiftyone as fo


class FlexibleDatasetIngester:
    """Ingest datasets from various formats into a unified FiftyOne hub."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name in fo.list_datasets():
            print(f"Loading existing hub: {self.hub_name}")
            self.hub = fo.load_dataset(self.hub_name)
        else:
            print(f"Creating new hub: {self.hub_name}")
            self.hub = fo.Dataset(
                name=self.hub_name,
                persistent=cfg.hub.persistent
            )
    
    def delete_by_tag(self, dataset_tag: str):
        """Delete all samples with specified dataset_tag."""
        view = self.hub.match(fo.ViewField("dataset_tag") == dataset_tag)
        num_samples = len(view)
        
        if num_samples == 0:
            print(f"No samples found with tag: {dataset_tag}")
            return
        
        print(f"Deleting {num_samples} samples with tag: {dataset_tag}")
        self.hub.delete_samples(view)
        print(f"Deleted {num_samples} samples")
    
    def ingest_dataset(self, dataset_cfg: DictConfig):
        """Ingest a single dataset based on its configuration."""
        dataset_type = dataset_cfg.type
        dataset_tag = dataset_cfg.tag
        
        print(f"\n{'=' * 80}")
        print(f"Ingesting: {dataset_tag} (type: {dataset_type})")
        print(f"{'=' * 80}")
        
        ingest_methods = {
            "gsr": self._ingest_gsr,
            "mot_frames": self._ingest_mot_frames,
            "mot_video": self._ingest_mot_video,
            "yolo": self._ingest_yolo,
            "coco": self._ingest_coco,
        }
        
        method = ingest_methods.get(dataset_type)
        if method:
            method(dataset_cfg)
        else:
            print(f"Unknown dataset type: {dataset_type}")
    
    def _ingest_gsr(self, dataset_cfg: DictConfig):
        """Ingest GSR format dataset."""
        root = Path(dataset_cfg.path)
        
        if not root.exists():
            print(f"Path not found: {root}")
            return
        
        status = dataset_cfg.get('status', 'gold')
        
        for split in dataset_cfg.splits:
            split_dir = root / split
            if not split_dir.exists():
                continue
            
            clip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            print(f"\nProcessing {split} split: {len(clip_dirs)} clips")
            
            for clip_dir in tqdm(clip_dirs, desc=f"GSR {split}"):
                self._process_gsr_clip(clip_dir, split, dataset_cfg.tag, status)
    
    def _process_gsr_clip(self, clip_dir: Path, split: str, dataset_tag: str,
                           status: str):
        """Process a single GSR clip directory."""
        json_path = clip_dir / "Labels-GameState.json"
        if not json_path.exists():
            return
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        info = data['info']
        img_dir = clip_dir / info['im_dir']
        
        if not img_dir.exists():
            return
        
        image_info_map = {
            img['image_id']: img for img in data.get('images', [])
        }
        
        frame_annotations = {}
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in frame_annotations:
                frame_annotations[image_id] = []
            frame_annotations[image_id].append(ann)
        
        for image_id, img_info in image_info_map.items():
            frame_path = img_dir / img_info['file_name']
            if not frame_path.exists():
                continue
            
            frame_num = int(img_info['file_name'].split('.')[0])
            
            sample = fo.Sample(filepath=str(frame_path))
            sample["dataset_tag"] = dataset_tag
            sample["clip_name"] = clip_dir.name
            sample["frame_number"] = frame_num
            sample["source"] = "gsr"
            sample["split"] = split
            sample["status"] = status
            sample["type"] = "video_frame"
            sample["frame_rate"] = info.get('frame_rate', 25)
            sample["image_width"] = img_info.get('width', 1920)
            sample["image_height"] = img_info.get('height', 1080)
            sample.tags.extend([
                f"dataset:{dataset_tag}",
                f"split:{split}",
                f"status:{status}"
            ])
            
            if image_id in frame_annotations:
                detections = []
                for ann in frame_annotations[image_id]:
                    det = self._create_gsr_detection(ann, img_info)
                    if det:
                        detections.append(det)
                if detections:
                    sample["detections"] = fo.Detections(detections=detections)
            
            self.hub.add_sample(sample)
    
    def _create_gsr_detection(self, ann: Dict, img_info: Dict):
        """Create FiftyOne detection from GSR annotation."""
        bbox_img = ann.get('bbox_image', {})
        if not bbox_img:
            return None
        
        x, y, w, h = bbox_img['x'], bbox_img['y'], bbox_img['w'], bbox_img['h']
        img_w, img_h = img_info.get('width', 1920), img_info.get('height', 1080)
        bbox_norm = [x / img_w, y / img_h, w / img_w, h / img_h]
        
        label_map = {1: 'player', 2: 'goalkeeper', 3: 'referee', 4: 'ball'}
        category_id = ann.get('category_id')
        label = label_map.get(category_id, 'player')
        
        attributes = ann.get('attributes', {})
        role = attributes.get('role', 'player')
        if role in label_map.values():
            label = role
        
        return fo.Detection(
            label=label,
            bounding_box=bbox_norm,
            track_id=ann.get('track_id'),
            jersey_number=attributes.get('jersey'),
            team=attributes.get('team'),
            confidence=1.0
        )
    
    def _ingest_mot_frames(self, dataset_cfg: DictConfig):
        """Ingest MOT format with frame directories."""
        root = Path(dataset_cfg.path)
        
        if not root.exists():
            print(f"Path not found: {root}")
            return
        
        status = dataset_cfg.get('status', 'raw')
        
        for split in dataset_cfg.splits:
            split_dir = root / split
            if not split_dir.exists():
                continue
            
            clip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
            print(f"\nProcessing {split} split: {len(clip_dirs)} clips")
            
            for clip_dir in tqdm(clip_dirs, desc=f"MOT {split}"):
                self._process_mot_frames_clip(clip_dir, split, dataset_cfg.tag, status)
    
    def _process_mot_frames_clip(self, clip_dir: Path, split: str,
                                  dataset_tag: str, status: str):
        """Process a single MOT frames clip directory."""
        img_dir = clip_dir / "img1"
        gt_file = clip_dir / "gt" / "gt.txt"
        
        if not img_dir.exists() or not gt_file.exists():
            return
        
        seqinfo = self._parse_seqinfo(clip_dir / "seqinfo.ini")
        img_w = seqinfo.get('imWidth', 1280)
        img_h = seqinfo.get('imHeight', 720)
        
        frame_annotations = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                
                frame_num = int(parts[0])
                if frame_num not in frame_annotations:
                    frame_annotations[frame_num] = []
                
                frame_annotations[frame_num].append({
                    'track_id': int(parts[1]),
                    'bbox': [float(p) for p in parts[2:6]],
                    'confidence': float(parts[6]) if len(parts) > 6 else 1.0
                })
        
        for frame_num, annotations in frame_annotations.items():
            frame_path = img_dir / f"{frame_num:06d}.jpg"
            if not frame_path.exists():
                continue
            
            sample = fo.Sample(filepath=str(frame_path))
            sample["dataset_tag"] = dataset_tag
            sample["clip_name"] = clip_dir.name
            sample["frame_number"] = frame_num
            sample["source"] = "mot_frames"
            sample["split"] = split
            sample["status"] = status
            sample["type"] = "video_frame"
            sample["frame_rate"] = seqinfo.get('frameRate', 25)
            sample["image_width"] = img_w
            sample["image_height"] = img_h
            sample.tags.extend([
                f"dataset:{dataset_tag}",
                f"split:{split}",
                f"status:{status}"
            ])
            
            detections = []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                bbox_norm = [x / img_w, y / img_h, w / img_w, h / img_h]
                
                detections.append(fo.Detection(
                    label='player',
                    bounding_box=bbox_norm,
                    track_id=ann['track_id'],
                    confidence=ann['confidence']
                ))
            
            if detections:
                sample["detections"] = fo.Detections(detections=detections)
            
            self.hub.add_sample(sample)
    
    def _parse_seqinfo(self, seqinfo_path: Path) -> Dict:
        """Parse MOT seqinfo.ini configuration file."""
        if not seqinfo_path.exists():
            return {}
        
        info = {}
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('['):
                    key, value = line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    
                    info[key] = value
        return info
    
    def _ingest_yolo(self, dataset_cfg: DictConfig):
        """Ingest YOLO format dataset."""
        root = Path(dataset_cfg.path)
        
        if not root.exists():
            print(f"Path not found: {root}")
            return
        
        class_map = {str(k): v for k, v in dataset_cfg.class_mapping.items()}
        status = dataset_cfg.get('status', 'gold')
        
        for split in dataset_cfg.splits:
            split_dir = root / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = sorted(images_dir.glob("*.jpg"))
            print(f"\nProcessing {split} split: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"YOLO {split}"):
                label_path = labels_dir / (img_path.stem + ".txt")
                
                sample = fo.Sample(filepath=str(img_path))
                sample["dataset_tag"] = dataset_cfg.tag
                sample["source"] = "yolo"
                sample["split"] = split
                sample["status"] = status
                sample["type"] = "image"
                sample.tags.extend([
                    f"dataset:{dataset_cfg.tag}",
                    f"split:{split}",
                    f"status:{status}"
                ])
                
                if label_path.exists():
                    detections = self._parse_yolo_labels(label_path, class_map)
                    if detections:
                        sample["detections"] = fo.Detections(
                            detections=detections
                        )
                
                self.hub.add_sample(sample)
    
    def _parse_yolo_labels(self, label_path: Path, class_map: Dict) -> List:
        """Parse YOLO format label file and return list of detections."""
        detections = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = parts[0]
                x_center, y_center = float(parts[1]), float(parts[2])
                width, height = float(parts[3]), float(parts[4])
                
                x = x_center - width / 2
                y = y_center - height / 2
                
                label = class_map.get(class_id, 'player')
                
                detections.append(fo.Detection(
                    label=label,
                    bounding_box=[x, y, width, height],
                    confidence=1.0
                ))
        return detections
    
    def _ingest_coco(self, dataset_cfg: DictConfig):
        """Ingest COCO format dataset (not yet implemented)."""
        print("COCO ingestion not yet implemented")
    
    def _ingest_mot_video(self, dataset_cfg: DictConfig):
        """Ingest MOT format with video files (not yet implemented)."""
        print("MOT video ingestion not yet implemented")


@hydra.main(version_base=None, config_path="conf", config_name="config_ingest")
def main(cfg: DictConfig):
    """Main entry point for dataset ingestion."""
    print(OmegaConf.to_yaml(cfg))
    
    ingester = FlexibleDatasetIngester(cfg)
    
    delete_datasets = cfg.get('delete_datasets', [])
    if delete_datasets:
        print(f"\n{'=' * 80}")
        print("Deleting datasets")
        print(f"{'=' * 80}")
        for dataset_tag in delete_datasets:
            ingester.delete_by_tag(dataset_tag)
    
    datasets_to_process_specified = 'datasets_to_process' in cfg
    datasets_to_process = cfg.get('datasets_to_process', [])
    
    if delete_datasets and not datasets_to_process_specified:
        print(f"\n{'=' * 80}")
        print(f"Deletion complete! Hub contains {len(ingester.hub)} samples")
        print(f"{'=' * 80}")
        return
    
    processed_count = 0
    for dataset_name, dataset_cfg in cfg.datasets.items():
        if not dataset_cfg.get('enabled', True):
            continue
        
        if datasets_to_process and dataset_cfg.tag not in datasets_to_process:
            continue
        
        ingester.ingest_dataset(dataset_cfg)
        processed_count += 1
    
    print(f"\n{'=' * 80}")
    print(f"Ingestion complete! Processed {processed_count} dataset(s)")
    print(f"Hub contains {len(ingester.hub)} samples")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

