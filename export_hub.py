#!/usr/bin/env python3

import json
import shutil
from pathlib import Path
from typing import List

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

import fiftyone as fo
from fiftyone import ViewField as F


class UnifiedExporter:
    """Export filtered subsets of the hub in YOLO/COCO formats."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name not in fo.list_datasets():
            raise ValueError(
                f"Hub '{self.hub_name}' not found. Run ingestion first."
            )
        
        self.hub = fo.load_dataset(self.hub_name)
        print(f"Loaded hub: {self.hub_name} ({len(self.hub)} samples)")
    
    def export(self, export_cfg: DictConfig):
        """Execute export pipeline based on configuration."""
        export_name = export_cfg.name
        output_dir = Path(export_cfg.output_dir)
        label_field = export_cfg.get('label_field', 'detections')
        
        print(f"\n{'=' * 80}")
        print(f"Export: {export_name}")
        print(f"Label field: {label_field}")
        print(f"{'=' * 80}")
        
        print(f"\n[1/5] Building filtered view...")
        view = self._build_filtered_view(export_cfg)
        print(f"Filtered samples: {len(view)}")
        
        print(f"\n[2/5] Subsampling video frames...")
        view = self._subsample_by_dataset(view, export_cfg)
        print(f"After subsampling: {len(view)}")
        
        print(f"\n[3/5] Filtering classes...")
        view = self._filter_classes(view, export_cfg.classes, label_field)
        print(f"After class filtering: {len(view)}")
        
        print(f"\n[4/5] Splitting data...")
        splits = self._split_data(view, export_cfg.splits)
        for split_name, split_view in splits.items():
            print(f"{split_name}: {len(split_view)} samples")
        
        print(f"\n[5/5] Exporting to formats...")
        self._export_formats(export_name, splits, export_cfg, output_dir, label_field)
        
        print(f"\n{'=' * 80}")
        print(f"Export Complete!")
        print(f"Output directory: {output_dir / export_name}")
        print(f"{'=' * 80}")
    
    def _build_filtered_view(self, export_cfg: DictConfig) -> fo.DatasetView:
        """Build filtered view based on dataset tags and status."""
        view = self.hub
        
        if export_cfg.dataset_tags:
            from functools import reduce
            import operator
            
            dataset_filter = [
                F("dataset_tag") == tag for tag in export_cfg.dataset_tags
            ]
            
            if len(dataset_filter) == 1:
                view = view.match(dataset_filter[0])
            else:
                combined = reduce(operator.or_, dataset_filter)
                view = view.match(combined)
            
            print(f"Filtered by datasets: {list(export_cfg.dataset_tags)}")
        
        if hasattr(export_cfg, 'status') and export_cfg.status:
            view = view.match(F("status") == export_cfg.status)
            print(f"Filtered by status: {export_cfg.status}")
        
        # Filter by sample tags
        if hasattr(export_cfg, 'sample_tags') and export_cfg.sample_tags:
            sample_tags = list(export_cfg.sample_tags)
            for tag in sample_tags:
                view = view.match_tags(tag)
            print(f"Filtered by sample tags: {sample_tags}")
        
        return view
    
    def _subsample_by_dataset(self, view: fo.DatasetView,
                              export_cfg: DictConfig) -> fo.DatasetView:
        """Subsample video frames per dataset tag."""
        if not export_cfg.get('subsample_configs'):
            print(f"No subsampling configured")
            return view
        
        all_sample_ids = []
        dataset_tags = view.distinct("dataset_tag")
        
        for dataset_tag in dataset_tags:
            dataset_view = view.match(F("dataset_tag") == dataset_tag)
            subsample_cfg = export_cfg.subsample_configs.get(dataset_tag)
            
            if subsample_cfg and subsample_cfg.get('enabled', False):
                frame_interval = subsample_cfg.frame_interval
                print(f"{dataset_tag}: Every {frame_interval} frames")
                
                video_frames = dataset_view.match(F("type") == "video_frame")
                images = dataset_view.match(F("type") == "image")
                
                if len(video_frames) > 0:
                    subsampled = video_frames.match(
                        (F("frame_number") % frame_interval) == 1
                    )
                    
                    if len(subsampled) == 0:
                        subsampled = video_frames.match(
                            (F("frame_number") % frame_interval) == 0
                        )
                    
                    all_sample_ids.extend(subsampled.values("id"))
                    print(f"Video: {len(video_frames)} → {len(subsampled)}")
                
                if len(images) > 0:
                    all_sample_ids.extend(images.values("id"))
                    print(f"Images: {len(images)} (all kept)")
            else:
                all_sample_ids.extend(dataset_view.values("id"))
                print(f"{dataset_tag}: No subsampling "
                      f"({len(dataset_view)} samples)")
        
        return self.hub.select(all_sample_ids)
    
    def _filter_classes(self, view: fo.DatasetView,
                        classes: List[str], label_field: str = "detections") -> fo.DatasetView:
        """Filter detections to specified classes and remove empty samples."""
        if not classes or classes == ['all']:
            print(f"Keeping all classes")
            return view
        
        print(f"Keeping classes: {classes}")
        
        # Extract the actual field name (remove 'frames.' prefix if present)
        field_name = label_field.split('.')[-1]
        
        filtered_view = view.filter_labels(
            label_field, F("label").is_in(classes)
        )
        filtered_view = filtered_view.match(
            F(f"{field_name}.detections").length() > 0
        )
        
        print(f"Samples with selected classes: {len(filtered_view)}")
        
        return filtered_view
    
    def _split_data(self, view: fo.DatasetView,
                    split_config: List[str]) -> dict:
        """Split data by train/val/test."""
        splits = {}
        split_name_map = {'valid': 'val', 'validation': 'val'}
        
        for split_name in split_config:
            split_view = view.match(F("split") == split_name)
            if len(split_view) > 0:
                output_name = split_name_map.get(split_name, split_name)
                splits[output_name] = split_view
        
        return splits
    
    def _export_formats(self, export_name: str, splits: dict,
                        export_cfg: DictConfig, output_dir: Path, label_field: str = "detections"):
        """Export to configured formats (YOLO/COCO)."""
        formats = export_cfg.formats
        classes = list(export_cfg.classes)
        
        for format_name in formats:
            print(f"\nExporting to {format_name.upper()} format...")
            
            if format_name == "yolo":
                self._export_yolo(export_name, splits, classes, output_dir, label_field)
            elif format_name == "coco":
                self._export_coco(export_name, splits, classes, output_dir, label_field)
            else:
                print(f"  Unknown format: {format_name}")
    
    def _export_yolo(self, export_name: str, splits: dict,
                     classes: List[str], output_dir: Path, label_field: str = "detections"):
        """Export to YOLO format with class remapping."""
        export_dir = output_dir / export_name / "yolo"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Class mapping:")
        for idx, cls in enumerate(classes):
            print(f"{idx}: {cls}")
        
        for split_name, split_view in splits.items():
            if len(split_view) == 0:
                continue
            
            split_dir = export_dir / split_name
            print(f"\n  Exporting {split_name} ({len(split_view)} samples)...")
            
            split_view.export(
                export_dir=str(split_dir),
                dataset_type=fo.types.YOLOv5Dataset,
                classes=classes,
                label_field=label_field
            )
            
            self._flatten_yolo_structure(split_dir)
        
        self._create_yolo_yaml(export_dir, classes, list(splits.keys()))
        print(f"\nYOLO export complete: {export_dir}")
        print(f"Classes: {classes} (IDs: 0-{len(classes)-1})")
    
    def _flatten_yolo_structure(self, split_dir: Path):
        """Flatten YOLO directory structure (remove extra nesting)."""
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        for base_dir in [images_dir, labels_dir]:
            if not base_dir.exists():
                continue
            
            for subdir in base_dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.iterdir():
                        if file.is_file():
                            dest = base_dir / file.name
                            if dest.exists():
                                counter = 1
                                while dest.exists():
                                    stem = file.stem
                                    suffix = file.suffix
                                    dest = base_dir / f"{stem}_{counter}{suffix}"
                                    counter += 1
                            shutil.move(str(file), str(dest))
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass
        
        print(f"Flattened directory structure")
        
    def _create_yolo_yaml(self, export_dir: Path, classes: List[str],
                          splits: List[str]):
        """Create data.yaml configuration file for YOLO."""
        data_yaml = {
            'path': str(export_dir.absolute()),
        }
        
        for split in splits:
            data_yaml[split] = split
        
        data_yaml['names'] = {idx: cls for idx, cls in enumerate(classes)}
        
        yaml_path = export_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    def _export_coco(self, export_name: str, splits: dict,
                     classes: List[str], output_dir: Path, label_field: str = "detections"):
        """Export to COCO format with class remapping (1-indexed)."""
        export_dir = output_dir / export_name / "coco"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = export_dir / "images"
        annotations_dir = export_dir / "annotations"
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        
        print(f"\nClass mapping:")
        for idx, cls in enumerate(classes):
            print(f"{idx + 1}: {cls}")
        
        for split_name, split_view in splits.items():
            if len(split_view) == 0:
                continue
            
            temp_dir = export_dir / f"_temp_{split_name}"
            print(f"\nExporting {split_name} ({len(split_view)} samples)...")
            
            split_view.export(
                export_dir=str(temp_dir),
                dataset_type=fo.types.COCODetectionDataset,
                classes=classes,
                label_field=label_field
            )
            
            self._restructure_coco(temp_dir, images_dir, annotations_dir, split_name)
        
        print(f"\nCOCO export complete: {export_dir}")
        print(f"Classes: {classes} (category_ids: 1-{len(classes)})")
    
    def _restructure_coco(self, temp_dir: Path, images_dir: Path,
                          annotations_dir: Path, split_name: str):
        """Restructure COCO export to standard format."""
        source_images = temp_dir / "data"
        source_labels = temp_dir / "labels.json"
        
        if not source_images.exists() or not source_labels.exists():
            print(f"Warning: Could not find COCO export files in {temp_dir}")
            return
        
        target_images = images_dir / split_name
        target_images.mkdir(exist_ok=True)
        
        for img_file in source_images.iterdir():
            if img_file.is_file():
                shutil.copy2(str(img_file), str(target_images / img_file.name))
        
        target_json = annotations_dir / f"instances_{split_name}.json"
        with open(source_labels, 'r') as f:
            data = json.load(f)
        
        for img in data.get('images', []):
            img['file_name'] = img['file_name'].split('/')[-1]
        
        with open(target_json, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        shutil.rmtree(temp_dir)
        print(f"COCO format: images/{split_name}/, annotations/instances_{split_name}.json")
    

@hydra.main(version_base=None, config_path="conf", config_name="config_export")
def main(cfg: DictConfig):
    """Main entry point for dataset export."""
    print(OmegaConf.to_yaml(cfg))
    
    exporter = UnifiedExporter(cfg)
    
    if 'export' not in cfg:
        raise ValueError("No export configuration found. Check your config files.")
    
    exporter.export(cfg.export)


if __name__ == "__main__":
    main()
