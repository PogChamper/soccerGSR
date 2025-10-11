"""
Convert soccer tracking dataset from custom format to COCO, YOLO, and MOT formats.

This module provides converters to transform the Labels-GameState.json format
into standard object detection and tracking annotation formats.
"""

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BBoxImage:
    """Bounding box in image coordinates."""
    x: float
    y: float
    x_center: float
    y_center: float
    w: float
    h: float


@dataclass
class Annotation:
    """Single object annotation."""
    id: str
    image_id: str
    track_id: int
    category_id: int
    bbox: BBoxImage
    attributes: Dict[str, Optional[str]]


@dataclass
class ImageInfo:
    """Image metadata."""
    image_id: str
    file_name: str
    height: int
    width: int
    is_labeled: bool


@dataclass
class SequenceInfo:
    """Sequence metadata."""
    name: str
    seq_length: int
    frame_rate: int
    im_dir: str
    im_ext: str
    width: int
    height: int


class COCOConverter:
    """Convert annotations to COCO format."""
    
    CATEGORY_MAPPING = {
        1: {'id': 1, 'name': 'player', 'supercategory': 'object'},
        2: {'id': 2, 'name': 'goalkeeper', 'supercategory': 'object'},
        3: {'id': 3, 'name': 'referee', 'supercategory': 'object'},
    }
    
    def __init__(self) -> None:
        """Initialize COCO converter."""
        self.annotation_id = 1
        self.image_id_mapping: Dict[str, int] = {}
        self.current_image_id = 1
    
    def convert_sequence(
        self,
        sequence_data: Dict[str, Any],
        sequence_name: str,
        split: str,
        output_dir: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        copy_images: bool = False
    ) -> Dict[str, Any]:
        """
        Convert a single sequence to COCO format.
        
        Args:
            sequence_data: Parsed Labels-GameState.json data
            sequence_name: Name of the sequence (e.g., SNGS-060)
            split: Dataset split (train/valid/test)
            output_dir: Root output directory for COCO format
            source_dir: Source directory containing images (for copying)
            copy_images: Whether to copy image files
        
        Returns:
            Dictionary in COCO format
        """
        coco_data = {
            'info': {
                'description': f'Soccer tracking dataset - {split}',
                'version': sequence_data['info']['version'],
                'year': 2024,
                'contributor': 'Soccer Dataset',
            },
            'images': [],
            'annotations': [],
            'categories': list(self.CATEGORY_MAPPING.values()),
        }
        
        # Create images directory if copying images
        if copy_images and output_dir is not None:
            images_dir = output_dir / 'images' / split / sequence_name / 'img1'
            images_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        for img_data in sequence_data['images']:
            if not img_data['is_labeled']:
                continue
            
            image_id = self._get_or_create_image_id(img_data['image_id'])
            coco_image = {
                'id': image_id,
                'file_name': f"{split}/{sequence_name}/img1/{img_data['file_name']}",
                'height': img_data['height'],
                'width': img_data['width'],
            }
            coco_data['images'].append(coco_image)
            
            # Copy image if requested
            if copy_images and output_dir is not None and source_dir is not None:
                source_image = source_dir / sequence_name / 'img1' / img_data['file_name']
                target_image = images_dir / img_data['file_name']
                if source_image.exists() and not target_image.exists():
                    shutil.copy2(source_image, target_image)
        
        # Process annotations
        for ann_data in sequence_data['annotations']:
            # Skip pitch annotations
            if ann_data.get('supercategory') == 'pitch':
                continue
            
            category_id = ann_data['category_id']
            
            # Check if player is a goalkeeper based on attributes
            attributes = ann_data.get('attributes', {})
            if category_id == 1 and attributes.get('role') == 'goalkeeper':
                category_id = 2  # Change to goalkeeper class
            
            if category_id not in self.CATEGORY_MAPPING:
                continue
            
            image_id = self._get_or_create_image_id(ann_data['image_id'])
            bbox_data = ann_data['bbox_image']
            
            coco_annotation = {
                'id': self.annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [
                    bbox_data['x'],
                    bbox_data['y'],
                    bbox_data['w'],
                    bbox_data['h']
                ],
                'area': bbox_data['w'] * bbox_data['h'],
                'iscrowd': 0,
                'track_id': ann_data['track_id'],
                'attributes': ann_data.get('attributes', {}),
            }
            coco_data['annotations'].append(coco_annotation)
            self.annotation_id += 1
        
        return coco_data
    
    def _get_or_create_image_id(self, original_id: str) -> int:
        """Map string image ID to integer ID."""
        if original_id not in self.image_id_mapping:
            self.image_id_mapping[original_id] = self.current_image_id
            self.current_image_id += 1
        return self.image_id_mapping[original_id]


class YOLOConverter:
    """Convert annotations to YOLO format."""
    
    CATEGORY_TO_CLASS = {
        1: 0,  # player
        2: 1,  # goalkeeper
        3: 2,  # referee
    }
    
    CLASS_NAMES = ['player', 'goalkeeper', 'referee']
    
    def convert_sequence(
        self,
        sequence_data: Dict[str, Any],
        output_dir: Path,
        sequence_name: str,
        source_dir: Optional[Path] = None,
        copy_images: bool = False
    ) -> None:
        """
        Convert a single sequence to YOLO format.
        
        Args:
            sequence_data: Parsed Labels-GameState.json data
            output_dir: Output directory for YOLO labels
            sequence_name: Name of the sequence
            source_dir: Source directory containing images (for copying)
            copy_images: Whether to copy image files
        """
        # Create directory for this sequence
        labels_dir = output_dir / sequence_name / 'labels'
        images_dir = output_dir / sequence_name / 'images'
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Group annotations by image
        annotations_by_image: Dict[str, List[Dict[str, Any]]] = {}
        for ann in sequence_data['annotations']:
            if ann.get('supercategory') == 'pitch':
                continue
            
            category_id = ann['category_id']
            
            # Check if player is a goalkeeper based on attributes
            attributes = ann.get('attributes', {})
            if category_id == 1 and attributes.get('role') == 'goalkeeper':
                category_id = 2  # Change to goalkeeper class
                ann = ann.copy()  # Create a copy to avoid modifying original
                ann['category_id'] = category_id
            
            if category_id not in self.CATEGORY_TO_CLASS:
                continue
            
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Create label files
        for img_data in sequence_data['images']:
            if not img_data['is_labeled']:
                continue
            
            image_id = img_data['image_id']
            file_name = Path(img_data['file_name']).stem
            label_file = labels_dir / f"{file_name}.txt"
            
            width = img_data['width']
            height = img_data['height']
            
            lines = []
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    bbox = ann['bbox_image']
                    class_id = self.CATEGORY_TO_CLASS[ann['category_id']]
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = bbox['x_center'] / width
                    y_center = bbox['y_center'] / height
                    w = bbox['w'] / width
                    h = bbox['h'] / height
                    
                    lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Write label file (empty if no annotations)
            label_file.write_text('\n'.join(lines) + '\n' if lines else '')
            
            # Copy image if requested
            if copy_images and source_dir is not None:
                source_image = source_dir / sequence_name / 'img1' / img_data['file_name']
                target_image = images_dir / img_data['file_name']
                if source_image.exists() and not target_image.exists():
                    shutil.copy2(source_image, target_image)
    
    @staticmethod
    def create_data_yaml(output_dir: Path, splits: List[str]) -> None:
        """
        Create data.yaml file for YOLO training.
        
        Args:
            output_dir: Root output directory
            splits: List of splits to include (e.g., ['train', 'valid'])
        """
        yaml_lines = [
            '# Soccer tracking dataset',
            f'path: {output_dir.absolute()}',
            'train: train',
            'val: valid',
            'test: test',
            '',
            '# Classes',
            'names:',
            '  0: player',
            '  1: goalkeeper',
            '  2: referee',
            '',
            'nc: 3',
        ]
        yaml_content = '\n'.join(yaml_lines) + '\n'
        
        yaml_file = output_dir / 'data.yaml'
        yaml_file.write_text(yaml_content)
        logger.info(f"Created {yaml_file}")


class MOTConverter:
    """Convert annotations to MOT Challenge format."""
    
    def convert_sequence(
        self,
        sequence_data: Dict[str, Any],
        output_dir: Path,
        sequence_name: str,
        source_dir: Optional[Path] = None,
        copy_images: bool = False
    ) -> None:
        """
        Convert a single sequence to MOT format.
        
        Args:
            sequence_data: Parsed Labels-GameState.json data
            output_dir: Output directory for MOT labels
            sequence_name: Name of the sequence
            source_dir: Source directory containing images (for copying)
            copy_images: Whether to copy image files
        """
        # Create sequence directory structure
        seq_dir = output_dir / sequence_name
        gt_dir = seq_dir / 'gt'
        img_dir = seq_dir / 'img1'
        gt_dir.mkdir(parents=True, exist_ok=True)
        if copy_images:
            img_dir.mkdir(parents=True, exist_ok=True)
        
        # Group annotations by image (only humans: players, goalkeepers and referees)
        annotations_by_image: Dict[str, List[Dict[str, Any]]] = {}
        for ann in sequence_data['annotations']:
            if ann.get('supercategory') == 'pitch':
                continue
            
            category_id = ann['category_id']
            
            # Check if player is a goalkeeper based on attributes
            attributes = ann.get('attributes', {})
            if category_id == 1 and attributes.get('role') == 'goalkeeper':
                category_id = 2  # Change to goalkeeper class
                ann = ann.copy()  # Create a copy to avoid modifying original
                ann['category_id'] = category_id
            
            # Only include players (1), goalkeepers (2) and referees (3), skip ball (4)
            if category_id not in [1, 2, 3]:
                continue
            
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Create gt.txt in MOT format
        gt_lines = []
        for img_data in sequence_data['images']:
            if not img_data['is_labeled']:
                continue
            
            # Extract frame number from file name
            frame_num = int(Path(img_data['file_name']).stem)
            image_id = img_data['image_id']
            
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    bbox = ann['bbox_image']
                    track_id = ann['track_id']
                    
                    # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                    # conf=1 for ground truth, x/y/z=-1 (not used)
                    line = (
                        f"{frame_num},{track_id},{bbox['x']:.2f},{bbox['y']:.2f},"
                        f"{bbox['w']:.2f},{bbox['h']:.2f},1,-1,-1,-1"
                    )
                    gt_lines.append(line)
        
        # Write gt.txt
        gt_file = gt_dir / 'gt.txt'
        gt_file.write_text('\n'.join(sorted(gt_lines)) + '\n')
        
        # Create seqinfo.ini
        info = sequence_data['info']
        seqinfo_lines = [
            '[Sequence]',
            f'name={sequence_name}',
            'imDir=img1',
            f'frameRate={info["frame_rate"]}',
            f'seqLength={info["seq_length"]}',
            f'imWidth={sequence_data["images"][0]["width"]}',
            f'imHeight={sequence_data["images"][0]["height"]}',
            f'imExt={info["im_ext"]}',
        ]
        seqinfo_content = '\n'.join(seqinfo_lines) + '\n'
        
        seqinfo_file = seq_dir / 'seqinfo.ini'
        seqinfo_file.write_text(seqinfo_content)
        
        # Copy images if requested
        if copy_images and source_dir is not None:
            source_img_dir = source_dir / sequence_name / 'img1'
            if source_img_dir.exists():
                for img_data in sequence_data['images']:
                    source_image = source_img_dir / img_data['file_name']
                    target_image = img_dir / img_data['file_name']
                    if source_image.exists() and not target_image.exists():
                        shutil.copy2(source_image, target_image)
        
        logger.info(f"Created MOT annotations for {sequence_name}")


class DatasetConverter:
    """Main converter orchestrating all format conversions."""
    
    def __init__(self, data_root: Path) -> None:
        """
        Initialize dataset converter.
        
        Args:
            data_root: Root directory containing train/valid/test folders
        """
        self.data_root = data_root
        self.coco_converter = COCOConverter()
        self.yolo_converter = YOLOConverter()
        self.mot_converter = MOTConverter()
    
    def convert_all(
        self,
        output_dir: Path,
        splits: Optional[List[str]] = None,
        formats: Optional[Set[str]] = None,
        copy_images: bool = False
    ) -> None:
        """
        Convert all splits to specified formats.
        
        Args:
            output_dir: Root directory for outputs
            splits: List of splits to process (default: ['train', 'valid', 'test'])
            formats: Set of formats to generate (default: {'coco', 'yolo', 'mot'})
            copy_images: Whether to copy image files to output directories
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
        
        if formats is None:
            formats = {'coco', 'yolo', 'mot'}
        
        for split in splits:
            logger.info(f"Processing {split} split...")
            self._convert_split(split, output_dir, formats, copy_images)
        
        # Create YOLO data.yaml if YOLO format was selected
        if 'yolo' in formats:
            YOLOConverter.create_data_yaml(output_dir / 'yolo', splits)
        
        logger.info("Conversion complete!")
    
    def _convert_split(
        self,
        split: str,
        output_dir: Path,
        formats: Set[str],
        copy_images: bool
    ) -> None:
        """
        Convert a single split to specified formats.
        
        Args:
            split: Split name (train/valid/test)
            output_dir: Root output directory
            formats: Set of formats to generate
            copy_images: Whether to copy image files
        """
        split_dir = self.data_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist, skipping")
            return
        
        # Create output directories for selected formats
        coco_dir = output_dir / 'coco'
        yolo_dir = output_dir / 'yolo' / split
        mot_dir = output_dir / 'mot' / split
        
        if 'coco' in formats:
            coco_dir.mkdir(parents=True, exist_ok=True)
        if 'yolo' in formats:
            yolo_dir.mkdir(parents=True, exist_ok=True)
        if 'mot' in formats:
            mot_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all sequences
        sequence_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        # Process each sequence
        all_coco_data: Dict[str, Any] = {
            'info': {
                'description': f'Soccer tracking dataset - {split}',
                'version': '1.3',
                'year': 2024,
                'contributor': 'Soccer Dataset',
            },
            'images': [],
            'annotations': [],
            'categories': list(COCOConverter.CATEGORY_MAPPING.values()),
        }
        
        for seq_dir in sequence_dirs:
            sequence_name = seq_dir.name
            labels_file = seq_dir / 'Labels-GameState.json'
            
            if not labels_file.exists():
                logger.warning(f"Labels file not found for {sequence_name}, skipping")
                continue
            
            try:
                with open(labels_file, 'r') as f:
                    sequence_data = json.load(f)
                
                logger.info(f"Converting {sequence_name}...")
                
                # Convert to COCO (accumulate)
                if 'coco' in formats:
                    seq_coco = self.coco_converter.convert_sequence(
                        sequence_data, sequence_name, split,
                        output_dir=coco_dir, source_dir=split_dir, copy_images=copy_images
                    )
                    all_coco_data['images'].extend(seq_coco['images'])
                    all_coco_data['annotations'].extend(seq_coco['annotations'])
                
                # Convert to YOLO
                if 'yolo' in formats:
                    self.yolo_converter.convert_sequence(
                        sequence_data, yolo_dir, sequence_name,
                        source_dir=split_dir, copy_images=copy_images
                    )
                
                # Convert to MOT
                if 'mot' in formats:
                    self.mot_converter.convert_sequence(
                        sequence_data, mot_dir, sequence_name,
                        source_dir=split_dir, copy_images=copy_images
                    )
                
            except Exception as e:
                logger.error(f"Error processing {sequence_name}: {e}", exc_info=True)
        
        # Save combined COCO file for the split
        if 'coco' in formats:
            coco_file = coco_dir / f'{split}.json'
            with open(coco_file, 'w') as f:
                json.dump(all_coco_data, f, indent=2)
            logger.info(f"Saved COCO annotations to {coco_file}")


def main() -> None:
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(
        description='Convert soccer tracking dataset to COCO, YOLO, and MOT formats')
    
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data'),
        help='Root directory containing train/valid/test folders (default: data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('converted_labels'),
        help='Output directory for converted labels (default: converted_labels)'
    )
    
    parser.add_argument(
        '--format',
        choices=['coco', 'yolo', 'mot', 'all'],
        nargs='+',
        default=['all'],
        help='Output format(s) to generate (default: all)'
    )
    
    parser.add_argument(
        '--splits',
        choices=['train', 'valid', 'test'],
        nargs='+',
        default=['train', 'valid', 'test'],
        help='Dataset splits to process (default: train valid test)'
    )
    
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy image files to output directories (COCO: images/, YOLO: images/, MOT: img1/)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' format
    if 'all' in args.format:
        formats = {'coco', 'yolo', 'mot'}
    else:
        formats = set(args.format)
    
    # Log configuration
    logger.info("Starting dataset conversion...")
    logger.info(f"Input directory: {args.data_root}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Formats: {', '.join(sorted(formats))}")
    logger.info(f"Splits: {', '.join(args.splits)}")
    logger.info(f"Copy images: {args.copy_images}")
    
    # Run converter
    converter = DatasetConverter(args.data_root)
    converter.convert_all(
        args.output_dir,
        splits=args.splits,
        formats=formats,
        copy_images=args.copy_images
    )
    
    logger.info("All conversions completed successfully!")


if __name__ == '__main__':
    main()

