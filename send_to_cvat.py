#!/usr/bin/env python3

import hydra
from functools import reduce
import operator
from omegaconf import DictConfig, OmegaConf

import fiftyone as fo
from fiftyone import ViewField as F


class CVATSender:
    """CVAT annotation workflow integration."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name not in fo.list_datasets():
            raise ValueError(f"Hub '{self.hub_name}' not found")
        
        self.hub = fo.load_dataset(self.hub_name)
        print(f"Loaded hub: {self.hub_name} ({len(self.hub)} samples)")
    
    def send(self, cvat_cfg: DictConfig):
        if cvat_cfg.get('load_annotations', False):
            self._load_annotations(cvat_cfg)
        else:
            self._send_to_cvat(cvat_cfg)
    
    def _send_to_cvat(self, cvat_cfg: DictConfig):
        print(f"\n{'=' * 80}")
        print(f"Sending to CVAT: {cvat_cfg.name}")
        print(f"{'=' * 80}")
        
        view = self._build_filtered_view(cvat_cfg)
        print(f"After filtering: {len(view)} samples")
        
        if len(view) == 0:
            print("No samples to send")
            return
        
        if cvat_cfg.get('subsample_configs'):
            view = self._subsample_by_dataset(view, cvat_cfg)
            print(f"After subsampling: {len(view)} samples")
        
        anno_key = cvat_cfg.anno_key
        label_field = cvat_cfg.label_field
        
        cvat_params = {
            'backend': 'cvat',
            'label_field': label_field,
            'launch_editor': cvat_cfg.get('launch_editor', False),
        }
        
        if cvat_cfg.get('classes'):
            cvat_params['classes'] = list(cvat_cfg.classes)
        
        if cvat_cfg.get('attributes'):
            cvat_params['attributes'] = dict(cvat_cfg.attributes)
        
        if cvat_cfg.get('label_type'):
            cvat_params['label_type'] = cvat_cfg.label_type
        
        cvat_params['allow_additions'] = cvat_cfg.get('allow_additions', True)
        cvat_params['allow_deletions'] = cvat_cfg.get('allow_deletions', True)
        cvat_params['allow_label_edits'] = cvat_cfg.get('allow_label_edits', True)
        cvat_params['allow_spatial_edits'] = cvat_cfg.get('allow_spatial_edits', True)
        
        if cvat_cfg.get('cvat_url'):
            cvat_params['url'] = cvat_cfg.cvat_url
        
        if cvat_cfg.get('task_size'):
            cvat_params['task_size'] = cvat_cfg.task_size
        
        if cvat_cfg.get('segment_size'):
            cvat_params['segment_size'] = cvat_cfg.segment_size
        
        if cvat_cfg.get('project_name'):
            cvat_params['project_name'] = cvat_cfg.project_name
        
        if cvat_cfg.get('task_assignee'):
            cvat_params['task_assignee'] = cvat_cfg.task_assignee
        
        if cvat_cfg.get('job_assignees'):
            cvat_params['job_assignees'] = list(cvat_cfg.job_assignees)
        
        if cvat_cfg.get('frame_start'):
            cvat_params['frame_start'] = cvat_cfg.frame_start
        
        if cvat_cfg.get('frame_stop'):
            cvat_params['frame_stop'] = cvat_cfg.frame_stop
        
        if cvat_cfg.get('frame_step'):
            cvat_params['frame_step'] = cvat_cfg.frame_step
        
        print(f"\nAnnotation parameters:")
        print(f"  Key: {anno_key}")
        print(f"  Label field: {label_field}")
        print(f"  Classes: {cvat_params.get('classes', 'all')}")
        print(f"  Allow additions: {cvat_params['allow_additions']}")
        print(f"  Allow deletions: {cvat_params['allow_deletions']}")
        print(f"  Allow label edits: {cvat_params['allow_label_edits']}")
        print(f"  Allow spatial edits: {cvat_params['allow_spatial_edits']}")
        
        view.annotate(anno_key, **cvat_params)
        
        print(f"\n{'=' * 80}")
        print(f"Sent to CVAT! Annotation key: {anno_key}")
        print(f"After annotating in CVAT, run:")
        print(f"  python send_to_cvat.py cvat={cvat_cfg.name} +cvat.load_annotations=true")
        if cvat_cfg.get('add_tag_on_import'):
            print(f"  (will add tag '{cvat_cfg.add_tag_on_import}' to samples)")
        print(f"{'=' * 80}")
    
    def _load_annotations(self, cvat_cfg: DictConfig):
        print(f"\n{'=' * 80}")
        print(f"Loading annotations: {cvat_cfg.name}")
        print(f"{'=' * 80}")
        
        anno_key = cvat_cfg.anno_key
        
        if anno_key not in self.hub.list_annotation_runs():
            print(f"Error: Annotation run '{anno_key}' not found")
            print(f"Available runs: {self.hub.list_annotation_runs()}")
            return
        
        dest_field = cvat_cfg.get('dest_field', None)
        if dest_field:
            print(f"Loading into field: {dest_field}")
            self.hub.load_annotations(anno_key, dest_field=dest_field)
        else:
            print(f"Loading into original field")
            self.hub.load_annotations(anno_key)
        
        print(f"Annotations loaded successfully!")
        
        if cvat_cfg.get('add_tag_on_import'):
            tag = cvat_cfg.add_tag_on_import
            print(f"\nAdding tag '{tag}' to annotated samples...")
            
            view = self.hub.load_annotation_view(anno_key)
            
            for sample in view:
                if tag not in sample.tags:
                    sample.tags.append(tag)
                    sample.save()
            
            print(f"Tag '{tag}' added to {len(view)} samples")
        
        if cvat_cfg.get('cleanup', False):
            print(f"\nCleaning up CVAT tasks...")
            results = self.hub.load_annotation_results(anno_key)
            results.cleanup()
            print(f"CVAT tasks deleted")
        
        if cvat_cfg.get('delete_run', False):
            print(f"\nDeleting annotation run record...")
            self.hub.delete_annotation_run(anno_key)
            print(f"Annotation run '{anno_key}' deleted")
        
        print(f"\n{'=' * 80}")
        print(f"Import complete!")
        print(f"{'=' * 80}")
    
    def _build_filtered_view(self, cvat_cfg: DictConfig) -> fo.DatasetView:
        view = self.hub
        
        if cvat_cfg.get('filter') and cvat_cfg.filter.get('dataset_tags'):
            tags = list(cvat_cfg.filter.dataset_tags)
            if len(tags) > 0:
                dataset_filter = [F("dataset_tag") == tag for tag in tags]
                if len(dataset_filter) == 1:
                    view = view.match(dataset_filter[0])
                else:
                    combined = reduce(operator.or_, dataset_filter)
                    view = view.match(combined)
                print(f"Filtered by dataset tags: {tags}")
        
        if cvat_cfg.get('filter') and cvat_cfg.filter.get('splits'):
            splits = list(cvat_cfg.filter.splits)
            if len(splits) > 0:
                splits_filter = F("split").is_in(splits)
                view = view.match(splits_filter)
                print(f"Filtered by splits: {splits}")
        
        if cvat_cfg.get('filter') and cvat_cfg.filter.get('status'):
            status = cvat_cfg.filter.status
            view = view.match(F("status") == status)
            print(f"Filtered by status: {status}")
        
        if cvat_cfg.get('filter') and cvat_cfg.filter.get('sample_tags'):
            sample_tags = list(cvat_cfg.filter.sample_tags)
            for tag in sample_tags:
                view = view.match_tags(tag)
            print(f"Filtered by sample tags: {sample_tags}")
        
        if cvat_cfg.get('filter') and cvat_cfg.filter.get('classes'):
            label_field = cvat_cfg.label_field
            classes = list(cvat_cfg.filter.classes)
            
            if label_field.startswith("frames."):
                field_name = label_field.split('.', 1)[1]
                view = view.match(
                    F(f"frames.{field_name}.detections").length() > 0
                )
            else:
                field_name = label_field
                view = view.filter_labels(
                    field_name, F("label").is_in(classes)
                )
                view = view.match(
                    F(f"{field_name}.detections").length() > 0
                )
            
            print(f"Filtered by classes: {classes}")
        
        if cvat_cfg.get('limit'):
            view = view.limit(cvat_cfg.limit)
            print(f"Limited to {cvat_cfg.limit} samples")
        
        return view
    
    def _subsample_by_dataset(self, view: fo.DatasetView,
                              cvat_cfg: DictConfig) -> fo.DatasetView:
        if not cvat_cfg.get('subsample_configs'):
            return view
        
        print(f"\nApplying frame subsampling...")
        all_sample_ids = []
        dataset_tags = view.distinct("dataset_tag")
        
        for dataset_tag in dataset_tags:
            dataset_view = view.match(F("dataset_tag") == dataset_tag)
            subsample_cfg = cvat_cfg.subsample_configs.get(dataset_tag)
            
            if subsample_cfg and subsample_cfg.get('enabled', False):
                frame_interval = subsample_cfg.frame_interval
                print(f"  {dataset_tag}: Every {frame_interval} frames")
                
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
                    print(f"    Video: {len(video_frames)} → {len(subsampled)}")
                
                if len(images) > 0:
                    all_sample_ids.extend(images.values("id"))
                    print(f"    Images: {len(images)} (all kept)")
            else:
                all_sample_ids.extend(dataset_view.values("id"))
                print(f"  {dataset_tag}: No subsampling ({len(dataset_view)} samples)")
        
        return self.hub.select(all_sample_ids)


@hydra.main(version_base=None, config_path="conf", config_name="config_cvat")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    sender = CVATSender(cfg)
    sender.send(cfg.cvat)


if __name__ == "__main__":
    main()

