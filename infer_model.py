#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import fiftyone as fo
from fiftyone import ViewField as F


class ModelInferencer:
    """Model inference on FiftyOne datasets."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name not in fo.list_datasets():
            raise ValueError(f"Hub '{self.hub_name}' not found")
        
        self.hub = fo.load_dataset(self.hub_name)
        print(f"Loaded hub: {self.hub_name} ({len(self.hub)} samples)")
    
    def infer(self, infer_cfg: DictConfig):
        print(f"\n{'=' * 80}")
        print(f"Inference: {infer_cfg.name}")
        print(f"{'=' * 80}")
        
        view = self._build_view(infer_cfg)
        print(f"Samples to process: {len(view)}")
        
        if len(view) == 0:
            print("No samples to process")
            return
        
        model = self._load_model(infer_cfg)
        self._run_inference(view, model, infer_cfg)
        
        print(f"\n{'=' * 80}")
        print(f"Inference complete! Predictions stored in '{infer_cfg.pred_field}'")
        print(f"{'=' * 80}")
    
    def _build_view(self, infer_cfg: DictConfig):
        view = self.hub
        
        if hasattr(infer_cfg, 'filter'):
            if infer_cfg.filter.get('dataset_tags'):
                tags_filter = [
                    F("dataset_tag") == tag 
                    for tag in infer_cfg.filter.dataset_tags
                ]
                if len(tags_filter) == 1:
                    view = view.match(tags_filter[0])
                else:
                    from functools import reduce
                    import operator
                    combined = reduce(operator.or_, tags_filter)
                    view = view.match(combined)
                print(f"Filtered by dataset tags: {list(infer_cfg.filter.dataset_tags)}")
            
            if infer_cfg.filter.get('splits'):
                splits_filter = F("split").is_in(infer_cfg.filter.splits)
                view = view.match(splits_filter)
                print(f"Filtered by splits: {list(infer_cfg.filter.splits)}")
            
            if infer_cfg.filter.get('status'):
                view = view.match(F("status") == infer_cfg.filter.status)
                print(f"Filtered by status: {infer_cfg.filter.status}")
        
        if infer_cfg.get('max_samples'):
            view = view.take(infer_cfg.max_samples)
            print(f"Limited to {infer_cfg.max_samples} samples")
        
        return view
    
    def _load_model(self, infer_cfg: DictConfig):
        model_type = infer_cfg.model.type
        model_path = infer_cfg.model.path
        
        print(f"\nLoading model: {model_type}")
        print(f"Model path: {model_path}")
        
        if model_type == "onnx":
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            return session
        
        if model_type == "ultralytics":
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model

        elif model_type == "fiftyone_zoo":
            import fiftyone.zoo as foz
            model = foz.load_zoo_model(model_path)
            return model
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _run_inference(self, view, model, infer_cfg: DictConfig):
        model_type = infer_cfg.model.type
        pred_field = infer_cfg.pred_field
        
        if model_type == "fiftyone_zoo":
            view.apply_model(model, pred_field)
            return        
        elif model_type == "onnx":
            self._infer_onnx(view, model, infer_cfg)
        
        elif model_type == "ultralytics":
            self._infer_ultralytics(view, model, infer_cfg)

    
    def _infer_ultralytics(self, view, model, infer_cfg: DictConfig):
        pred_field = infer_cfg.pred_field
        conf_thresh = infer_cfg.model.get('conf_threshold', 0.25)
        img_size = infer_cfg.model.get('img_size', 640)
        class_map = infer_cfg.model.get('class_map', {})
        
        for sample in tqdm(view, desc="Inference"):
            results = model.predict(
                sample.filepath,
                conf=conf_thresh,
                imgsz=img_size,
                verbose=False
            )
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    xyxyn = boxes.xyxyn[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxyn
                    w, h = x2 - x1, y2 - y1
                    
                    cls_id = int(boxes.cls[i].item())
                    label = class_map.get(cls_id, model.names[cls_id])
                    conf = float(boxes.conf[i].item())
                    
                    detections.append(fo.Detection(
                        label=label,
                        bounding_box=[x1, y1, w, h],
                        confidence=conf
                    ))
            
            sample[pred_field] = fo.Detections(detections=detections)
            sample.save()
    
    def _infer_onnx(self, view, model, infer_cfg: DictConfig):
        import cv2
        
        pred_field = infer_cfg.pred_field
        conf_thresh = infer_cfg.model.get('conf_threshold', 0.25)
        nms_thresh = infer_cfg.model.get('nms_threshold', 0.45)
        img_size = infer_cfg.model.get('img_size', 640)
        class_map = infer_cfg.model.get('class_map', {})
        
        input_name = model.get_inputs()[0].name
        
        for sample in tqdm(view, desc="Inference"):
            img = cv2.imread(sample.filepath)
            img_h, img_w = img.shape[:2]
            
            img_resized = cv2.resize(img, (img_size, img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
            
            outputs = model.run(None, {input_name: img_input})
            predictions = outputs[0]
            
            boxes = []
            scores = []
            class_ids = []
            
            if predictions.shape[1] == 6:
                for pred in predictions[0]:
                    x1, y1, x2, y2, conf, cls_id = pred
                    if conf < conf_thresh:
                        continue
                    
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(float(conf))
                    class_ids.append(int(cls_id))
            else:
                predictions = predictions.squeeze()
                if len(predictions.shape) == 2:
                    predictions = predictions.T
                
                for i in range(predictions.shape[0]):
                    pred = predictions[i]
                    x_center, y_center, w, h = pred[:4]
                    
                    class_scores = pred[4:]
                    conf = np.max(class_scores)
                    
                    if conf < conf_thresh:
                        continue
                    
                    cls_id = int(np.argmax(class_scores))
                    
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    
                    boxes.append([x1, y1, w, h])
                    scores.append(float(conf))
                    class_ids.append(cls_id)
            
            detections = []
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
                
                if len(indices) > 0:
                    for idx in indices.flatten():
                        box = boxes[idx]
                        x1, y1, w, h = box
                        
                        x1_norm = x1 / img_size
                        y1_norm = y1 / img_size
                        w_norm = w / img_size
                        h_norm = h / img_size
                        
                        label = class_map.get(class_ids[idx], str(class_ids[idx]))
                        
                        detections.append(fo.Detection(
                            label=label,
                            bounding_box=[x1_norm, y1_norm, w_norm, h_norm],
                            confidence=scores[idx]
                        ))
            
            sample[pred_field] = fo.Detections(detections=detections)
            sample.save()


@hydra.main(version_base=None, config_path="conf", config_name="config_infer")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    inferencer = ModelInferencer(cfg)
    inferencer.infer(cfg.infer)


if __name__ == "__main__":
    main()

