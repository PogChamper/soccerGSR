#!/usr/bin/env python3

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import fiftyone as fo
from fiftyone import ViewField as F


class ModelEvaluator:
    """Model evaluation with metrics and reports."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hub_name = cfg.hub.name
        
        if self.hub_name not in fo.list_datasets():
            raise ValueError(f"Hub '{self.hub_name}' not found")
        
        self.hub = fo.load_dataset(self.hub_name)
        print(f"Loaded hub: {self.hub_name} ({len(self.hub)} samples)")
    
    def evaluate(self, eval_cfg: DictConfig):
        print(f"\n{'=' * 80}")
        print(f"Evaluation: {eval_cfg.name}")
        print(f"{'=' * 80}")
        
        view = self._build_view(eval_cfg)
        print(f"Samples to evaluate: {len(view)}")
        
        if len(view) == 0:
            print("No samples to evaluate")
            return
        
        results = self._run_evaluation(view, eval_cfg)
        self._generate_reports(results, view, eval_cfg)
        self._save_views(view, eval_cfg)
        
        print(f"\n{'=' * 80}")
        print(f"Evaluation complete! Results stored with key '{eval_cfg.eval_key}'")
        print(f"{'=' * 80}")
    
    def _build_view(self, eval_cfg: DictConfig):
        view = self.hub
        
        if hasattr(eval_cfg, 'filter'):
            if eval_cfg.filter.get('dataset_tags'):
                tags_filter = [
                    F("dataset_tag") == tag 
                    for tag in eval_cfg.filter.dataset_tags
                ]
                if len(tags_filter) == 1:
                    view = view.match(tags_filter[0])
                else:
                    from functools import reduce
                    import operator
                    combined = reduce(operator.or_, tags_filter)
                    view = view.match(combined)
                print(f"Filtered by dataset tags: {list(eval_cfg.filter.dataset_tags)}")
            
            if eval_cfg.filter.get('splits'):
                splits_filter = F("split").is_in(eval_cfg.filter.splits)
                view = view.match(splits_filter)
                print(f"Filtered by splits: {list(eval_cfg.filter.splits)}")
            
            if eval_cfg.filter.get('status'):
                view = view.match(F("status") == eval_cfg.filter.status)
                print(f"Filtered by status: {eval_cfg.filter.status}")
            
            if eval_cfg.filter.get('pred_field_exists'):
                view = view.exists(eval_cfg.pred_field)
                print(f"Filtered to samples with predictions")
        
        if eval_cfg.get('max_samples'):
            view = view.take(eval_cfg.max_samples)
            print(f"Limited to {eval_cfg.max_samples} samples")
        
        return view
    
    def _run_evaluation(self, view, eval_cfg: DictConfig):
        print(f"\nRunning evaluation...")
        print(f"Method: {eval_cfg.method}")
        print(f"Predictions: {eval_cfg.pred_field}")
        print(f"Ground truth: {eval_cfg.gt_field}")
        
        eval_params = {
            'pred_field': eval_cfg.pred_field,
            'gt_field': eval_cfg.gt_field,
            'eval_key': eval_cfg.eval_key,
            'method': eval_cfg.method,
            'compute_mAP': eval_cfg.get('compute_mAP', True),
        }
        
        if eval_cfg.get('iou'):
            eval_params['iou'] = eval_cfg.iou
        
        if eval_cfg.get('classwise') is not None:
            eval_params['classwise'] = eval_cfg.classwise
        
        if eval_cfg.get('iscrowd_attr'):
            eval_params['iscrowd'] = eval_cfg.iscrowd_attr
        
        results = view.evaluate_detections(**eval_params)
        
        return results
    
    def _generate_reports(self, results, view, eval_cfg: DictConfig):
        print(f"\nEvaluation metrics:")
        
        if eval_cfg.get('print_metrics', True):
            print(results.metrics())
        
        if eval_cfg.get('compute_mAP', True):
            print(f"\nmAP: {results.mAP():.4f}")
        
        if eval_cfg.get('print_report', True):
            classes = eval_cfg.get('report_classes')
            if classes:
                results.print_report(classes=classes)
            else:
                counts = view.count_values(f"{eval_cfg.gt_field}.detections.label")
                top_classes = sorted(counts, key=counts.get, reverse=True)[:10]
                if top_classes:
                    results.print_report(classes=top_classes)
        
        eval_key = eval_cfg.eval_key
        print(f"\nSample-level statistics:")
        print(f"  TP: {view.sum(f'{eval_key}_tp')}")
        print(f"  FP: {view.sum(f'{eval_key}_fp')}")
        print(f"  FN: {view.sum(f'{eval_key}_fn')}")
        
        if eval_cfg.get('plot_pr_curves', False):
            classes = eval_cfg.get('pr_curve_classes')
            if not classes:
                counts = view.count_values(f"{eval_cfg.gt_field}.detections.label")
                classes = sorted(counts, key=counts.get, reverse=True)[:5]
            
            if classes:
                print(f"\nGenerating PR curves for: {classes}")
                plot = results.plot_pr_curves(classes=classes)
                
                output_dir = Path(eval_cfg.get('output_dir', 'outputs'))
                output_dir.mkdir(exist_ok=True, parents=True)
                plot_path = output_dir / f"{eval_cfg.eval_key}_pr_curves.html"
                plot.write_html(str(plot_path))
                print(f"Saved PR curves to: {plot_path}")
        
        if eval_cfg.get('plot_confusion_matrix', False):
            classes = eval_cfg.get('confusion_matrix_classes')
            if not classes:
                counts = view.count_values(f"{eval_cfg.gt_field}.detections.label")
                classes = sorted(counts, key=counts.get, reverse=True)[:10]
            
            if classes:
                print(f"\nGenerating confusion matrix for: {classes}")
                try:
                    plot = results.plot_confusion_matrix(classes=classes)
                    
                    output_dir = Path(eval_cfg.get('output_dir', 'outputs'))
                    output_dir.mkdir(exist_ok=True, parents=True)
                    plot_path = output_dir / f"{eval_cfg.eval_key}_confusion_matrix.html"
                    
                    if hasattr(plot, 'write_html'):
                        plot.write_html(str(plot_path))
                        print(f"Saved confusion matrix to: {plot_path}")
                    elif hasattr(plot, '_figure'):
                        import plotly.graph_objects as go
                        fig = go.Figure(plot._figure)
                        fig.write_html(str(plot_path))
                        print(f"Saved confusion matrix to: {plot_path}")
                    else:
                        print(f"Could not save confusion matrix (interactive plot type)")
                except (ImportError, AttributeError) as e:
                    print(f"Skipping confusion matrix plot: {e}")
                    print("Note: Confusion matrix plots work best in Jupyter notebooks")
    
    def _save_views(self, view, eval_cfg: DictConfig):
        if not eval_cfg.get('save_views', True):
            return
        
        print(f"\nSaving evaluation views...")
        
        eval_key = eval_cfg.eval_key
        
        best_samples = view.sort_by(f"{eval_key}_tp", reverse=True).limit(
            eval_cfg.get('top_n', 50)
        )
        self.hub.save_view(f"{eval_key}_best_samples", best_samples)
        print(f"Saved view: {eval_key}_best_samples")
        
        worst_samples = view.sort_by(f"{eval_key}_fp", reverse=True).limit(
            eval_cfg.get('top_n', 50)
        )
        self.hub.save_view(f"{eval_key}_worst_samples", worst_samples)
        print(f"Saved view: {eval_key}_worst_samples")
        
        fp_view = view.filter_labels(eval_cfg.pred_field, F("eval") == "fp")
        self.hub.save_view(f"{eval_key}_false_positives", fp_view)
        print(f"Saved view: {eval_key}_false_positives")
        
        fn_view = view.filter_labels(eval_cfg.gt_field, F("eval") == "fn")
        self.hub.save_view(f"{eval_key}_false_negatives", fn_view)
        print(f"Saved view: {eval_key}_false_negatives")


@hydra.main(version_base=None, config_path="conf", config_name="config_evaluate")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    evaluator = ModelEvaluator(cfg)
    evaluator.evaluate(cfg.evaluate)


if __name__ == "__main__":
    main()

