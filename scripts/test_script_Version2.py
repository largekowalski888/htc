#!/usr/bin/env python3
"""
HTC Framework Test Script: Complete ML Workflow with Confusion Matrix
=====================================================================

This script demonstrates the complete workflow for:
1. Loading datasets from local/NextCloud sources
2. Defining training/validation/testing sets
3. Generating confusion matrices
4. Computing performance metrics
5. Parameter variation and explainability

Based on the htc framework tutorials (General.ipynb and ConfusionMatrix.ipynb)

FIRST: Set the path to the downloaded NextCloud dataset
python test_script_Version2.py \
    --data-dir C:/Cuyler's Work/DKFZ 2026 Work/htc/MLData/data/Cat_Pig/Cat_atlas/Cat_0007_kidney/data \
    --output-dir ./results
    
SECOND: Use instructions in cmd window to run:
python test_script_Version2.py \
    --data-dir /your/local/path \
    --output-dir ./results \
    --limit 10  # Test with 10 images first
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional
import argparse

# HTC Framework imports
from htc import Config, DataPath, DatasetImage, LabelMapping, settings
from htc.evaluation.metrics.scores import normalize_grouped_cm
from htc.models.common.MetricAggregationClassification import MetricAggregationClassification
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.helper_functions import sort_labels, sort_labels_cm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HTCWorkflow:
    """Main workflow class for ML pipeline with confusion matrix generation"""
    
    def __init__(self, config_path: str, data_source: str = "local", verbose: bool = True):
        """
        Initialize HTC Workflow
        
        Args:
            config_path: Path to config JSON file or model run directory
            data_source: "local" or "nextcloud" 
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.data_source = data_source
        
        # Load or create configuration
        if isinstance(config_path, str) and config_path.endswith('.json'):
            self.config = Config(config_path)
            logger.info(f"Loaded config from: {config_path}")
        else:
            # Create minimal config for testing
            self.config = Config({
                "input/n_channels": 100,
                "input/preprocessing": "L1",
                "input/annotation_name": "polygon#annotator1",
                "label_mapping": None,  # Will be set based on data
            })
            logger.info("Created default config")
        
        self.label_mapping = LabelMapping.from_config(self.config)
        self.results = {}
        
    # ============================================================================
    # WORKPACKAGE 1.1: Load Datasets
    # ============================================================================
    
    def load_datasets_from_source(
        self, 
        source_path: str,
        pattern: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[DataPath]:
        """
        Load datasets from local or NextCloud source
        
        Args:
            source_path: Path to dataset directory (local or NextCloud mount)
            pattern: Optional image name pattern filter
            limit: Limit number of images loaded
            
        Returns:
            List of DataPath objects
        """
        logger.info(f"Loading datasets from {self.data_source} source: {source_path}")
        
        paths = []
        try:
            # Iterate through available data
            all_paths = list(DataPath.iterate(source_path))
            logger.info(f"Found {len(all_paths)} total images")
            
            # Apply filters
            if pattern:
                all_paths = [p for p in all_paths if pattern in p.image_name()]
                logger.info(f"Filtered to {len(all_paths)} images matching pattern: {pattern}")
            
            if limit and len(all_paths) > limit:
                all_paths = all_paths[:limit]
                logger.info(f"Limited to {limit} images")
            
            paths = all_paths
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
        
        logger.info(f"Successfully loaded {len(paths)} DataPath objects")
        return paths
    
    # ============================================================================
    # WORKPACKAGE 1.2: Define Training/Validation/Testing Sets
    # ============================================================================
    
    def define_train_val_test_split(
        self,
        paths: List[DataPath],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Dict[str, List[DataPath]]:
        """
        Split datasets into training, validation, and testing sets
        
        Args:
            paths: List of DataPath objects
            train_ratio: Fraction for training (default 0.6)
            val_ratio: Fraction for validation (default 0.2)
            test_ratio: Fraction for testing (default 0.2)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing DataPath lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(random_seed)
        n_paths = len(paths)
        indices = np.random.permutation(n_paths)
        
        n_train = int(n_paths * train_ratio)
        n_val = int(n_paths * val_ratio)
        
        split = {
            'train': [paths[i] for i in indices[:n_train]],
            'val': [paths[i] for i in indices[n_train:n_train + n_val]],
            'test': [paths[i] for i in indices[n_train + n_val:]],
        }
        
        logger.info(f"Split datasets:")
        logger.info(f"  Train: {len(split['train'])} images")
        logger.info(f"  Val:   {len(split['val'])} images")
        logger.info(f"  Test:  {len(split['test'])} images")
        
        return split
    
    # ============================================================================
    # WORKPACKAGE 1.3: Output Confusion Matrix
    # ============================================================================
    
    def compute_confusion_matrix(
        self,
        test_paths: List[DataPath],
        model_predictions: Optional[np.ndarray] = None,
        use_ground_truth: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrix from test data
        
        Args:
            test_paths: List of test DataPath objects
            model_predictions: Optional pre-computed predictions
            use_ground_truth: Use ground truth labels if True
            
        Returns:
            Dictionary containing confusion matrices and metrics
        """
        logger.info(f"Computing confusion matrix for {len(test_paths)} test images")
        
        # Create dataset
        dataset = DatasetImage(test_paths, train=False, config=self.config)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Accumulate confusion matrix components
        all_labels = []
        all_predictions = []
        
        for idx, sample in enumerate(dataset):
            if idx % max(1, len(dataset) // 10) == 0:
                logger.info(f"  Processing sample {idx}/{len(dataset)}")
            
            labels = sample['labels'].numpy()
            valid_pixels = sample['valid_pixels'].numpy()
            
            # Use valid pixels only
            valid_mask = valid_pixels > 0
            labels_valid = labels[valid_mask]
            all_labels.extend(labels_valid.flatten())
            
            # For demonstration: use labels as predictions
            # In practice, replace with actual model predictions
            if model_predictions is None:
                predictions = labels_valid
            else:
                predictions = model_predictions[valid_mask]
            
            all_predictions.extend(predictions.flatten())
        
        # Compute confusion matrix
        n_classes = len(self.label_mapping)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        
        for true_label, pred_label in zip(all_labels, all_predictions):
            if 0 <= true_label < n_classes and 0 <= pred_label < n_classes:
                cm[true_label, pred_label] += 1
        
        logger.info(f"Confusion matrix shape: {cm.shape}")
        
        # Normalize confusion matrix
        cm_rel, cm_rel_std = normalize_grouped_cm(np.stack([cm]))
        cm_rel = cm_rel * 100  # Convert to percentage
        cm_rel_std = cm_rel_std * 100
        
        results = {
            'cm_absolute': cm,
            'cm_relative': cm_rel,
            'cm_relative_std': cm_rel_std,
            'n_classes': n_classes,
        }
        
        return results
    
    def compute_metrics(
        self,
        confusion_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute classification metrics from confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            
        Returns:
            Dictionary of metrics (accuracy, precision, recall, f1)
        """
        logger.info("Computing classification metrics")
        
        metrics = {}
        
        # Overall accuracy
        correct = np.trace(confusion_matrix)
        total = np.sum(confusion_matrix)
        metrics['accuracy'] = correct / total if total > 0 else 0.0
        
        # Per-class metrics
        precision_list = []
        recall_list = []
        f1_list = []
        
        for i in range(confusion_matrix.shape[0]):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        
        metrics['macro_precision'] = np.mean(precision_list)
        metrics['macro_recall'] = np.mean(recall_list)
        metrics['macro_f1'] = np.mean(f1_list)
        
        logger.info(f"Metrics computed:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def visualize_confusion_matrix(
        self,
        cm_relative: np.ndarray,
        labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 14)
    ) -> None:
        """
        Visualize confusion matrix as heatmap
        
        Args:
            cm_relative: Relative (percentage) confusion matrix
            labels: Optional list of class labels
            output_path: Optional path to save figure
            figsize: Figure size
        """
        logger.info("Visualizing confusion matrix")
        
        if labels is None:
            labels = [self.label_mapping.index_to_name(i) for i in range(len(cm_relative))]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm_relative,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to: {output_path}")
        
        plt.show()
    
    # ============================================================================
    # WORKPACKAGE 1.4: Combine Datasets/Folders
    # ============================================================================
    
    def combine_datasets(
        self,
        dataset_dirs: List[str]
    ) -> List[DataPath]:
        """
        Combine multiple dataset directories
        
        Args:
            dataset_dirs: List of paths to dataset directories
            
        Returns:
            Combined list of DataPath objects
        """
        logger.info(f"Combining {len(dataset_dirs)} dataset directories")
        
        combined_paths = []
        for dataset_dir in dataset_dirs:
            logger.info(f"  Loading from: {dataset_dir}")
            try:
                paths = list(DataPath.iterate(dataset_dir))
                combined_paths.extend(paths)
                logger.info(f"    Added {len(paths)} images")
            except Exception as e:
                logger.warning(f"    Failed to load from {dataset_dir}: {e}")
        
        logger.info(f"Total combined: {len(combined_paths)} images")
        return combined_paths
    
    # ============================================================================
    # WORKPACKAGE 1.5-1.6: Vary Parameters and Permutate Combinations
    # ============================================================================
    
    def parameter_sweep(
        self,
        paths: List[DataPath],
        param_grid: Dict[str, List],
        test_paths: Optional[List[DataPath]] = None
    ) -> pd.DataFrame:
        """
        Perform parameter sweep and collect results
        
        Args:
            paths: List of DataPath objects
            param_grid: Dictionary of parameter names to list of values
            test_paths: Optional separate test set
            
        Returns:
            DataFrame with results for each parameter combination
        """
        logger.info(f"Starting parameter sweep over {len(param_grid)} parameters")
        
        results_list = []
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Total combinations: {len(combinations)}")
        
        for idx, combo in enumerate(combinations):
            logger.info(f"\nCombination {idx + 1}/{len(combinations)}: {dict(zip(param_names, combo))}")
            
            # Apply parameters
            params = dict(zip(param_names, combo))
            self.apply_parameters(params)
            
            # Update config
            if 'preprocessing' in params:
                self.config['input/preprocessing'] = params['preprocessing']
            
            # Split data
            split_ratios = params.get('train_ratio', 0.6), params.get('val_ratio', 0.2), params.get('test_ratio', 0.2)
            split = self.define_train_val_test_split(
                paths,
                train_ratio=split_ratios[0],
                val_ratio=split_ratios[1],
                test_ratio=split_ratios[2]
            )
            
            # Use provided test set or from split
            test_set = test_paths if test_paths else split['test']
            
            # Compute confusion matrix
            try:
                cm_results = self.compute_confusion_matrix(test_set)
                metrics = self.compute_metrics(cm_results['cm_absolute'])
                
                # Store results
                result_row = params.copy()
                result_row.update(metrics)
                result_row['n_test_samples'] = len(test_set)
                results_list.append(result_row)
                
            except Exception as e:
                logger.warning(f"Failed for combination {dict(zip(param_names, combo))}: {e}")
        
        results_df = pd.DataFrame(results_list)
        logger.info(f"\nParameter sweep complete. Results shape: {results_df.shape}")
        
        return results_df
    
    def apply_parameters(self, params: Dict) -> None:
        """Apply parameters to workflow"""
        logger.info(f"Applying parameters: {params}")
        # Implementation would depend on specific parameters
        pass
    
    # ============================================================================
    # WORKPACKAGE 1.7: Output Explainability
    # ============================================================================
    
    def compute_per_class_analysis(
        self,
        confusion_matrix: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute per-class analysis for explainability
        
        Args:
            confusion_matrix: Confusion matrix
            
        Returns:
            DataFrame with per-class metrics
        """
        logger.info("Computing per-class analysis for explainability")
        
        n_classes = confusion_matrix.shape[0]
        analysis_rows = []
        
        for i in range(n_classes):
            class_name = self.label_mapping.index_to_name(i)
            
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = np.sum(confusion_matrix) - tp - fp - fn
            
            # Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Identify most confused classes
            confusion_col = confusion_matrix[:, i].copy()
            confusion_col[i] = -1  # Exclude diagonal
            top_confused_idx = np.argmax(confusion_col)
            top_confused_name = self.label_mapping.index_to_name(top_confused_idx)
            top_confused_count = confusion_col[top_confused_idx]
            
            analysis_rows.append({
                'class_name': class_name,
                'class_index': i,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'most_confused_with': top_confused_name,
                'confusion_count': top_confused_count,
            })
        
        analysis_df = pd.DataFrame(analysis_rows)
        logger.info(f"Per-class analysis complete:\n{analysis_df}")
        
        return analysis_df
    
    def generate_explainability_report(
        self,
        confusion_matrix: np.ndarray,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive explainability report
        
        Args:
            confusion_matrix: Confusion matrix
            output_path: Optional path to save report
            
        Returns:
            Dictionary containing full explainability analysis
        """
        logger.info("Generating explainability report")
        
        report = {
            'per_class_analysis': self.compute_per_class_analysis(confusion_matrix).to_dict(),
            'confusion_matrix_stats': {
                'shape': confusion_matrix.shape,
                'total_predictions': int(np.sum(confusion_matrix)),
                'correct_predictions': int(np.trace(confusion_matrix)),
                'accuracy': float(np.trace(confusion_matrix) / np.sum(confusion_matrix)),
            },
            'error_analysis': self._compute_error_analysis(confusion_matrix),
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved report to: {output_path}")
        
        return report
    
    def _compute_error_analysis(self, cm: np.ndarray) -> Dict:
        """Compute error analysis from confusion matrix"""
        n_classes = cm.shape[0]
        errors = []
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    class_i = self.label_mapping.index_to_name(i)
                    class_j = self.label_mapping.index_to_name(j)
                    errors.append({
                        'true_class': class_i,
                        'predicted_class': class_j,
                        'count': int(cm[i, j]),
                        'error_type': 'false_positive' if i < j else 'false_negative'
                    })
        
        # Sort by count
        errors.sort(key=lambda x: x['count'], reverse=True)
        return {'top_errors': errors[:10]}  # Top 10 errors


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='HTC Framework ML Workflow Test')
    parser.add_argument('--data-dir', type=str, default='.', 
                       help='Path to dataset directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of images')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize confusion matrix')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize workflow
    logger.info("=" * 70)
    logger.info("HTC Framework ML Workflow - Confusion Matrix Generation")
    logger.info("=" * 70)
    
    workflow = HTCWorkflow(args.config or {}, verbose=True)
    
    try:
        # Workpackage 1.1: Load datasets
        logger.info("\n[WP 1.1] Loading Datasets...")
        paths = workflow.load_datasets_from_source(args.data_dir, limit=args.limit)
        
        # Workpackage 1.2: Define train/val/test split
        logger.info("\n[WP 1.2] Defining Train/Val/Test Split...")
        split = workflow.define_train_val_test_split(paths)
        
        # Workpackage 1.3: Output confusion matrix
        logger.info("\n[WP 1.3] Computing Confusion Matrix...")
        cm_results = workflow.compute_confusion_matrix(split['test'])
        
        # Compute metrics
        metrics = workflow.compute_metrics(cm_results['cm_absolute'])
        logger.info(f"\nMetrics: {metrics}")
        
        # Save confusion matrix
        cm_path = Path(args.output_dir) / 'confusion_matrix.npy'
        np.save(cm_path, cm_results['cm_absolute'])
        logger.info(f"Saved confusion matrix to: {cm_path}")
        
        # Visualize if requested
        if args.visualize:
            viz_path = Path(args.output_dir) / 'confusion_matrix.png'
            workflow.visualize_confusion_matrix(cm_results['cm_relative'], output_path=str(viz_path))
        
        # Workpackage 1.7: Explainability
        logger.info("\n[WP 1.7] Generating Explainability Report...")
        report = workflow.generate_explainability_report(
            cm_results['cm_absolute'],
            output_path=Path(args.output_dir) / 'explainability_report.json'
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("Workflow Complete!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
