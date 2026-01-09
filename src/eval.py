"""
Evaluation Module for Urban Issues Detection
Handles model evaluation, metrics calculation, and visualization
"""

import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


class YOLOEvaluator:
    """YOLO model evaluator for urban issues detection"""
    
    def __init__(self, model_path, data_yaml_path=None):
        """
        Initialize YOLO evaluator
        
        Args:
            model_path: Path to trained YOLO model
            data_yaml_path: Path to data configuration YAML
        """
        self.model_path = Path(model_path)
        self.data_yaml_path = data_yaml_path
        self.model = None
        self.metrics = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained YOLO model"""
        print(f"\n📊 Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully")
        return self.model
    
    def evaluate(self, data_yaml=None, split='val', **kwargs):
        """
        Evaluate model on validation/test set
        
        Args:
            data_yaml: Path to data configuration YAML
            split: Dataset split to evaluate on ('val' or 'test')
            **kwargs: Additional evaluation parameters
        
        Returns:
            Evaluation metrics
        """
        if data_yaml is None:
            data_yaml = self.data_yaml_path
        
        if data_yaml is None:
            raise ValueError("data_yaml path must be provided")
        
        print(f"\n📊 Evaluating model on {split} set...")
        
        # Run validation
        self.metrics = self.model.val(
            data=data_yaml,
            split=split,
            **kwargs
        )
        
        return self.metrics
    
    def get_metrics_summary(self):
        """
        Get summary of evaluation metrics
        
        Returns:
            Dictionary with key metrics
        """
        if self.metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        metrics_summary = {
            'mAP@50': self.metrics.box.map50,
            'mAP@50-95': self.metrics.box.map,
            'Precision': self.metrics.box.mp,
            'Recall': self.metrics.box.mr,
        }
        
        return metrics_summary
    
    def print_metrics(self):
        """Print evaluation metrics in formatted way"""
        if self.metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        summary = self.get_metrics_summary()
        
        print("\n" + "="*50)
        print("📈 MODEL PERFORMANCE METRICS")
        print("="*50)
        for metric, value in summary.items():
            print(f"{metric:15s}: {value:.4f}")
        print("="*50)
    
    def visualize_metrics(self, output_path=None):
        """
        Visualize evaluation metrics
        
        Args:
            output_path: Path to save visualization
        """
        if self.metrics is None:
            raise ValueError("No metrics available. Run evaluate() first.")
        
        summary = self.get_metrics_summary()
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': list(summary.keys()),
            'Value': list(summary.values())
        })
        
        print("\n📊 Metrics DataFrame:")
        print(metrics_df)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar plot
        axes[0].bar(metrics_df['Metric'], metrics_df['Value'], 
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(metrics_df['Value']):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Radar chart
        categories = metrics_df['Metric'].tolist()
        values = metrics_df['Value'].tolist()
        values += values[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(122, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Metrics visualization saved to {output_path}")
        
        plt.show()
        
        return metrics_df
    
    def predict(self, source, **kwargs):
        """
        Run inference on images/videos
        
        Args:
            source: Path to image, video, or directory
            **kwargs: Additional prediction parameters
        
        Returns:
            Prediction results
        """
        print(f"\n🔍 Running inference on {source}...")
        results = self.model.predict(source=source, **kwargs)
        print(f"✅ Inference completed on {len(results)} images")
        return results
    
    def predict_and_visualize(self, image_path, conf=0.25, save_path=None):
        """
        Predict and visualize results on a single image
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            save_path: Path to save annotated image
        
        Returns:
            Annotated image
        """
        # Run prediction
        results = self.model.predict(source=image_path, conf=conf, save=False)
        
        # Get the first result
        result = results[0]
        
        # Plot result
        annotated_img = result.plot()
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Detection Results (Confidence >= {conf})', fontsize=14, fontweight='bold')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated_img)
            print(f"\n✅ Annotated image saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return annotated_img


def evaluate_model(
    model_path,
    data_yaml_path,
    split='val',
    output_dir='../outputs',
    visualize=True
):
    """
    Convenience function to evaluate a trained model
    
    Args:
        model_path: Path to trained model
        data_yaml_path: Path to data configuration YAML
        split: Dataset split to evaluate ('val' or 'test')
        output_dir: Output directory for results
        visualize: Whether to create visualizations
    
    Returns:
        Evaluator object with metrics
    """
    # Initialize evaluator
    evaluator = YOLOEvaluator(model_path, data_yaml_path)
    
    # Run evaluation
    metrics = evaluator.evaluate(data_yaml=data_yaml_path, split=split)
    
    # Print metrics
    evaluator.print_metrics()
    
    # Visualize metrics
    if visualize:
        output_path = Path(output_dir) / 'plots' / 'model_metrics.png'
        evaluator.visualize_metrics(output_path=output_path)
    
    return evaluator


def compare_models(model_paths, data_yaml_path, model_names=None):
    """
    Compare multiple models
    
    Args:
        model_paths: List of model paths
        data_yaml_path: Path to data configuration YAML
        model_names: Optional list of model names for display
    
    Returns:
        DataFrame with comparison results
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results = []
    
    for model_path, name in zip(model_paths, model_names):
        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print(f"{'='*50}")
        
        evaluator = YOLOEvaluator(model_path, data_yaml_path)
        evaluator.evaluate()
        metrics = evaluator.get_metrics_summary()
        metrics['Model'] = name
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'mAP@50', 'mAP@50-95', 'Precision', 'Recall']]
    
    print("\n📊 Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    metrics_to_plot = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        values = comparison_df[metric].values
        ax.bar(x + i*width, values, width, label=metric, color=color)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for urban issues detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (val/test)')
    parser.add_argument('--output', type=str, default='../outputs', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluator = evaluate_model(
        model_path=args.model,
        data_yaml_path=args.data,
        split=args.split,
        output_dir=args.output,
        visualize=not args.no_viz
    )
    
    print("\n✅ Evaluation script completed!")
