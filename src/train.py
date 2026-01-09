"""
Training Module for Urban Issues Detection
Handles YOLO model training with configurable parameters
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


class YOLOTrainer:
    """YOLO model trainer for urban issues detection"""
    
    def __init__(self, model_name='yolo12n.pt', config_path=None):
        """
        Initialize YOLO trainer
        
        Args:
            model_name: Name of YOLO model to use (default: yolo12n.pt)
            config_path: Path to data configuration YAML file
        """
        self.model_name = model_name
        self.config_path = config_path
        self.model = None
        self.results = None
        
    def load_model(self):
        """Load YOLO model"""
        print(f"\n🚀 Loading {self.model_name}...")
        self.model = YOLO(self.model_name)
        print(f"✅ Model {self.model_name} loaded successfully")
        return self.model
    
    def train(self, 
              data=None,
              epochs=10,
              imgsz=640,
              batch=64,
              name='urban_issues_detector',
              patience=5,
              save=True,
              plots=True,
              val=True,
              project='../outputs/runs',
              device=None,
              **kwargs):
        """
        Train YOLO model
        
        Args:
            data: Path to data configuration YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            name: Name for training run
            patience: Early stopping patience
            save: Save model checkpoints
            plots: Generate training plots
            val: Run validation during training
            project: Project directory for saving results
            device: Device to use for training (cpu/cuda)
            **kwargs: Additional training parameters
        
        Returns:
            Training results object
        """
        if self.model is None:
            self.load_model()
        
        if data is None:
            data = self.config_path
        
        if data is None:
            raise ValueError("data configuration path must be provided")
        
        print(f"\n🚀 Starting YOLOv12 Training...")
        print(f"Configuration:")
        print(f"  - Data config: {data}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Image size: {imgsz}")
        print(f"  - Batch size: {batch}")
        print(f"  - Device: {device if device else 'auto'}")
        
        # Train the model
        self.results = self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=patience,
            save=save,
            plots=plots,
            val=val,
            project=project,
            device=device,
            **kwargs
        )
        
        print("\n✅ Training completed!")
        return self.results
    
    def save_model(self, save_path):
        """
        Save trained model
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path))
        print(f"\n✅ Model saved to {save_path}")
    
    def export_model(self, format='onnx', save_path=None):
        """
        Export model to different formats
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            save_path: Path to save exported model
        
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("No model to export. Train a model first.")
        
        print(f"\n📦 Exporting model to {format}...")
        export_path = self.model.export(format=format)
        
        if save_path:
            import shutil
            shutil.copy(export_path, save_path)
            print(f"✅ Model exported to {save_path}")
            return save_path
        
        print(f"✅ Model exported to {export_path}")
        return export_path


def train_urban_detector(
    data_yaml_path,
    model_name='yolo12n.pt',
    epochs=10,
    imgsz=640,
    batch=64,
    output_dir='../outputs',
    **kwargs
):
    """
    Convenience function to train urban issues detector
    
    Args:
        data_yaml_path: Path to data configuration YAML
        model_name: YOLO model to use
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        output_dir: Output directory for results
        **kwargs: Additional training parameters
    
    Returns:
        Trained YOLO model
    """
    # Initialize trainer
    trainer = YOLOTrainer(model_name=model_name, config_path=data_yaml_path)
    
    # Train model
    results = trainer.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=f'{output_dir}/runs',
        **kwargs
    )
    
    # Save model
    model_save_path = Path(output_dir) / 'models' / 'best.pt'
    trainer.save_model(model_save_path)
    
    return trainer.model


def resume_training(
    checkpoint_path,
    data_yaml_path,
    epochs=10,
    **kwargs
):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_yaml_path: Path to data configuration YAML
        epochs: Additional epochs to train
        **kwargs: Additional training parameters
    
    Returns:
        Trained YOLO model
    """
    print(f"\n🔄 Resuming training from {checkpoint_path}...")
    
    # Load model from checkpoint
    model = YOLO(checkpoint_path)
    
    # Resume training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        resume=True,
        **kwargs
    )
    
    print("\n✅ Training resumed and completed!")
    return model


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model for urban issues detection')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo12n.pt', help='YOLO model to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='../outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.resume:
        # Resume training
        model = resume_training(
            checkpoint_path=args.resume,
            data_yaml_path=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
    else:
        # Start new training
        model = train_urban_detector(
            data_yaml_path=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            output_dir=args.output,
            device=args.device
        )
    
    print("\n✅ Training script completed!")
