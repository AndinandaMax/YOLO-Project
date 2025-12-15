# Urban Issues Detection with YOLOv12

A comprehensive object detection system for identifying and classifying urban infrastructure problems using YOLOv12 deep learning model.

## 📋 Overview

This project implements an automated urban issues detection system capable of identifying 10 different types of urban problems including potholes, damaged infrastructure, illegal parking, vandalism, and other municipal issues. The system uses state-of-the-art YOLOv12 architecture for real-time object detection.

## 🎯 Detected Urban Issues

The model can detect the following classes:

1. **Damaged Road** - Road surface deterioration
2. **Pothole** - Holes in road surfaces
3. **Illegal Parking** - Vehicles parked in prohibited areas
4. **Broken Road Sign** - Damaged or vandalized traffic signs
5. **Fallen Trees** - Trees blocking roads or pathways
6. **Littering/Garbage** - Waste in public places
7. **Vandalism** - Graffiti and property damage
8. **Dead Animal** - Animal carcasses causing pollution
9. **Damaged Concrete** - Deteriorated concrete structures
10. **Damaged Electric** - Damaged electrical poles and wires

## 🏗️ Project Structure

```
YOLO-Project/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading and analysis
│   ├── train.py                 # Model training module
│   └── eval.py                  # Model evaluation module
├── config/                       # Configuration files
│   └── data.yaml                # YOLO dataset configuration
├── data/                         # Dataset directory
│   ├── res/                     # Raw dataset
│   └── processed/               # Processed dataset
│       ├── train/               # Training data
│       ├── valid/               # Validation data
│       └── test/                # Test data
├── outputs/                      # Training outputs
│   ├── models/                  # Saved models
│   ├── runs/                    # Training runs
│   └── plots/                   # Visualizations
├── notebooks/                    # Jupyter notebooks
│   └── urbanissues.ipynb        # Main analysis notebook
├── requirements.txt              # Python dependencies
├── app.py                       # Streamlit web application
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd YOLO-Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Using the Data Loader Module

```python
from src.data_loader import YOLODatasetAnalyzer, reorganize_dataset, create_data_yaml
from pathlib import Path

# Define class mapping
class_mapping = {
    'Potholes and RoadCracks': 0,
    'IllegalParking': 2,
    'DamagedRoadSigns': 3,
    'FallenTrees': 4,
    'Garbage': 5,
    'Graffitti': 6,
    'DeadAnimalsPollution': 7,
    'Damaged concrete structures': 8,
    'DamagedElectricalPoles': 9,
}

# Reorganize dataset
stats = reorganize_dataset(
    source_path='data/res',
    dest_path='data/processed',
    class_mapping=class_mapping
)

# Create YAML configuration
yaml_path = create_data_yaml(
    organized_path='data/processed',
    output_path='config',
    class_names={...}
)
```

## 🎓 Training the Model

### Command Line Training

```bash
python src/train.py \
    --data config/data.yaml \
    --model yolo12n.pt \
    --epochs 50 \
    --batch 64 \
    --imgsz 640 \
    --device cuda
```

### Python API Training

```python
from src.train import train_urban_detector

model = train_urban_detector(
    data_yaml_path='config/data.yaml',
    model_name='yolo12n.pt',
    epochs=50,
    batch=64,
    imgsz=640,
    output_dir='outputs'
)
```

### Resume Training from Checkpoint

```bash
python src/train.py \
    --data config/data.yaml \
    --resume outputs/models/last.pt \
    --epochs 20
```

## 📈 Model Evaluation

### Command Line Evaluation

```bash
python src/eval.py \
    --model outputs/models/best.pt \
    --data config/data.yaml \
    --split val \
    --output outputs
```

### Python API Evaluation

```python
from src.eval import evaluate_model

evaluator = evaluate_model(
    model_path='outputs/models/best.pt',
    data_yaml_path='config/data.yaml',
    split='val',
    visualize=True
)

# Get metrics
metrics = evaluator.get_metrics_summary()
print(metrics)
```

### Compare Multiple Models

```python
from src.eval import compare_models

comparison = compare_models(
    model_paths=['model1.pt', 'model2.pt', 'model3.pt'],
    data_yaml_path='config/data.yaml',
    model_names=['YOLOv12n', 'YOLOv12s', 'YOLOv12m']
)
```

## 🔍 Inference

### Single Image Prediction

```python
from src.eval import YOLOEvaluator

evaluator = YOLOEvaluator('outputs/models/best.pt')

# Predict and visualize
evaluator.predict_and_visualize(
    image_path='test_image.jpg',
    conf=0.25,
    save_path='outputs/predictions/result.jpg'
)
```

### Batch Prediction

```python
# Predict on directory of images
results = evaluator.predict(
    source='data/test/images',
    conf=0.25,
    save=True
)
```

## 🌐 Web Application

Run the Streamlit web application for interactive detection:

```bash
streamlit run app.py
```

The web app allows you to:
- Upload images for detection
- Adjust confidence threshold
- View detection results in real-time
- Download annotated images

## 📊 Performance Metrics

The trained model achieves the following performance metrics:

| Metric      | Value  |
|-------------|--------|
| mAP@50      | 0.637  |
| mAP@50-95   | 0.425  |
| Precision   | 0.651  |
| Recall      | 0.599  |

*Note: Metrics may vary based on training configuration and dataset*

## 🛠️ Model Architecture

- **Base Model**: YOLOv12n (Nano version)
- **Input Size**: 640x640 pixels
- **Classes**: 10 urban issue categories
- **Parameters**: ~2.5M parameters
- **GFLOPs**: 6.3

## 📁 Dataset Statistics

- **Total Images**: 47,140
- **Training Set**: 38,086 images
- **Validation Set**: 4,894 images
- **Test Set**: 4,160 images
- **Total Annotations**: ~110,698 objects

## 🔧 Advanced Features

### Export Model to Different Formats

```python
from src.train import YOLOTrainer

trainer = YOLOTrainer('outputs/models/best.pt')

# Export to ONNX
trainer.export_model(format='onnx', save_path='model.onnx')

# Export to TensorRT
trainer.export_model(format='engine', save_path='model.engine')
```

### Dataset Analysis and Visualization

```python
from src.data_loader import YOLODatasetAnalyzer

analyzer = YOLODatasetAnalyzer('data/processed', class_names)

# Analyze dataset
class_counts, bbox_counts, bbox_sizes = analyzer.analyze_dataset()

# Create visualizations
analyzer.visualize_analysis('outputs')
analyzer.visualize_samples(num_samples=6, output_path='outputs')
```

## 📝 Configuration

Edit `config/data.yaml` to customize dataset paths and classes:

```yaml
path: data/processed
train: train/images
val: valid/images
test: test/images
nc: 10
names:
  0: Damaged Road issues
  1: Pothole Issues
  2: Illegal Parking Issues
  # ... more classes
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv12 by Ultralytics
- Urban Issues Dataset contributors
- Open-source community

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🔗 References

- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/)
- [YOLO Object Detection](https://github.com/ultralytics/ultralytics)
- [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes)

---

