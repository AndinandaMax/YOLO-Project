"""
Data Loader Module for Urban Issues Detection
Handles dataset loading, organization, and preparation for YOLO training
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


class YOLODatasetAnalyzer:
    """Comprehensive YOLO dataset analyzer for EDA"""
    
    def __init__(self, dataset_path, class_names):
        """
        Initialize the dataset analyzer
        
        Args:
            dataset_path: Path to the dataset directory
            class_names: Dictionary mapping class IDs to class names
        """
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.stats = defaultdict(list)

    def find_images_and_labels(self):
        """Find all images and labels in dataset"""
        self.image_files = []
        self.label_files = []

        # Search for images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            self.image_files.extend(list(self.dataset_path.rglob(ext)))

        # Search for labels
        self.label_files = list(self.dataset_path.rglob('*.txt'))

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.label_files)} label files")

        return self.image_files, self.label_files

    def analyze_dataset(self):
        """Perform comprehensive EDA"""
        self.find_images_and_labels()

        class_counts = Counter()
        bbox_counts = []
        image_with_labels = 0
        bbox_sizes = []

        for label_file in tqdm(self.label_files, desc="Analyzing labels"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        image_with_labels += 1
                        bbox_counts.append(len(lines))

                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                                w, h = float(parts[3]), float(parts[4])
                                bbox_sizes.append((w, h))
            except Exception as e:
                print(f"Error reading {label_file}: {e}")

        self.class_counts = class_counts
        self.bbox_counts = bbox_counts
        self.bbox_sizes = bbox_sizes

        return class_counts, bbox_counts, bbox_sizes

    def visualize_analysis(self, output_path):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Class Distribution
        plt.subplot(2, 3, 1)
        classes = [self.class_names[i] for i in sorted(self.class_counts.keys())]
        counts = [self.class_counts[i] for i in sorted(self.class_counts.keys())]

        plt.barh(classes, counts, color='steelblue')
        plt.xlabel('Number of Instances', fontsize=12)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')

        for i, v in enumerate(counts):
            plt.text(v, i, f' {v}', va='center', fontsize=10)

        # 2. Class Distribution Pie Chart
        plt.subplot(2, 3, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')

        # 3. Objects per Image
        plt.subplot(2, 3, 3)
        plt.hist(self.bbox_counts, bins=30, color='coral', edgecolor='black')
        plt.xlabel('Number of Objects per Image', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Objects per Image\n(Avg: {np.mean(self.bbox_counts):.2f})',
                  fontsize=14, fontweight='bold')

        # 4. Bounding Box Width Distribution
        plt.subplot(2, 3, 4)
        widths = [w for w, h in self.bbox_sizes]
        plt.hist(widths, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Normalized Width', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Bounding Box Width Distribution', fontsize=14, fontweight='bold')

        # 5. Bounding Box Height Distribution
        plt.subplot(2, 3, 5)
        heights = [h for w, h in self.bbox_sizes]
        plt.hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.xlabel('Normalized Height', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Bounding Box Height Distribution', fontsize=14, fontweight='bold')

        # 6. Summary Statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')

        total_images = len(self.image_files)
        total_labels = len(self.label_files)
        total_objects = sum(counts)

        summary_text = f"""
        DATASET SUMMARY
        {'='*40}

        Total Images: {total_images}
        Total Label Files: {total_labels}
        Total Objects: {total_objects}

        Number of Classes: {len(self.class_counts)}

        Avg Objects/Image: {np.mean(self.bbox_counts):.2f}
        Max Objects/Image: {max(self.bbox_counts)}
        Min Objects/Image: {min(self.bbox_counts)}

        Most Common Class: 
        {self.class_names[max(self.class_counts, key=self.class_counts.get)]}
        ({max(counts)} instances)

        Least Common Class: 
        {self.class_names[min(self.class_counts, key=self.class_counts.get)]}
        ({min(counts)} instances)
        """

        plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig(output_path / 'plots' / 'eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'total_images': total_images,
            'total_objects': total_objects,
            'class_counts': self.class_counts,
        }

    def visualize_samples(self, num_samples=6, output_path=None):
        """Visualize random samples with annotations"""
        import random
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        samples = random.sample(self.image_files, min(num_samples, len(self.image_files)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))

        for idx, img_path in enumerate(samples):
            # Find corresponding label
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                label_path = Path(str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt'))

            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Draw annotations
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Convert to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)

                            # Draw bbox
                            color = (np.array(colors[class_id][:3]) * 255).astype(int)
                            cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 2)

                            # Add label
                            label = self.class_names[class_id]
                            cv2.putText(img, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path / 'plots' / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()


def reorganize_dataset(source_path, dest_path, class_mapping):
    """
    Reorganize dataset from class-separated folders to unified structure
    
    Args:
        source_path: Source dataset path
        dest_path: Destination path for reorganized dataset
        class_mapping: Dictionary mapping class names to class IDs
    
    Returns:
        Dictionary with statistics of reorganized dataset
    """
    stats = {
        'train': {'images': 0, 'labels': 0},
        'valid': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0}
    }

    source_path = Path(source_path)
    dest_path = Path(dest_path)

    for class_folder_name, class_id in class_mapping.items():
        print(f"\n📦 Processing: {class_folder_name} (Class ID: {class_id})")

        class_folders = list(source_path.rglob(class_folder_name))

        if not class_folders:
            print(f"  ⚠️ Warning: Folder '{class_folder_name}' not found")
            continue

        class_folder = max(class_folders, key=lambda p: len(p.parts))

        for split in ['train', 'valid', 'test']:
            split_path = class_folder / split

            if not split_path.exists():
                print(f"  ⚠️ {split} folder not found")
                continue

            images_path = split_path / 'images'
            labels_path = split_path / 'labels'

            if images_path.exists():
                image_files = list(images_path.glob('*.[jp][pn]g')) + \
                             list(images_path.glob('*.jpeg'))

                for img_file in image_files:
                    new_name = f"{class_folder_name.replace(' ', '_')}_{img_file.name}"
                    dest_img = dest_path / split / 'images' / new_name

                    shutil.copy2(img_file, dest_img)
                    stats[split]['images'] += 1

                    label_file = labels_path / (img_file.stem + '.txt')
                    if label_file.exists():
                        dest_label = dest_path / split / 'labels' / \
                                    (new_name.rsplit('.', 1)[0] + '.txt')

                        with open(label_file, 'r') as f:
                            lines = f.readlines()

                        updated_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = str(class_id)
                                updated_lines.append(' '.join(parts) + '\n')

                        with open(dest_label, 'w') as f:
                            f.writelines(updated_lines)

                        stats[split]['labels'] += 1

                print(f"  ✅ {split}: {len(image_files)} images")

    return stats


def create_data_yaml(organized_path, output_path, class_names):
    """
    Create data.yaml configuration file for YOLO training
    
    Args:
        organized_path: Path to organized dataset
        output_path: Output path for data.yaml
        class_names: Dictionary of class names
    
    Returns:
        Path to created data.yaml file
    """
    data_yaml_content = {
        'path': str(organized_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = Path(output_path) / 'data.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"\n✅ Created data.yaml at {yaml_path}")
    print("\n📄 Content:")
    with open(yaml_path, 'r') as f:
        print(f.read())

    return yaml_path


def verify_dataset_structure(organized_path):
    """
    Verify the structure of organized dataset
    
    Args:
        organized_path: Path to organized dataset
    """
    print("\n🔍 Verifying organized dataset structure:")
    for split in ['train', 'valid', 'test']:
        img_count = len(list((organized_path / split / 'images').glob('*')))
        lbl_count = len(list((organized_path / split / 'labels').glob('*.txt')))
        print(f"{split}: {img_count} images, {lbl_count} labels")


if __name__ == "__main__":
    # Example usage
    DATASET_INPUT = Path("../data/res")
    OUTPUT_PATH = Path("../outputs")
    
    class_names = {
        0: "Damaged Road",
        1: "Pothole",
        2: "Illegal Parking",
        3: "Broken Road Sign",
        4: "Fallen Trees",
        5: "Littering/Garbage",
        6: "Vandalism",
        7: "Dead Animal",
        8: "Damaged Concrete",
        9: "Damaged Electric"
    }
    
    # Initialize analyzer
    analyzer = YOLODatasetAnalyzer(DATASET_INPUT, class_names)
    
    # Run analysis
    print("\n Analyzing dataset...")
    class_counts, bbox_counts, bbox_sizes = analyzer.analyze_dataset()
    
    # Visualize
    print("\n Visualizing analysis...")
    stats = analyzer.visualize_analysis(OUTPUT_PATH)
    
    print("\n Visualizing samples...")
    analyzer.visualize_samples(num_samples=6, output_path=OUTPUT_PATH)
