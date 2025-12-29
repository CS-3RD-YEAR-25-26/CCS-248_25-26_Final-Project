"""
Cattle Body Parts Detection Utilities
Handles dataset loading, preprocessing, and evaluation metrics
"""

import tensorflow as tf
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from PIL import Image

# Class definitions based on existing annotations
CLASSES = {
    0: "Head",
    1: "Leg", 
    2: "Back"
}

NUM_CLASSES = len(CLASSES)

# Detection slots per class (based on typical annotation patterns)
MAX_DETECTIONS = {
    "Head": 1,
    "Leg": 4,
    "Back": 1
}

TOTAL_DETECTIONS = sum(MAX_DETECTIONS.values())  # 6 total detection heads


class CattleDatasetLoader:
    """Loads and preprocesses cattle body parts dataset from LabelMe annotations"""
    
    def __init__(self, images_dir: str, annotations_dir: str, 
                 image_size: int = 256, batch_size: int = 8, seed: int = 42):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.img_size = (image_size, image_size)
        self.batch_size = batch_size
        self.seed = seed
        
        # Map class names to IDs
        self.class_to_id = {name: idx for idx, name in CLASSES.items()}
        
        # Add compute_iou and compute_map as instance methods for compatibility
        self.compute_iou = compute_iou
        self.compute_map = compute_map
        
    def parse_labelme_annotation(self, ann_path: str) -> Dict:
        """Parse LabelMe JSON annotation file"""
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        # Extract image info
        if 'imagePath' in data:
            img_filename = data['imagePath']
        else:
            # Fallback: derive from annotation filename
            img_filename = Path(ann_path).stem + '.jpg'
        
        img_width = data.get('imageWidth', 1280)
        img_height = data.get('imageHeight', 720)
        
        # Parse objects/shapes
        objects = []
        for shape in data.get('shapes', []):
            label = shape['label']
            points = shape['points']
            
            # Convert rectangle points to bbox [x_min, y_min, x_max, y_max]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            bbox = [
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords)
            ]
            
            # Get class ID
            class_id = self.class_to_id.get(label, -1)
            if class_id == -1:
                continue  # Skip unknown classes
            
            objects.append({
                'class_id': class_id,
                'class_name': label,
                'bbox': bbox
            })
        
        return {
            'filename': img_filename,
            'width': img_width,
            'height': img_height,
            'objects': objects
        }
    
    def load_and_preprocess_image(self, img_path: str) -> tf.Tensor:
        """Load and preprocess image"""
        # Read image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize to target size
        img = tf.image.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        
        return img
    
    def normalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Normalize bounding box coordinates to [0, 1]"""
        x_min, y_min, x_max, y_max = bbox
        
        return [
            x_min / img_width,
            y_min / img_height,
            x_max / img_width,
            y_max / img_height
        ]
    
    def create_fixed_output_target(self, objects: List[Dict], img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create fixed-size output targets for multi-object detection
        Returns:
            bboxes: (6, 4) - 6 detection slots, 4 bbox coords each
            classes: (6, NUM_CLASSES) - one-hot encoded class labels
        """
        # Initialize outputs
        bboxes = np.zeros((TOTAL_DETECTIONS, 4), dtype=np.float32)
        classes = np.zeros((TOTAL_DETECTIONS, NUM_CLASSES), dtype=np.float32)
        
        # Group objects by class
        objects_by_class = {
            "Head": [],
            "Leg": [],
            "Back": []
        }
        
        for obj in objects:
            class_name = CLASSES[obj['class_id']]
            objects_by_class[class_name].append(obj)
        
        # Assign objects to detection slots
        slot_idx = 0
        
        for class_name in ["Head", "Leg", "Back"]:
            max_count = MAX_DETECTIONS[class_name]
            objs = objects_by_class[class_name][:max_count]  # Take up to max_count
            
            for obj in objs:
                # Normalize bbox
                norm_bbox = self.normalize_bbox(obj['bbox'], img_width, img_height)
                bboxes[slot_idx] = norm_bbox
                
                # One-hot encode class
                classes[slot_idx, obj['class_id']] = 1.0
                
                slot_idx += 1
            
            # Move to next class slots (even if current class has fewer objects)
            slot_idx += (max_count - len(objs))
        
        return bboxes, classes
    
    def get_all_annotations(self) -> List[str]:
        """Get all annotation file paths"""
        return sorted(list(self.annotations_dir.glob('*.json')))
    
    def create_dataset_split(self, train_ratio: float = 0.8, random_seed: int = 42):
        """Split dataset into train and validation sets"""
        all_annotations = self.get_all_annotations()
        
        np.random.seed(random_seed)
        np.random.shuffle(all_annotations)
        
        split_idx = int(len(all_annotations) * train_ratio)
        
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
        
        return train_annotations, val_annotations
    
    def create_dataset_triple_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                                   test_ratio: float = 0.1, random_seed: int = 42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            train_ratio: Proportion for training (default 0.8)
            val_ratio: Proportion for validation (default 0.1)
            test_ratio: Proportion for testing (default 0.1)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        all_annotations = self.get_all_annotations()
        
        # Shuffle annotations
        np.random.seed(random_seed)
        np.random.shuffle(all_annotations)
        
        # Calculate split indices
        total = len(all_annotations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split annotations
        train_annotations = all_annotations[:train_end]
        val_annotations = all_annotations[train_end:val_end]
        test_annotations = all_annotations[val_end:]
        
        print(f"Dataset split complete:")
        print(f"  Train: {len(train_annotations)} images ({len(train_annotations)/total*100:.1f}%)")
        print(f"  Val: {len(val_annotations)} images ({len(val_annotations)/total*100:.1f}%)")
        print(f"  Test: {len(test_annotations)} images ({len(test_annotations)/total*100:.1f}%)")
        
        # Create TensorFlow datasets with instance batch_size
        train_ds = self.create_tf_dataset(train_annotations, batch_size=self.batch_size, shuffle=True, augment=True)
        val_ds = self.create_tf_dataset(val_annotations, batch_size=self.batch_size, shuffle=False, augment=False)
        test_ds = self.create_tf_dataset(test_annotations, batch_size=self.batch_size, shuffle=False, augment=False)
        
        return train_ds, val_ds, test_ds
    
    def generator(self, annotation_paths: List[str]):
        """Generator function for tf.data.Dataset"""
        for ann_path in annotation_paths:
            try:
                # Parse annotation
                ann_data = self.parse_labelme_annotation(str(ann_path))
                
                # Get image path
                img_path = self.images_dir / ann_data['filename']
                
                if not img_path.exists():
                    continue
                
                # Load and preprocess image
                img = self.load_and_preprocess_image(str(img_path))
                
                # Create target outputs
                bboxes, classes = self.create_fixed_output_target(
                    ann_data['objects'],
                    ann_data['width'],
                    ann_data['height']
                )
                
                yield img.numpy(), bboxes, classes
                
            except Exception as e:
                print(f"Error processing {ann_path}: {e}")
                continue
    
    def create_tf_dataset(self, annotation_paths: List[str], batch_size: int = 8, 
                         shuffle: bool = True, augment: bool = False):
        """Create TensorFlow dataset"""
        
        def gen():
            return self.generator(annotation_paths)
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(TOTAL_DETECTIONS, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(TOTAL_DETECTIONS, NUM_CLASSES), dtype=tf.float32)
            )
        )
        
        if augment:
            dataset = dataset.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Restructure to match model's dual output format
        dataset = dataset.map(
            lambda img, bbox, cls: (img, {"bbox_output": bbox, "cls_output": cls}),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def augment_data(self, img, bboxes, classes):
        """Apply data augmentation"""
        # Random brightness
        img = tf.image.random_brightness(img, max_delta=0.2)
        
        # Random contrast
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        
        # Random saturation
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
        # Clip values to [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img, bboxes, classes


# Evaluation Metrics
def compute_iou(pred_bbox: np.ndarray, true_bbox: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes
    Args:
        pred_bbox: [x_min, y_min, x_max, y_max]
        true_bbox: [x_min, y_min, x_max, y_max]
    """
    # Get intersection coordinates
    x_min_inter = max(pred_bbox[0], true_bbox[0])
    y_min_inter = max(pred_bbox[1], true_bbox[1])
    x_max_inter = min(pred_bbox[2], true_bbox[2])
    y_max_inter = min(pred_bbox[3], true_bbox[3])
    
    # Compute intersection area
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        intersection = 0.0
    else:
        intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    # Compute union area
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    true_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
    union = pred_area + true_area - intersection
    
    if union <= 0:
        return 0.0
    
    iou = intersection / union
    return iou


def compute_average_precision(pred_bboxes: List[np.ndarray], 
                              true_bboxes: List[np.ndarray],
                              iou_threshold: float = 0.5) -> float:
    """
    Compute Average Precision (AP) for a single class
    """
    if len(true_bboxes) == 0:
        return 0.0
    
    if len(pred_bboxes) == 0:
        return 0.0
    
    # Compute IoU for all prediction-ground truth pairs
    ious = []
    for pred in pred_bboxes:
        max_iou = 0.0
        for true in true_bboxes:
            iou = compute_iou(pred, true)
            max_iou = max(max_iou, iou)
        ious.append(max_iou)
    
    # Count true positives
    tp = sum(1 for iou in ious if iou >= iou_threshold)
    
    # Compute precision and recall
    precision = tp / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    recall = tp / len(true_bboxes) if len(true_bboxes) > 0 else 0.0
    
    # Simplified AP calculation (single precision-recall point)
    ap = precision * recall
    
    return ap


def compute_map(pred_all: Dict[int, List[np.ndarray]], 
                true_all: Dict[int, List[np.ndarray]],
                iou_threshold: float = 0.5) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean Average Precision (mAP) across all classes
    Args:
        pred_all: Dictionary mapping class_id to list of predicted bboxes
        true_all: Dictionary mapping class_id to list of ground truth bboxes
    Returns:
        mAP value and per-class AP dictionary
    """
    aps = {}
    
    for class_id in range(NUM_CLASSES):
        pred_bboxes = pred_all.get(class_id, [])
        true_bboxes = true_all.get(class_id, [])
        
        ap = compute_average_precision(pred_bboxes, true_bboxes, iou_threshold)
        aps[class_id] = ap
    
    # Compute mean
    map_value = np.mean(list(aps.values())) if aps else 0.0
    
    return map_value, aps


# Visualization utilities
def draw_bounding_boxes(image: np.ndarray, 
                       bboxes: List[np.ndarray], 
                       classes: List[int],
                       scores: List[float] = None,
                       denormalize: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on image
    Args:
        image: numpy array (H, W, 3) in range [0, 1]
        bboxes: list of [x_min, y_min, x_max, y_max] (normalized if denormalize=True)
        classes: list of class IDs
        scores: optional confidence scores
        denormalize: whether to denormalize bbox coords
    """
    # Convert to uint8
    img_display = (image * 255).astype(np.uint8).copy()
    h, w = img_display.shape[:2]
    
    # Color map for classes
    colors = {
        0: (0, 255, 0),    # Head - Green
        1: (255, 0, 0),    # Leg - Blue
        2: (0, 165, 255)   # Back - Orange
    }
    
    for idx, (bbox, class_id) in enumerate(zip(bboxes, classes)):
        if denormalize:
            x_min = int(bbox[0] * w)
            y_min = int(bbox[1] * h)
            x_max = int(bbox[2] * w)
            y_max = int(bbox[3] * h)
        else:
            x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Skip invalid boxes
        if x_max <= x_min or y_max <= y_min:
            continue
        
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label
        label = CLASSES[class_id]
        if scores is not None and idx < len(scores):
            label = f"{label}: {scores[idx]:.2f}"
        
        # Draw background for text
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_display, (x_min, y_min - text_h - baseline - 5), 
                     (x_min + text_w, y_min), color, -1)
        
        # Draw text
        cv2.putText(img_display, label, (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_display
