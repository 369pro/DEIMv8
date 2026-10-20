"""
Copy-Paste augmentation implementation for DEIM
Based on the paper "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Optional, Tuple

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class CopyPaste(T.Transform):
    """
    Copy-Paste augmentation that copies objects from one image and pastes them to another.
    This is particularly effective for object detection tasks.
    """
    
    def __init__(self, 
                 p: float = 0.5,
                 max_paste_objects: int = 3,
                 min_paste_objects: int = 1,
                 scale_range: Tuple[float, float] = (0.5, 1.5),
                 rotation_range: Tuple[float, float] = (-15, 15),
                 use_cache: bool = True,
                 max_cached_samples: int = 100):
        """
        Args:
            p (float): Probability of applying Copy-Paste augmentation
            max_paste_objects (int): Maximum number of objects to paste
            min_paste_objects (int): Minimum number of objects to paste
            scale_range (Tuple[float, float]): Range for scaling pasted objects
            rotation_range (Tuple[float, float]): Range for rotating pasted objects (degrees)
            use_cache (bool): Whether to use cached samples for source objects
            max_cached_samples (int): Maximum number of cached samples
        """
        super().__init__()
        self.p = p
        self.max_paste_objects = max_paste_objects
        self.min_paste_objects = min_paste_objects
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.use_cache = use_cache
        self.max_cached_samples = max_cached_samples

        # Cache for storing source samples
        self.cache = []

    def _convert_boxes_to_list(self, boxes):
        """Convert BoundingBoxes object to list of tensors"""
        if hasattr(boxes, 'data'):
            # BoundingBoxes object
            return boxes.data.tolist()
        elif isinstance(boxes, list):
            # Already a list
            return boxes
        
    def forward(self, *inputs: Any) -> Any:
        """
        Apply Copy-Paste augmentation
        
        Args:
            inputs: (image, target, dataset) tuple
            
        Returns:
            Augmented (image, target, dataset) tuple
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        
        image, target, dataset = inputs
        
        # Skip augmentation with probability 1 - p
        if random.random() >= self.p:
            return image, target, dataset
            
        # Get source samples for copying
        if self.use_cache and len(self.cache) > 0:
            source_samples = self._get_samples_from_cache()
        else:
            source_samples = self._get_samples_from_dataset(image, target, dataset)
            
        if not source_samples:
            return image, target, dataset
            
        # Apply Copy-Paste augmentation
        augmented_image, augmented_target = self._apply_copy_paste(
            image, target, source_samples
        )
        
        # Update cache
        if self.use_cache:
            self._update_cache(image, target)
            
        return augmented_image, augmented_target, dataset
    
    def _get_samples_from_cache(self) -> List[Tuple[Image.Image, Dict]]:
        """Get random samples from cache"""
        num_samples = random.randint(self.min_paste_objects, self.max_paste_objects)
        return random.sample(self.cache, min(num_samples, len(self.cache)))
    
    def _get_samples_from_dataset(self, image, target, dataset) -> List[Tuple[Image.Image, Dict]]:
        """Get random samples from dataset"""
        if not hasattr(dataset, 'load_item'):
            return []
            
        num_samples = random.randint(self.min_paste_objects, self.max_paste_objects)
        samples = []
        
        for _ in range(num_samples):
            try:
                idx = random.randint(0, len(dataset) - 1)
                sample_image, sample_target = dataset.load_item(idx)
                if 'boxes' in sample_target and len(sample_target['boxes']) > 0:
                    samples.append((sample_image, sample_target))
            except (IndexError, KeyError, AttributeError):
                continue
                
        return samples
    
    def _apply_copy_paste(self, 
                         image: Image.Image, 
                         target: Dict, 
                         source_samples: List[Tuple[Image.Image, Dict]]) -> Tuple[Image.Image, Dict]:
        """Apply Copy-Paste augmentation"""
        # Convert to PIL if needed
        if hasattr(image, 'convert'):
            pil_image = image.convert('RGB')
        else:
            pil_image = image
            
        # Create a copy for augmentation
        augmented_image = pil_image.copy()
        augmented_target = target.copy()
        
        # Get existing boxes and labels
        existing_boxes = target.get('boxes', [])
        existing_labels = target.get('labels', [])
        
        new_boxes = []
        new_labels = []
        
        for source_image, source_target in source_samples:
            if 'boxes' not in source_target or len(source_target['boxes']) == 0:
                continue
                
            # Select a random object from source
            num_objects = len(source_target['boxes'])
            obj_idx = random.randint(0, num_objects - 1)
            
            source_box = source_target['boxes'][obj_idx]
            source_label = source_target['labels'][obj_idx]
            
            # Convert source image to PIL if needed
            if hasattr(source_image, 'convert'):
                source_pil = source_image.convert('RGB')
            else:
                source_pil = source_image
                
            # Extract object from source image
            pasted_image, pasted_box, pasted_label = self._extract_and_paste_object(
                augmented_image, source_pil, source_box, source_label, 
                self._convert_boxes_to_list(existing_boxes) + new_boxes
            )
            
            if pasted_image is not None:
                augmented_image = pasted_image
                new_boxes.append(pasted_box)
                new_labels.append(pasted_label)
        
        # Update target with new boxes and labels
        if new_boxes:
            if isinstance(existing_boxes, list):
                augmented_target['boxes'] = self._convert_boxes_to_list(existing_boxes) + new_boxes
                augmented_target['labels'] = existing_labels + new_labels
            else:
                # Convert to tensor if needed
                all_boxes = torch.cat([existing_boxes, torch.stack(new_boxes)])
                all_labels = torch.cat([existing_labels, torch.stack(new_labels)])
                augmented_target['boxes'] = all_boxes
                augmented_target['labels'] = all_labels
        
        return augmented_image, augmented_target
    
    def _extract_and_paste_object(self, 
                                 target_image: Image.Image,
                                 source_image: Image.Image, 
                                 source_box: torch.Tensor,
                                 source_label: torch.Tensor,
                                 existing_boxes: List) -> Tuple[Optional[Image.Image], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract object from source and paste to target"""
        try:
            # Convert box coordinates to pixel coordinates
            if source_box.dim() == 1:
                source_box = source_box.unsqueeze(0)
                
            # Assuming box format is [x, y, w, h] normalized
            img_w, img_h = target_image.size
            x, y, w, h = source_box[0]
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            if x2 <= x1 or y2 <= y1:
                return None, None, None
                
            # Extract object from source image
            source_w, source_h = source_image.size
            src_x1 = int((x - w/2) * source_w)
            src_y1 = int((y - h/2) * source_h)
            src_x2 = int((x + w/2) * source_w)
            src_y2 = int((y + h/2) * source_h)
            
            src_x1 = max(0, min(src_x1, source_w))
            src_y1 = max(0, min(src_y1, source_h))
            src_x2 = max(0, min(src_x2, source_w))
            src_y2 = max(0, min(src_y2, source_h))
            
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                return None, None, None
                
            # Crop object from source
            obj_crop = source_image.crop((src_x1, src_y1, src_x2, src_y2))
            
            # Apply random transformations
            scale = random.uniform(*self.scale_range)
            rotation = random.uniform(*self.rotation_range)
            
            # Resize object
            new_w = int(obj_crop.width * scale)
            new_h = int(obj_crop.height * scale)
            obj_crop = obj_crop.resize((new_w, new_h), Image.LANCZOS)
            
            # Rotate object
            if abs(rotation) > 1e-6:
                obj_crop = obj_crop.rotate(rotation, expand=True)
            
            # Find a suitable position to paste
            paste_x, paste_y = self._find_paste_position(
                target_image, obj_crop, existing_boxes
            )
            
            if paste_x is None or paste_y is None:
                return None, None, None
                
            # Convert to RGBA for transparency support
            obj_crop_rgba = obj_crop.convert('RGBA')
            
            # Paste object to target image with transparency
            target_image.paste(obj_crop_rgba, (paste_x, paste_y), obj_crop_rgba)
            
            # Calculate new box coordinates
            new_w_norm = new_w / img_w
            new_h_norm = new_h / img_h
            new_x = (paste_x + new_w/2) / img_w
            new_y = (paste_y + new_h/2) / img_h
            
            new_box = torch.tensor([new_x, new_y, new_w_norm, new_h_norm])
            
            return target_image, new_box, source_label
            
        except Exception as e:
            print(f"Error in CopyPaste: {e}")
            return None, None, None
    
    def _find_paste_position(self, 
                           target_image: Image.Image, 
                           obj_crop: Image.Image,
                           existing_boxes: List) -> Tuple[Optional[int], Optional[int]]:
        """Find a suitable position to paste the object"""
        img_w, img_h = target_image.size
        obj_w, obj_h = obj_crop.size
        
        # Try multiple random positions
        for _ in range(10):
            x = random.randint(0, max(0, img_w - obj_w))
            y = random.randint(0, max(0, img_h - obj_h))
            
            # Check overlap with existing boxes
            new_box = torch.tensor([(x + obj_w/2)/img_w, (y + obj_h/2)/img_h, obj_w/img_w, obj_h/img_h])
            
            has_overlap = False
            for existing_box in existing_boxes:
                if self._calculate_iou(new_box, existing_box) > 0.3:  # Threshold for overlap
                    has_overlap = True
                    break
                    
            if not has_overlap:
                return x, y
                
        return None, None
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate IoU between two boxes"""
        try:
            # Convert to xyxy format
            x1_1, y1_1, w1, h1 = box1
            x1_2, y1_2, w2, h2 = box2
            
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
                
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            union = w1 * h1 + w2 * h2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except:
            return 0.0
    
    def _update_cache(self, image: Image.Image, target: Dict):
        """Update cache with new sample"""
        if len(self.cache) >= self.max_cached_samples:
            # Remove oldest sample
            self.cache.pop(0)
            
        self.cache.append((image, target))
