"""
MAL (Matching-Aware Loss) implementation for DEIM
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .box_ops import box_cxcywh_to_xyxy, box_iou
from .improved_matcher import ImprovedHungarianMatcher


class MALoss(nn.Module):
    """
    Matching-Aware Loss implementation
    """
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 1.5,
                 num_classes: int = 80,
                 use_improved_iou: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.use_improved_iou = use_improved_iou
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: List[Dict[str, torch.Tensor]], 
                indices: List[Tuple[torch.Tensor, torch.Tensor]],
                num_boxes: int) -> Dict[str, torch.Tensor]:
        """
        Compute MAL loss
        
        Args:
            outputs: Model outputs containing 'pred_logits' and 'pred_boxes'
            targets: List of target dictionaries containing 'labels' and 'boxes'
            indices: List of (pred_idx, target_idx) tuples for each batch
            num_boxes: Total number of boxes for normalization
            
        Returns:
            Dictionary containing 'loss_mal' tensor
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Get source and target boxes
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Compute IoU
        if self.use_improved_iou:
            # Use improved IoU calculation if available
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            # Standard IoU calculation
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        
        # Get source logits and target classes
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Create target tensor
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        
        # Create target scores based on IoU
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        
        # Compute MAL loss
        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        
        # Apply MAL weighting
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target
        
        # Compute binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction='none'
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        
        return {'loss_mal': loss}
    
    def _get_src_permutation_idx(self, indices):
        """Get source permutation indices from matching results"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class MALMatcher(ImprovedHungarianMatcher):
    """
    MAL-aware Hungarian Matcher
    Inherits from ImprovedHungarianMatcher with MAL-specific configurations
    """
    def __init__(self, 
                 weight_dict: Dict[str, float],
                 iou_alpha: float = 0.7,
                 nwd_alpha: float = 0.3,
                 use_inner_iou: bool = True,
                 use_improved_matching: bool = True):
        super().__init__(
            weight_dict=weight_dict,
            iou_alpha=iou_alpha,
            nwd_alpha=nwd_alpha,
            use_inner_iou=use_inner_iou,
            use_improved_matching=use_improved_matching
        )
        
    def forward(self, outputs, targets):
        """
        Perform MAL-aware matching
        """
        return super().forward(outputs, targets)
