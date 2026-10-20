"""
Improved Hungarian Matcher with IoU_alpha*(1-NWD)^(1-alpha) matching strategy
Based on the suggestions from the supervisor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from ..core import register


def normalized_wasserstein_distance(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Normalized Wasserstein Distance (NWD) between two sets of boxes.
    
    Args:
        box1: [N, 4] tensor of boxes in cxcywh format
        box2: [M, 4] tensor of boxes in cxcywh format
        
    Returns:
        [N, M] tensor of NWD values
    """
    # Convert to xyxy format
    box1_xyxy = box_cxcywh_to_xyxy(box1)
    box2_xyxy = box_cxcywh_to_xyxy(box2)
    
    # Calculate center points and sizes
    c1 = box1_xyxy[:, :2] + (box1_xyxy[:, 2:] - box1_xyxy[:, :2]) / 2  # [N, 2]
    c2 = box2_xyxy[:, :2] + (box2_xyxy[:, 2:] - box2_xyxy[:, :2]) / 2  # [M, 2]
    
    s1 = box1_xyxy[:, 2:] - box1_xyxy[:, :2]  # [N, 2]
    s2 = box2_xyxy[:, 2:] - box2_xyxy[:, :2]  # [M, 2]
    
    # Calculate distance between centers
    center_dist = torch.cdist(c1, c2, p=2)  # [N, M]
    
    # Calculate size difference
    s1_expanded = s1.unsqueeze(1)  # [N, 1, 2]
    s2_expanded = s2.unsqueeze(0)  # [1, M, 2]
    size_diff = torch.norm(s1_expanded - s2_expanded, dim=2)  # [N, M]
    
    # Calculate NWD
    nwd = center_dist / (torch.norm(s1, dim=1, keepdim=True) + torch.norm(s2, dim=1, keepdim=True).T + 1e-8)
    nwd = nwd + size_diff / (torch.norm(s1, dim=1, keepdim=True) + torch.norm(s2, dim=1, keepdim=True).T + 1e-8)
    
    return nwd


def inner_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Inner IoU between two sets of boxes.
    
    Args:
        box1: [N, 4] tensor of boxes in cxcywh format
        box2: [M, 4] tensor of boxes in cxcywh format
        
    Returns:
        [N, M] tensor of Inner IoU values
    """
    # Convert to xyxy format
    box1_xyxy = box_cxcywh_to_xyxy(box1)
    box2_xyxy = box_cxcywh_to_xyxy(box2)
    
    # Calculate areas
    area1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])
    area2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])
    
    # Calculate intersection
    lt = torch.max(box1_xyxy[:, None, :2], box2_xyxy[:, :2])  # [N, M, 2]
    rb = torch.min(box1_xyxy[:, None, 2:], box2_xyxy[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate union
    union = area1[:, None] + area2 - inter  # [N, M]
    
    # Calculate Inner IoU (intersection over minimum area)
    min_area = torch.min(area1[:, None], area2)  # [N, M]
    inner_iou = inter / (min_area + 1e-8)
    
    return inner_iou


@register()
class ImprovedHungarianMatcher(nn.Module):
    """
    Improved Hungarian Matcher with IoU_alpha*(1-NWD)^(1-alpha) matching strategy
    """
    
    def __init__(self, 
                 weight_dict: Dict[str, float],
                 use_focal_loss: bool = False,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 iou_alpha: float = 0.7,
                 nwd_alpha: float = 0.3,
                 use_inner_iou: bool = True,
                 use_improved_matching: bool = True):
        """
        Args:
            weight_dict: Dictionary containing cost weights
            use_focal_loss: Whether to use focal loss for classification cost
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
            iou_alpha: Weight for IoU component in improved matching
            nwd_alpha: Weight for NWD component in improved matching
            use_inner_iou: Whether to use Inner IoU instead of standard IoU
            use_improved_matching: Whether to use improved matching strategy
        """
        super().__init__()
        self.cost_class = weight_dict.get('cost_class', 2.0)
        self.cost_bbox = weight_dict.get('cost_bbox', 5.0)
        self.cost_giou = weight_dict.get('cost_giou', 2.0)
        self.cost_nwd = weight_dict.get('cost_nwd', 1.0)
        
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        
        self.iou_alpha = iou_alpha
        self.nwd_alpha = nwd_alpha
        self.use_inner_iou = use_inner_iou
        self.use_improved_matching = use_improved_matching
        
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict], return_topk: bool = False):
        """
        Performs the matching with improved strategy
        
        Args:
            outputs: Model outputs containing pred_logits and pred_boxes
            targets: List of target dictionaries
            return_topk: Whether to return top-k matches
            
        Returns:
            Dictionary containing matching indices
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten predictions
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        # Concatenate targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute classification cost
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]
        
        # Compute bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute IoU cost
        if self.use_inner_iou:
            iou = inner_iou(out_bbox, tgt_bbox)
        else:
            iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Compute improved matching cost if enabled
        if self.use_improved_matching:
            # Calculate NWD
            nwd = normalized_wasserstein_distance(out_bbox, tgt_bbox)
            
            # Calculate improved IoU: IoU_alpha * (1 - NWD)^(1-alpha)
            improved_iou = self.iou_alpha * iou + (1 - self.iou_alpha) * (1 - nwd) ** (1 - self.nwd_alpha)
            cost_improved_iou = -improved_iou
            
            # Use improved IoU cost
            cost_iou = cost_improved_iou
        else:
            cost_iou = cost_giou
        
        # Final cost matrix
        C = (self.cost_bbox * cost_bbox + 
             self.cost_class * cost_class + 
             self.cost_giou * cost_iou)
        
        C = C.view(bs, num_queries, -1).cpu()
        
        # Solve assignment problem
        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                  for i, j in indices_pre]
        
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}
        
        return {'indices': indices}
    
    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        """Get top-k matches"""
        indices_list = []
        
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) 
                       for j in range(len(sizes))]
        
        return indices_list
