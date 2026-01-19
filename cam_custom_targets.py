"""
Custom target classes for CAM methods to handle cat/dog group aggregation.
"""
import torch
from typing import List, Callable

class CatDogGroupTarget:
    """
    Target for cat/dog binary classification using ImageNet class aggregation.
    """
    def __init__(self, label: int, idx_cat: torch.Tensor, idx_dog: torch.Tensor):
        """
        Args:
            label: 0 for cat, 1 for dog
            idx_cat: Tensor of cat class indices in ImageNet
            idx_dog: Tensor of dog class indices in ImageNet
        """
        self.label = int(label)
        self.idx_cat = idx_cat
        self.idx_dog = idx_dog
    
    def __call__(self, model_output):
        """
        Compute aggregated score for cat or dog group.
        
        Args:
            model_output: Logits from model (B, C) or (C,)
        
        Returns:
            Aggregated score for the target species group
        """
        # Handle both batched and single output
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        
        # Softmax to get probabilities
        probs = torch.softmax(model_output, dim=1)
        
        # Sum probabilities for the target group
        if self.label == 0:  # Cat
            score = probs[:, self.idx_cat].sum(dim=1)
        else:  # Dog
            score = probs[:, self.idx_dog].sum(dim=1)
        
        return score


class SemanticTarget:
    """
    Generic semantic target that aggregates multiple class indices.
    """
    def __init__(self, class_indices: List[int], use_prob: bool = True):
        """
        Args:
            class_indices: List of class indices to aggregate
            use_prob: If True, use softmax probabilities; if False, use logits
        """
        self.class_indices = torch.tensor(class_indices, dtype=torch.long)
        self.use_prob = use_prob
    
    def __call__(self, model_output):
        """
        Compute aggregated score for semantic group.
        
        Args:
            model_output: Logits from model (B, C) or (C,)
        
        Returns:
            Aggregated score for the semantic group
        """
        # Handle both batched and single output
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        
        # Move indices to same device as output
        indices = self.class_indices.to(model_output.device)
        
        if self.use_prob:
            # Softmax to get probabilities
            probs = torch.softmax(model_output, dim=1)
            score = probs[:, indices].sum(dim=1)
        else:
            # Use raw logits
            score = model_output[:, indices].sum(dim=1)
        
        return score
