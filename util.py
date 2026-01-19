import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
from typing import Optional
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import sys
sys.path.append("pytorch_grad_cam")
from pytorch_grad_cam import ActivationsAndGradients

class Preprocessing_Layer(torch.nn.Module):
    def __init__(self, mean, std):
        super(Preprocessing_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def preprocess(self, img, mean, std):
        img = img.clone()
        img[:,0,:,:] = (img[:,0,:,:] - mean[0]) / std[0]
        img[:,1,:,:] = (img[:,1,:,:] - mean[1]) / std[1]
        img[:,2,:,:] = (img[:,2,:,:] - mean[2]) / std[2]
        return img

    def forward(self, x):
        return self.preprocess(x, self.mean, self.std)


def normlization_max_min(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    min_value = saliency_map.view(saliency_map.size(0),-1).min(dim=-1)[0]
    delta = torch.clamp(max_value - min_value, min=1e-8)
    min_value = min_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    delta = delta.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = (saliency_map - min_value) / delta
    return norm_saliency_map


def normlization_sigmoid(saliency_map):
    norm_saliency_map = 1/2*(nn.Tanh()(saliency_map/2)+1)
    return norm_saliency_map

def normlization_max(saliency_map):
    max_value = saliency_map.view(saliency_map.size(0),-1).max(dim=-1)[0]
    max_value = torch.clamp(max_value, min=1e-8)
    max_value = max_value.reshape((saliency_map.shape[0],1,1,1)).repeat((1,saliency_map.shape[1],saliency_map.shape[2],saliency_map.shape[3]))
    norm_saliency_map = saliency_map / max_value
    return norm_saliency_map


def _compute_canonical_loss(s_orig, s_masked, method='mse'):
    """
    Compute a canonical per-sample loss between original scores and masked image scores.
    
    Args:
        s_orig: Original image scores f(x) - shape (B,)
        s_masked: Masked image scores f(x⊙mask) - shape (B,)
        method: Loss type - 'abs' (L1) or 'mse' (squared L2, recommended)
    
    Returns:
        Per-sample loss values - shape (B,)
    """
    if method == 'abs':
        return torch.abs(s_orig - s_masked)
    if method == 'mse':
        return (s_orig - s_masked) ** 2
    raise Exception(f"Unknown canonical loss method: {method}")

class Basic_OptCAM:
    def __init__(self,
            model,
            device,
            max_iter=100,
            learning_rate=0.01,
            target_layer = '42',
            name_f = 'logit_predict',
            name_loss = 'mse',
            name_norm = 'max_min',
            canonical_loss = 'mse',  # MSE (squared L2) for smooth gradients and stable optimization
            # objective: 'mask' to maximize class score for masked image (matches Opti-CAM paper)
            # or 'diff' to minimize squared difference for smooth convergence (MSE recommended)
            objective: str = 'diff',
            use_prob: bool = False,
            idx_cat=None,
            idx_dog=None,
            name_mode = 'vgg',
            # Threshold for filtering low-confidence samples (used with only_correct)
            min_orig: float = 0.05,
            # Early stopping params
            # delta_change_threshold: stop when |loss_t - loss_{t-1}| < threshold (0 disables)
            delta_change_threshold: float = 0.0,
            save_best_weights: bool = False,
            save_best_weights_dir: Optional[str] = None,
            ):
        self.model = model.eval()
        self.device = device
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.fea_ext = ActivationsAndGradients(model, target_layer, None)
        self.name_f = name_f
        self.name_loss = name_loss
        self.name_norm = name_norm
        self.canonical_loss = canonical_loss
        self.objective = objective
        self.use_prob = bool(use_prob)
        if idx_cat is not None and idx_dog is not None:
            self.idx_cat = idx_cat
            self.idx_dog = idx_dog
        else:
            self.idx_cat = None
            self.idx_dog = None
        self.name_mode = name_mode
        self.last_feature = None
        self.last_internal_loss_history = None
        # Threshold for filtering low-confidence samples
        self.min_orig = float(min_orig)
        # Early stopping config (per-step delta on internal loss)
        self.delta_change_threshold = float(delta_change_threshold) if delta_change_threshold is not None else 0.0
        self.save_best_weights = bool(save_best_weights)
        self.save_best_weights_dir = save_best_weights_dir
        if self.save_best_weights and self.save_best_weights_dir is None:
            # If user requested saving but no directory provided, create a local fallback
            self.save_best_weights_dir = './best_weights'
        if self.save_best_weights:
            try:
                import os
                os.makedirs(self.save_best_weights_dir, exist_ok=True)
            except Exception:
                pass

    def get_f(self, x, y):
        outputs = self.model(x)
        if self.use_prob:
            probs = torch.softmax(outputs, dim=1)
            if self.idx_cat is not None and self.idx_dog is not None:
                p_cat = probs[:, self.idx_cat].sum(dim=1)
                p_dog = probs[:, self.idx_dog].sum(dim=1)
                return torch.where(y == 0, p_cat, p_dog)
            return probs[range(x.size(0)), y]
        else:
            predict_labels = y.to(outputs.device).long()
            j = outputs.gather(1, predict_labels.view(-1, 1)).squeeze(1)
            return j

    def get_loss(self, new_images, predict_labels, f_images):
        # Support canonical losses: 'abs' (or legacy 'norm'), 'mse'
        # Remove legacy 'plain' option. Use _compute_canonical_loss to
        # produce per-sample values then average.
        # If objective is 'mask', maximize the class score for the masked image
        # i.e. minimize the negative score. Use get_f which returns logits when
        # use_prob=False or probabilities when use_prob=True.
        if getattr(self, 'objective', 'mask') == 'mask':
            s_new = self.get_f(new_images, predict_labels)
            # minimize negative mean to maximize score
            loss = - s_new.mean()
            return loss

        # Otherwise fallback to legacy behavior: canonical difference loss
        s_new = self.get_f(new_images, predict_labels)
        # Prefer canonical_loss attribute (new unified API). Fall back to
        # legacy name_loss if canonical_loss isn't set.
        method = None
        if getattr(self, 'canonical_loss', None) is not None:
            method = self.canonical_loss
        else:
            if self.name_loss == 'abs':
                method = 'abs'
            elif self.name_loss == 'mse':
                method = 'mse'

        if method is not None:
            # f_images is expected to be the reference (original image scores)
            if f_images is None:
                raise ValueError("f_images (original image scores) is required to compute canonical loss")
            per_sample = _compute_canonical_loss(f_images.detach(), s_new, method=method)
            loss = per_sample.mean()
        else:
            raise Exception(f"Not Implemented loss type: {self.name_loss}")
        return loss

    def normalization(self, saliency_map):
        if self.name_norm =='max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        else:
            raise Exception("Not Implemented")

    def combine_activations(self, feature, w, images):
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        norm_saliency_map = self.normalization(saliency_map)
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        if self.name_mode == 'vgg' or self.name_mode == 'vgg_norm':
            feature = relu(self.fea_ext(images)[0])
        else:
            _ = self.fea_ext(images)
            feature = relu(self.fea_ext.activations[0]).to(self.device)
        self.last_feature = feature.detach()
        w = torch.full((feature.shape[0], feature.shape[1], 1, 1), 0.5, dtype=torch.float, device=self.device, requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        internal_history = []
        canonical_history = []
        predict_labels = labels
        # Precompute reference scores in the same representation used by get_f
        try:
            f_images = self.get_f(images, predict_labels).detach()
        except Exception:
            f_images = None

        # If we already have f_images (via get_f), use it as the canonical reference
        s_orig_ref = None
        if f_images is not None:
            s_orig_ref = f_images
        else:
            # Fall back to a single forward pass and pick logits or probs according to self.use_prob
            try:
                with torch.no_grad():
                    outputs = self.model(images)
                    if self.use_prob:
                        probs_orig = torch.softmax(outputs, dim=1)
                        if getattr(self, 'idx_cat', None) is not None and getattr(self, 'idx_dog', None) is not None:
                            p_cat = probs_orig[:, self.idx_cat].sum(dim=1)
                            p_dog = probs_orig[:, self.idx_dog].sum(dim=1)
                            s_orig_ref = torch.where(predict_labels == 0, p_cat, p_dog)
                        else:
                            s_orig_ref = probs_orig[range(images.size(0)), predict_labels]
                    else:
                        # use logits directly
                        if getattr(self, 'idx_cat', None) is not None and getattr(self, 'idx_dog', None) is not None:
                            p_cat = outputs[:, self.idx_cat].sum(dim=1)
                            p_dog = outputs[:, self.idx_dog].sum(dim=1)
                            s_orig_ref = torch.where(predict_labels == 0, p_cat, p_dog)
                        else:
                            s_orig_ref = outputs[range(images.size(0)), predict_labels]
            except Exception:
                s_orig_ref = None

        for step in range(self.max_iter):
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            try:
                internal_history.append(float(loss.detach().cpu().item()))
            except Exception:
                pass
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % max(1, (self.max_iter // 10)) == 0:
                print(f"[Basic_OptCAM] step={step:03d} loss={internal_history[-1]:.6f}")

            # --- Early stopping logic based on |Δloss| ---
            if self.delta_change_threshold > 0.0 and len(internal_history) >= 2:
                current_loss = internal_history[-1]
                prev_loss = internal_history[-2]
                delta = abs(current_loss - prev_loss)
                # Track best weights for potential saving
                if 'best_loss' not in locals() or current_loss < best_loss:
                    best_loss = current_loss
                    best_step = step
                    best_w = w.detach().clone()
                if delta < self.delta_change_threshold:
                    print(f"[Basic_OptCAM] Early stopping at step {step} (|Δloss|={delta:.3e} < {self.delta_change_threshold:.3e})")
                    break

        # Restore best weights if tracked
        if 'best_w' in locals():
            with torch.no_grad():
                w.data = best_w.data
        norm_saliency_map, new_images = self.combine_activations(feature, w, images)

        # Optionally save best weights snapshot (batch-level)
        if self.save_best_weights:
            try:
                import os
                tag_step = best_step if ('best_step' in locals()) else (self.max_iter - 1)
                tag_loss = best_loss if ('best_loss' in locals()) else internal_history[-1] if len(internal_history) else float('nan')
                out_fname = f"basic_optcam_w_best_step{tag_step}_loss{tag_loss:.6f}.pt"
                out_path = os.path.join(self.save_best_weights_dir, out_fname)
                torch.save({'w': w.detach().cpu(),
                            'best_step': tag_step,
                            'best_loss': tag_loss,
                            'loss_history': internal_history}, out_path)
                print(f"[Basic_OptCAM] Saved best weights to {out_path}")
            except Exception as e:
                print(f"[Basic_OptCAM] Failed to save best weights: {e}")
        try:
            self.last_internal_loss_history = internal_history
        except Exception:
            self.last_internal_loss_history = None
        return norm_saliency_map, new_images

    def __call__(self, images, labels):
        return self.forward(images, labels)

    def __enter__(self):
        return self


class MultiComponentOptCAM:
    """
    Multi-component OptiCAM with explicit consistency constraint.
    
    HƯỚNG 1: Soft Constraint với Regularization (Simplified)
    
    Loss = L_fidelity + λ * L_consistency
    
    Where:
        - L_fidelity: Preserve score of combined/aggregated CAM (DIFF objective with MSE)
                     = (f(x ⊙ CAM_combined) - f(x))²
                     Ensures faithfulness to original model prediction
        
        - L_consistency: Explicit sum constraint
                        = (Σ_k f(x ⊙ mask_k) - f(x))²
                        Ensures sum of component scores equals original score
        
        - λ (lambda_consistency): Weight for consistency constraint
                                 Higher λ → stronger consistency enforcement
    
    Terminology:
        - 'combined' = aggregated mask from all K components (weighted sum)
        - 'fidelity' = faithfulness/preservation of original prediction
        - 'consistency' = additive decomposition constraint
    
    This is simpler than the complex conservation mode with residuals.
    """
    def __init__(self,
                 model,
                 device,
                 target_layers,
                 num_components=3,
                 max_iter=60,
                 learning_rate=1e-3,
                 name_norm='max_min',
                 cat_indices=None,
                 dog_indices=None,
                 lambda_consistency: float = 1.0,
                 learn_scalars: bool = True,
                 mask_scaling: bool = True,
                 min_orig: float = 0.05,
                 monitoring_metric: str = 'mse',
                 all_default_history: str = 'mse',
                 use_lambda_scheduling: bool = False,
                 lambda_start: float = 0.1,
                 lambda_end: float = 0.5,
                 use_mixed_precision: bool = False,
                 max_batch_size: Optional[int] = None,
                 init_method: str = 'adaptive',
                 inference_model=None):
        """
        Args:
            init_method: Initialization strategy for channel weights W
                - 'adaptive': Baseline-compatible (K=1: constant 0.5, K>1: constant + tiny noise)
                - 'random': Random Gaussian (original approach, good for K>1 but not baseline-compatible)
                - 'constant': Pure constant 0.5 (only use for K=1, will fail for K>1 due to symmetry)
            inference_model: Optional separate model for inference (compression mode)
                - If None: uses 'model' for both feature extraction and inference (default, backward compatible)
                - If provided: uses 'model' for feature extraction (mask generation) and 'inference_model' for inference
                - Enables compression: Teacher generates masks, Student performs inference
        """
        # Feature extraction model (for mask generation)
        self.feature_model = model.eval()
        
        # Inference model (for score computation during optimization)
        # If not provided, use same model as feature extraction (backward compatible)
        if inference_model is None:
            self.inference_model = model  # Same model for both (default)
        else:
            self.inference_model = inference_model.eval()  # Separate model (compression mode)
        
        # Legacy alias for backward compatibility (some code may reference self.model)
        self.model = self.inference_model
        self.device = device
        self.k = num_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.name_norm = name_norm
        self.init_method = init_method
        # monitoring_metric can be 'abs' | 'mse' | 'all' (for logging only, NOT optimization)
        self.monitoring_metric = monitoring_metric
        # PURE PROBABILITY APPROACH: Always use probabilities for BOTH fidelity and consistency
        # No use_prob flag needed - optimization is hardcoded to probability space
        # Use feature_model for feature extraction (masks), inference_model used in _score methods
        self.fea_ext = ActivationsAndGradients(self.feature_model, target_layers, None)
        self.last_feature = None  # (B,C,Hf,Wf)
        self.last_loss_history = None     # lịch sử đơn
        self.last_loss_histories = None   # dict nhiều lịch sử khi 'all'
        self.last_component_weights = None
        
        # OPTIMIZATION 1: BATCHED FORWARD PASS (Speed optimization ~75%)
        # Instead of K+1 separate forwards, batch all into 1 forward
        # Safe: mathematically identical, only changes memory layout
        # Auto-detect safe batch size based on GPU memory, user can override
        if max_batch_size is None:
            self.max_batch_size = self._detect_safe_batch_size()
        elif max_batch_size == 0:
            self.max_batch_size = 0  # Disable batching (sequential mode)
        else:
            self.max_batch_size = int(max_batch_size)
        
        # SIMPLIFIED LOSS PARAMETERS
        self.lambda_consistency = float(lambda_consistency)  # λ for consistency constraint
        self.learn_scalars = bool(learn_scalars)
        self.mask_scaling = bool(mask_scaling)
        self.min_orig = float(min_orig)
        
        # LAMBDA SCHEDULING
        self.use_lambda_scheduling = use_lambda_scheduling
        self.lambda_start = float(lambda_start)
        self.lambda_end = float(lambda_end)
        
        # MIXED PRECISION TRAINING
        self.use_amp = use_mixed_precision
        if self.use_amp:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        if cat_indices is not None and dog_indices is not None:
            self.idx_cat = torch.as_tensor(cat_indices, dtype=torch.long).to(self.device)
            self.idx_dog = torch.as_tensor(dog_indices, dtype=torch.long).to(self.device)
        else:
            self.idx_cat = None
            self.idx_dog = None
        # When monitoring_metric == 'all', choose which single-history to expose
        # via `last_loss_history`. Valid values: 'abs', 'mse'. Default
        # remains 'abs' for backward compatibility.
        self.all_default_history = all_default_history

    def _normalization(self, saliency_map):
        if self.name_norm == 'max_min':
            return normlization_max_min(saliency_map)
        if self.name_norm == 'sigmoid':
            return normlization_sigmoid(saliency_map)
        if self.name_norm == 'max':
            return normlization_max(saliency_map)
        raise Exception("Not Implemented")

    def _get_adaptive_lambda(self, step, current_violation=None):
        """Compute adaptive lambda value based on step progress and violation.
        
        Args:
            step: Current optimization step
            current_violation: Optional mean constraint violation (for adaptive boost)
        
        Returns:
            Adaptive lambda value
        """
        if not self.use_lambda_scheduling:
            return self.lambda_consistency
        
        # FIX BUG 3: Reverse scheduling direction
        # Start high (enforce consistency) → End low (focus on fidelity)
        # This helps consistency converge first, then fine-tune fidelity
        progress = step / max(1, self.max_iter - 1)
        # Swap lambda_start and lambda_end in the formula
        lambda_t = self.lambda_start - (self.lambda_start - self.lambda_end) * progress
        
        # Optional: Boost lambda if violation is too high
        if current_violation is not None and current_violation > 3.0:
            lambda_t *= 1.5
        
        return lambda_t
    
    def _score(self, x, y):
        """Compute scores in logit space (for fidelity loss - stable gradients).
        
        This method ALWAYS returns logits regardless of use_prob flag,
        ensuring consistency with OptiCAM baseline's main objective.
        
        Uses inference_model for score computation (enables compression).
        """
        logits = self.inference_model(x)
        if self.idx_cat is not None and self.idx_dog is not None:
            # Aggregate species logits (for cat/dog binary classification)
            logit_cat = logits[:, self.idx_cat].sum(dim=1)
            logit_dog = logits[:, self.idx_dog].sum(dim=1)
            return torch.where(y == 0, logit_cat, logit_dog)
        else:
            # Single class: return logit for target class
            return logits[range(x.size(0)), y]
    
    def _score_prob(self, x, y):
        """Compute scores in probability space (for consistency constraint - mathematically correct).
        
        This method ALWAYS returns probabilities for use in consistency constraint,
        where additivity assumption (Σp_k ≈ p_original) is more valid than logit additivity.
        
        Uses inference_model for score computation (enables compression).
        """
        logits = self.inference_model(x)
        probs = torch.softmax(logits, dim=1)
        if self.idx_cat is not None and self.idx_dog is not None:
            p_cat = probs[:, self.idx_cat].sum(dim=1)
            p_dog = probs[:, self.idx_dog].sum(dim=1)
            return torch.where(y == 0, p_cat, p_dog)
        else:
            return probs[range(x.size(0)), y]
    
    def _detect_safe_batch_size(self) -> int:
        """Auto-detect safe batch size based on GPU memory.
        
        Conservative thresholds to avoid OOM:
        - 4GB GPU → 40 (small batches)
        - 8GB GPU → 60 
        - 11GB GPU → 80 (typical RTX 2080 Ti)
        - 16GB GPU → 120
        - 20GB+ GPU → 150 (high-end cards)
        
        Returns:
            Safe max batch size for batched forward pass
        """
        if not torch.cuda.is_available():
            return 40  # Conservative default for CPU/unknown
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            if gpu_memory_gb < 6:
                return 40
            elif gpu_memory_gb < 10:
                return 60
            elif gpu_memory_gb < 14:
                return 80
            elif gpu_memory_gb < 18:
                return 120
            else:
                return 150
        except Exception:
            return 40  # Fallback if detection fails
    
    def _score_both(self, x, y):
        """OPTIMIZED: Compute both logits and probabilities in ONE forward pass.
        
        Uses inference_model for score computation (enables compression).
        
        Returns:
            tuple: (logit_scores, prob_scores) - both in (B,) shape
        """
        logits = self.inference_model(x)  # Single forward pass
        probs = torch.softmax(logits, dim=1)
        
        if self.idx_cat is not None and self.idx_dog is not None:
            logit_cat = logits[:, self.idx_cat].sum(dim=1)
            logit_dog = logits[:, self.idx_dog].sum(dim=1)
            logit_scores = torch.where(y == 0, logit_cat, logit_dog)
            
            p_cat = probs[:, self.idx_cat].sum(dim=1)
            p_dog = probs[:, self.idx_dog].sum(dim=1)
            prob_scores = torch.where(y == 0, p_cat, p_dog)
        else:
            logit_scores = logits[range(x.size(0)), y]
            prob_scores = probs[range(x.size(0)), y]
        
        return logit_scores, prob_scores

    def _build_masks_from_channel_weights(self, feature, images, Wparam):
        B, C, Hf, Wf = feature.shape
        _, _, H_img, W_img = images.shape
        alpha = torch.softmax(Wparam, dim=2)       # (B,K,C,1,1)
        feat_exp = feature.unsqueeze(1)            # (B,1,C,Hf,Wf)
        sal = (alpha * feat_exp).sum(dim=2)        # (B,K,Hf,Wf)
        sal = sal.unsqueeze(2)                     # (B,K,1,Hf,Wf)
        sal_up = F.interpolate(
            sal.view(B * self.k, 1, Hf, Wf),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        )
        sal_up = self._normalization(sal_up)
        sal_up = sal_up.view(B, self.k, 1, H_img, W_img)
        return sal_up

    def forward(self, images, labels):
        relu = torch.nn.ReLU()
        images = images.to(self.device)
        labels = labels.to(self.device)
        _ = self.fea_ext(images)
        feature = relu(self.fea_ext.activations[0]).to(self.device)
        self.last_feature = feature.detach()
        B, C, Hf, Wf = feature.shape
        
        # ========== ADAPTIVE INITIALIZATION STRATEGY ==========
        # Goal: Baseline-compatible when K=1, symmetry-breaking when K>1
        # Reference: Glorot & Bengio (2010), He et al. (2015) - symmetry breaking in neural networks
        
        if self.init_method == 'constant':
            # Pure constant initialization (only safe for K=1)
            # WARNING: Will fail for K>1 due to symmetry - all components stay identical!
            W_raw = torch.full((B, self.k, C, 1, 1), 0.5, dtype=torch.float, device=self.device)
            if self.k > 1:
                print(f"[WARNING] init_method='constant' with K={self.k}>1 may cause symmetry problem!")
        
        elif self.init_method == 'random':
            # Random Gaussian initialization (original approach)
            # Good for K>1 (breaks symmetry), but not baseline-compatible
            W_raw = torch.randn((B, self.k, C, 1, 1), dtype=torch.float, device=self.device) * 0.01
        
        elif self.init_method == 'adaptive':
            # RECOMMENDED: Adaptive strategy based on K
            if self.k == 1:
                # BASELINE MODE (K=1): Constant init identical to Basic_OptCAM
                # This ensures fair comparison and reproducibility with baseline
                W_raw = torch.full((B, 1, C, 1, 1), 0.5, dtype=torch.float, device=self.device)
            else:
                # MULTI MODE (K>1): Constant + tiny noise for symmetry breaking
                # Start from uniform (fair) but add minimal noise to break symmetry
                W_raw = torch.full((B, self.k, C, 1, 1), 0.5, dtype=torch.float, device=self.device)
                # Add tiny Gaussian noise (1e-4) to break symmetry without changing initialization significantly
                noise = torch.randn_like(W_raw) * 1e-3
                W_raw = W_raw + noise
        
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}. Must be 'adaptive', 'random', or 'constant'")
        
        W_raw.requires_grad = True  # Set requires_grad AFTER initialization to ensure leaf tensor
        if self.learn_scalars:
            init_val = max(0.001, 1.0 / float(self.k))
            beta_raw = torch.full((B, self.k), float(init_val), dtype=torch.float, device=self.device, requires_grad=True)
            params = [W_raw, beta_raw]
        else:
            beta_raw = None
            params = [W_raw]
        optimizer = optim.Adam(params, lr=self.learning_rate)
        
        # Precompute original scores in PROBABILITY space ONLY (Pure Prob Approach):
        # - Probabilities: For BOTH fidelity and consistency (same scale, mathematically correct)
        # This ensures scale matching between loss components and enables proper soft constraint optimization
        with torch.no_grad():
            s_orig_prob = self._score_prob(images, labels).detach()  # (B,) - probabilities only
        internal_history = []
        # NEW TRACKING SYSTEM: Separate histories for detailed analysis
        hist_total = []          # Total loss (optimization target)
        hist_fidelity = []       # L_fidelity component (prob space MSE - preserve confidence)
        hist_consistency = []    # L_consistency component (prob space MSE - decomposition constraint)
        hist_lambda = []         # Lambda schedule (adaptive weight)
        hist_violation = []      # Constraint violation |Σp_k - p_orig|

        for step in range(self.max_iter):
            # Get adaptive lambda for this step
            current_lambda = self._get_adaptive_lambda(step)
            
            # Wrap forward pass in autocast for mixed precision
            if self.use_amp:
                from torch.amp import autocast
                autocast_ctx = autocast('cuda')
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            
            with autocast_ctx:
                # ========== FORWARD PASS ==========
                # Build masks for each component
                masks = self._build_masks_from_channel_weights(feature, images, W_raw)  # (B,K,1,H,W)
                
                # Get scalar weights (optional)
                if self.learn_scalars and beta_raw is not None:
                    beta = torch.nn.functional.softmax(beta_raw, dim=1)
                else:
                    beta = torch.ones((B, self.k), dtype=torch.float, device=self.device) / float(self.k)
                
                # Scale masks by beta if enabled
                if self.mask_scaling:
                    c_resh = beta.view(B, self.k, 1, 1, 1)
                    masks_scaled = (c_resh * masks)
                else:
                    masks_scaled = masks
                
                # Apply masks to get masked images for each component
                masks_apply = masks_scaled.repeat(1, 1, images.size(1), 1, 1)
                x_all = masks_apply * images.unsqueeze(1)  # (B,K,C,H,W)
                
                # Flatten for batch forward
                Bk = B * self.k
                x_all_flat = x_all.view(Bk, images.size(1), images.size(2), images.size(3))
                labels_rep = labels.unsqueeze(1).repeat(1, self.k).view(Bk)
                
                # Build combined/aggregated mask from all K components BEFORE forward
                # This is the weighted sum of individual component masks
                # FIX BUG 2: Use masks_scaled (already weighted) to avoid double scaling
                if self.mask_scaling:
                    weighted_masks = masks_scaled.sum(dim=1)  # (B,1,H,W) - already has beta
                else:
                    # When not scaling, apply beta weighting here
                    if self.learn_scalars and beta_raw is not None:
                        c_resh = beta.view(B, self.k, 1, 1, 1)
                        weighted_masks = (c_resh * masks).sum(dim=1)
                    else:
                        weighted_masks = masks.mean(dim=1)  # Simple average
                combined_mask_clamped = torch.clamp(weighted_masks, min=0.0, max=1.0)
                
                # Apply combined mask to get reconstructed image
                x_combined = combined_mask_clamped.repeat(1, images.size(1), 1, 1) * images
                
                # ========== OPTIMIZATION 1: BATCHED FORWARD PASS ==========
                # OLD: K+1 separate forwards (slow, redundant)
                #   si_all_prob = self._score_prob(x_all_flat, labels_rep)     # K forwards
                #   s_combined_prob = self._score_prob(x_combined, labels)      # 1 forward
                # NEW: 1 batched forward for all (fast, identical results)
                #   Concatenate all inputs → single forward → split results
                # Speed: ~75% faster (4 forwards → 1 forward per iteration)
                # Safety: Mathematically identical, only changes memory layout
                
                total_batch_required = Bk + B  # B*K components + B combined
                
                # Check if GPU can handle batched approach (mitigation: fallback if too large)
                if self.max_batch_size > 0 and total_batch_required <= self.max_batch_size:
                    # BATCHED PATH (FAST): Single forward for all K+1 masks
                    # Concatenate: [component_1, ..., component_K, combined]
                    x_batch = torch.cat([x_all_flat, x_combined], dim=0)      # (B*K+B, C, H, W)
                    labels_batch = torch.cat([labels_rep, labels], dim=0)     # (B*K+B,)
                    
                    # Single forward pass for all
                    scores_batch_prob = self._score_prob(x_batch, labels_batch)  # (B*K+B,)
                    
                    # Split results: first B*K for components, last B for combined
                    si_all_prob = scores_batch_prob[:Bk].view(B, self.k)      # (B,K) - component scores
                    s_combined_prob = scores_batch_prob[Bk:]                   # (B,) - combined score
                else:
                    # SEQUENTIAL PATH (SAFE): Separate forwards if batch too large
                    # Fallback to avoid OOM on small GPUs or large batch_size
                    si_all_prob = self._score_prob(x_all_flat, labels_rep).view(B, self.k)  # (B,K)
                    s_combined_prob = self._score_prob(x_combined, labels)                    # (B,)
                    
                    # Log fallback on first iteration only
                    if step == 0:
                        print(f"[MultiOptCAM] Batch size {total_batch_required} > threshold {self.max_batch_size}, "
                              f"using sequential forwards (safer but ~75% slower)")
                
                # Compute sum of component scores in PROBABILITY space (CONSISTENCY TARGET)
                # Σp(mask_k) ≈ p(original) - mathematically valid assumption
                # FIX BUG 1: When mask_scaling=True, masks already scaled by beta,
                # so scores need beta weighting. When False, masks not scaled, so weight here.
                if self.mask_scaling:
                    sum_component_probs = (si_all_prob * beta).sum(dim=1)  # (B,) - weight by beta
                else:
                    sum_component_probs = si_all_prob.sum(dim=1)  # (B,) - no weighting
                
                # ========== PURE PROBABILITY LOSS FUNCTION (Mathematically Correct) ==========
                # Total Loss = L_fidelity(probs) + λ_t * L_consistency(probs)
                # Both components in SAME SCALE [0,1] → proper soft constraint optimization
                
                # L_fidelity: Fidelity/faithfulness loss - preserve original confidence (probability)
                # Uses PROBABILITIES (consistent with thesis objective: "bảo toàn confidence")
                # Objective: p(x ⊙ combined_mask) ≈ p(x)
                loss_fidelity = torch.pow(s_orig_prob - s_combined_prob, 2).mean()
                
                    # L_consistency: Consistency constraint loss (soft regularization)
                # Uses PROBABILITIES for mathematical correctness (valid additivity)
                # Objective: Σ_k p(x ⊙ mask_k) ≈ p(x) - decomposition constraint
                # Constraint violation in probability space
                constraint_violation_prob = sum_component_probs - s_orig_prob  # (B,)
                loss_consistency = torch.pow(constraint_violation_prob, 2).mean()
                
                # Total loss (weighted sum with adaptive lambda - SAME SCALE!)
                # Lambda now has real meaning: balance faithfulness vs consistency
                loss = loss_fidelity + (current_lambda * loss_consistency)
            
            # Track all components for detailed analysis
            internal_history.append(float(loss.detach().cpu().item()))
            hist_total.append(float(loss.detach().cpu().item()))
            hist_fidelity.append(float(loss_fidelity.detach().cpu().item()))
            hist_consistency.append(float(loss_consistency.detach().cpu().item()))
            hist_lambda.append(float(current_lambda))
            hist_violation.append(float(constraint_violation_prob.abs().mean().detach().cpu().item()))
            
            optimizer.zero_grad()
            
            # Use GradScaler for mixed precision backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward(retain_graph=True)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward(retain_graph=True)
                optimizer.step()
            
            if step % max(1, self.max_iter // 10) == 0:
                # Show constraint violation in probability space (more interpretable: 0-1 range)
                cons_violation_mean = float(constraint_violation_prob.abs().mean().cpu().item())
                lambda_display = current_lambda if self.use_lambda_scheduling else self.lambda_consistency
                print(f"\n[MultiComp] step={step:03d} loss={internal_history[-1]:.6f} "
                      f"L_fidelity(prob)={float(loss_fidelity.detach().cpu().item()):.6f} "
                      f"L_consistency(prob)={float(loss_consistency.detach().cpu().item()):.6f} "
                      f"violation(prob)={cons_violation_mean:.6f} "
                      f"lambda={lambda_display:.4f}")

        masks = self._build_masks_from_channel_weights(feature, images, W_raw)
        
        # NEW: Store detailed component histories for plotting and analysis
        self.last_loss_histories = {
            'total': hist_total,
            'fidelity': hist_fidelity,
            'consistency': hist_consistency,
            'lambda': hist_lambda,
            'violation': hist_violation
        }
        # Backward compatibility: keep internal_history for existing code
        self.last_internal_loss_history = internal_history
        # Default history points to total loss
        self.last_loss_history = hist_total
        if self.learn_scalars and beta_raw is not None:
            try:
                self.last_component_weights = beta.detach()
            except Exception:
                self.last_component_weights = torch.nn.functional.softmax(beta_raw, dim=1).detach()
        else:
            self.last_component_weights = None
        return masks.detach()

    def __call__(self, images, labels):
        return self.forward(images, labels)


# === Advanced metrics for CAM (Insertion/Deletion AUC, AOPC) ==================
# (Giữ nguyên phần advanced_metrics cũ)
def _am_make_baseline(img_float: np.ndarray, mode: str = "black", blur_ksize: int = 11) -> np.ndarray:
    if mode == "black":
        return np.zeros_like(img_float, dtype=np.float32)
    if mode == "blur":
        import cv2  # lazy import
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur((img_float * 255.0).astype(np.uint8), (k, k), 0)
        return blur.astype(np.float32) / 255.0
    raise ValueError(f"Unknown baseline mode: {mode}")

def _am_score_np_image(model, device, x_np: np.ndarray, label_scalar: int,
                       idx_cat=None, idx_dog=None, use_logit: bool = False) -> float:
    x_t = torch.from_numpy(x_np.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x_t)
        if (idx_cat is not None) and (idx_dog is not None) and isinstance(idx_cat, torch.Tensor) and isinstance(idx_dog, torch.Tensor) and idx_cat.numel() > 0 and idx_dog.numel() > 0:
            probs = torch.softmax(logits, dim=1)
            p_cat = probs[0, idx_cat].sum().item()
            p_dog = probs[0, idx_dog].sum().item()
            return float(p_cat if int(label_scalar) == 0 else p_dog)
        if use_logit:
            return float(logits[0, int(label_scalar)].item())
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, int(label_scalar)].item())

def _am_build_topk_masks_from_ranking(s_map: np.ndarray, steps: int) -> list:
    H, W = s_map.shape
    flat = s_map.reshape(-1)
    order = np.argsort(flat)[::-1]
    N = flat.size
    masks = []
    for t in range(0, steps + 1):
        k = int(round(N * (t / steps)))
        M = np.zeros(N, dtype=np.float32)
        if k > 0:
            M[order[:k]] = 1.0
        masks.append(M.reshape(H, W))
    return masks

def _am_normalize01(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = m.astype(np.float32)
    m = m - np.min(m)
    denom = float(np.max(m))
    if denom < eps:
        return np.zeros_like(m, dtype=np.float32)
    return m / (denom + eps)

def _am_compute_id_curves_core(
    img_float: np.ndarray,
    union_mask01: np.ndarray,
    model,
    device,
    label_scalar: int,
    idx_cat=None,
    idx_dog=None,
    steps: int = 100,
    baseline_mode: str = "black",
    blur_ksize: int = 11,
    use_logit: bool = False
):
    union_norm = _am_normalize01(union_mask01)
    topk_masks = _am_build_topk_masks_from_ranking(union_norm, steps)
    baseline = _am_make_baseline(img_float, baseline_mode, blur_ksize)
    y = _am_score_np_image(model, device, img_float, label_scalar, idx_cat, idx_dog, use_logit)
    bscore = _am_score_np_image(model, device, baseline, label_scalar, idx_cat, idx_dog, use_logit)
    ins_vals = []
    del_vals = []
    for Mt in topk_masks:
        Mt3 = Mt[:, :, None]
        x_ins = baseline * (1.0 - Mt3) + img_float * Mt3
        s_ins = _am_score_np_image(model, device, x_ins, label_scalar, idx_cat, idx_dog, use_logit)
        ins_vals.append(s_ins)
        x_del = img_float * (1.0 - Mt3) + baseline * Mt3
        s_del = _am_score_np_image(model, device, x_del, label_scalar, idx_cat, idx_dog, use_logit)
        del_vals.append(s_del)
    ins_curve = np.array(ins_vals, dtype=np.float32)
    del_curve = np.array(del_vals, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, num=len(ins_curve), dtype=np.float32)
    ins_auc = float(np.trapz(ins_curve, xs))
    del_auc = float(np.trapz(del_curve, xs))
    if len(ins_curve) > 1:
        aopc_ins = float(np.mean(ins_curve[1:] - bscore))
        aopc_del = float(np.mean(y - del_curve[1:]))
    else:
        aopc_ins = 0.0
        aopc_del = 0.0
    return ins_curve, del_curve, ins_auc, del_auc, aopc_ins, aopc_del, y, bscore

def advanced_metrics(
    img_float: np.ndarray,
    union_mask01: np.ndarray,
    model,
    device,
    label_scalar: int,
    *,
    idx_cat=None,
    idx_dog=None,
    use_logit: bool = False,
    steps: int = 100,
    baseline: str = "black",
    blur_ksize: int = 11,
    enable_ins_auc: bool = True,
    enable_del_auc: bool = True,
    enable_aopc_ins: bool = True,
    enable_aopc_del: bool = True,
    return_curves: bool = False
) -> dict:
    union_mask01 = _am_normalize01(union_mask01)
    ins_curve, del_curve, ins_auc, del_auc, aopc_ins, aopc_del, y, bscore = _am_compute_id_curves_core(
        img_float=img_float,
        union_mask01=union_mask01,
        model=model,
        device=device,
        label_scalar=int(label_scalar),
        idx_cat=idx_cat,
        idx_dog=idx_dog,
        steps=int(steps),
        baseline_mode=str(baseline),
        blur_ksize=int(blur_ksize),
        use_logit=bool(use_logit)
    )
    out = {
        "orig_score": y,
        "baseline_score": bscore,
        "id_steps": int(steps),
        "id_baseline": str(baseline),
        "id_blur_ksize": int(blur_ksize),
    }
    if enable_ins_auc:
        out["auc_insertion"] = ins_auc
    if enable_del_auc:
        out["auc_deletion"] = del_auc
    if enable_aopc_ins:
        out["aopc_insertion"] = aopc_ins
    if enable_aopc_del:
        out["aopc_deletion"] = aopc_del
    if return_curves:
        out["insertion_curve"] = ins_curve
        out["deletion_curve"] = del_curve
    return out