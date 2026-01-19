import os
import time
import csv
import shutil
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from absl import flags, app

import sys
sys.path.append("pytorch_grad_cam")
from pytorch_grad_cam.utils.image import show_cam_on_image

from imagenet_loader import ImageNetLoader

from util import Preprocessing_Layer, MultiComponentOptCAM

# Import logger helpers: per-loss row + summary row (append-only TXT, không .tsv)
from tools.generate_log_tables import append_row_from_flags, append_summary_row_from_multi

from tools.compute_metrics import (
    build_cat_dog_index_sets,
    compute_metrics_per_sample,
    _normalize_map01,
    _build_combined_mask,
    _make_comp_display_map,
    advanced_metrics,
    ID_STEPS,
    ID_BASELINE,
    ID_BLUR_KSIZE,
)

FLAGS = flags.FLAGS

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def sanitize_name(name: str) -> str:
    base = os.path.basename(str(name))
    return os.path.splitext(base.replace('/', '_').replace('\\', '_'))[0]

def load_student_model(student_path, student_arch, device):
    """
    Load optional student model for speed optimization (inference acceleration).
    
    USE CASE: Speed Optimization
    - Teacher (ResNet50): Feature extraction for mask generation (accurate)
    - Student (ResNet18/34): Fast inference during optimization
    - Metrics: Evaluated on Teacher (reflects deployment performance)
    
    Returns:
        - model: Full student model with preprocessing
        - info: Dict with student model metadata
    """
    print(f"\n[Speed Optimization] Loading student model: {student_arch} from {student_path}")
    
    checkpoint = torch.load(student_path, map_location=device)
    
    # Detect checkpoint format
    is_direct_state_dict = isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint
    
    # Create student architecture
    if student_arch == 'resnet18':
        try:
            from torchvision.models import ResNet18_Weights
            student_core = models.resnet18(weights=None)
        except:
            student_core = models.resnet18(pretrained=False)
    elif student_arch == 'resnet34':
        try:
            from torchvision.models import ResNet34_Weights
            student_core = models.resnet34(weights=None)
        except:
            student_core = models.resnet34(pretrained=False)
    else:
        raise ValueError(f"Unknown student architecture: {student_arch}")
    
    # Load weights
    if is_direct_state_dict:
        print(f"  [Format] Direct state_dict (train_student_classifier.py)")
        student_core.load_state_dict(checkpoint)
        info = {
            'architecture': student_arch,
            'format': 'direct_state_dict',
            'agreement': 'N/A'
        }
    else:
        print(f"  [Format] Metadata format (train_student_model.py)")
        student_core.load_state_dict(checkpoint['model_state_dict'])
        info = {
            'architecture': checkpoint.get('architecture', student_arch),
            'format': 'metadata',
            'epoch': checkpoint.get('epoch', 'N/A'),
            'agreement': checkpoint.get('agreement', 'N/A')
        }
    
    # Wrap with preprocessing
    preprocess_layer = Preprocessing_Layer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    student_model = nn.Sequential(preprocess_layer, student_core).to(device).eval()
    
    print(f"  [Loaded] {student_arch} ({sum(p.numel() for p in student_core.parameters())/1e6:.1f}M params)")
    if info.get('agreement') != 'N/A':
        print(f"  [Agreement] {info['agreement']:.2f}%" if isinstance(info['agreement'], float) else f"  [Agreement] {info['agreement']}")
    
    return student_model, info

def precreate_loss_folders(metrics_root: str, losses=('abs','mse')):
    # Create only requested loss subfolders (default to all)
    for loss in losses:
        ensure_dir(os.path.join(metrics_root, loss))

def migrate_misplaced_files(base_dir: str, losses=('abs','mse')):
    metrics_root = os.path.join(base_dir, "metrics")
    plot_dir = os.path.join(base_dir, "plot")
    scalars_dir = os.path.join(base_dir, "scalars")
    ensure_dir(metrics_root); ensure_dir(plot_dir); ensure_dir(scalars_dir)
    for loss in losses:
        dst_dir = os.path.join(metrics_root, loss)
        ensure_dir(dst_dir)
        patterns = [
            f"metrics_summary_{loss}.txt",
            f"metrics_per_image_{loss}.csv",
            f"batch_loss_histories_{loss}.npy",
            f"batch_final_losses_{loss}.npy",
            f"batch_times_{loss}.npy",
            f"overall_loss_iterations_{loss}.png",
            f"overall_loss_batches_{loss}.png",
            f"overall_loss_iterations_{loss}_regen.png",
            f"overall_loss_batches_{loss}_regen.png",
        ]
        for pat in patterns:
            src = os.path.join(metrics_root, pat)
            if os.path.exists(src):
                try:
                    shutil.move(src, os.path.join(dst_dir, pat))
                except Exception:
                    pass
    for loss in losses:
        old_plot = os.path.join(metrics_root, loss, "plot")
        if os.path.isdir(old_plot):
            # Move legacy per-image loss files; include new Can/Inter patterns
            for pattern in (f"*_Loss_{loss}.npy", f"*_Can_Loss_{loss}.npy", f"*_Inter_Loss_{loss}.npy"):
                for npy in glob.glob(os.path.join(old_plot, pattern)):
                    try: shutil.move(npy, os.path.join(plot_dir, os.path.basename(npy)))
                    except Exception: pass
    for loss in losses:
        old_scalars = os.path.join(metrics_root, loss, "scalars")
        if os.path.isdir(old_scalars):
            for f in glob.glob(os.path.join(old_scalars, "*.npy")):
                try: shutil.move(f, os.path.join(scalars_dir, os.path.basename(f)))
                except Exception: pass

def main(_):
    global_start_time = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    running_all = (FLAGS.monitoring_metric == "all")
    min_orig = FLAGS.min_orig
    only_correct = FLAGS.only_correct
    print(f"[Metric Settings] min_orig={min_orig}  only_correct={only_correct}  eval_reduce={FLAGS.eval_reduce}")
    print(f"[Viz Settings] combine_rule={FLAGS.combine_rule}  viz_from_combined={FLAGS.viz_from_combined}  weighted_temp={FLAGS.weighted_temp}")
    print(f"[Comp Viz] mode={FLAGS.comp_viz_mode}  min_comp_share={FLAGS.min_comp_share}")
    print(f"[AdvMetrics] toggles: insAUC={FLAGS.am_enable_ins_auc} delAUC={FLAGS.am_enable_del_auc} AOPCins={FLAGS.am_enable_aopc_ins} AOPCdel={FLAGS.am_enable_aopc_del}")
    print(f"[AdvMetrics] ID config: steps={ID_STEPS} baseline={ID_BASELINE} blur_k={ID_BLUR_KSIZE}")
    print(f"[Loss] monitoring_metric: {'abs, mse (simultaneous)' if running_all else FLAGS.monitoring_metric}")

    compute_adv_master = any([
        FLAGS.am_enable_ins_auc,
        FLAGS.am_enable_del_auc,
        FLAGS.am_enable_aopc_ins,
        FLAGS.am_enable_aopc_del,
    ])
    print(f"[AdvMetrics] master enabled: {compute_adv_master}")

    # Backbone (Teacher model - always ResNet50)
    try:
        from torchvision.models import ResNet50_Weights
        teacher_core = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError):
        teacher_core = models.resnet50(pretrained=True)
    preprocess_layer = Preprocessing_Layer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    teacher_model = nn.Sequential(preprocess_layer, teacher_core).to(device).eval()
    print(f"[Teacher Model] ResNet50 loaded ({sum(p.numel() for p in teacher_core.parameters())/1e6:.1f}M params)")
    
    # Optional: Load student model for speed optimization
    student_model = None
    student_info = None
    if hasattr(FLAGS, 'student_path') and FLAGS.student_path:
        student_model, student_info = load_student_model(
            FLAGS.student_path,
            FLAGS.student_arch,
            device
        )
        print(f"\n[Speed Optimization Mode]")
        print(f"  Feature extraction (masks): Teacher (ResNet50) - Accurate")
        print(f"  Optimization inference:     Student ({FLAGS.student_arch}) - Fast")
        print(f"  Metrics evaluation:         Teacher (ResNet50) - Accurate")
        print(f"  Expected: 2-3x faster optimization with same accuracy\n")
    else:
        print(f"[Standard Mode] Using single model (ResNet50) for all operations\n")
    
    # Use teacher_model for both feature extraction and inference (backward compatible)
    # If student is provided, it will be passed to OptCAM separately
    model = teacher_model

    # Binary cat/dog dataset: 2 classes (0=cat, 1=dog)
    # For binary classification, we use simple single-class indexing
    # No need for multi-species aggregation like ImageNet
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V1.meta.get('categories', None)
    except Exception:
        categories = None
    
    # Build indices for compatibility with util.py _score() method
    # For binary dataset: class 0 (cat) and class 1 (dog)
    if categories is not None:
        cat_indices, dog_indices = build_cat_dog_index_sets(categories)
    else:
        # Binary dataset: set to empty to trigger single-class mode
        cat_indices, dog_indices = [], []
    idx_cat = torch.tensor(cat_indices, dtype=torch.long, device=device) if len(cat_indices)>0 else torch.tensor([],dtype=torch.long,device=device)
    idx_dog = torch.tensor(dog_indices, dtype=torch.long, device=device) if len(dog_indices)>0 else torch.tensor([],dtype=torch.long,device=device)

    # Data
    valdir = './images/'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = ImageNetLoader(valdir, './revisited_imagenet_2012_val.csv', transform)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=use_cuda)
    to_pil = transforms.ToPILImage()

    # Dirs
    base_dir = os.path.join('./results', FLAGS.name_path)
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')
    metrics_root = os.path.join(base_dir, 'metrics')
    plot_dir = os.path.join(base_dir, 'plot')
    scalars_dir = os.path.join(base_dir, 'scalars')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    for d in [base_dir, images_dir, masks_dir, metrics_root, plot_dir, scalars_dir, checkpoint_dir]:
        ensure_dir(d)
    # Create only requested loss subfolders
    losses_needed = ('abs','mse') if running_all else (FLAGS.monitoring_metric,)
    precreate_loss_folders(metrics_root, losses=losses_needed)

    # OptCAM with simplified consistency constraint (HƯỚNG 1)
    target_layers = [model[1].layer4[-1]]
    # NOTE: Optimization is HARDCODED to pure probability space (removed use_prob parameter)
    # FLAGS.use_logit below ONLY affects metrics computation (AD/AG/AI), NOT optimization
    
    # Handle max_batch_size: -1 means None (auto-detect), 0 means disable
    max_batch_param = None if FLAGS.max_batch_size == -1 else FLAGS.max_batch_size
    
    OptCAM = MultiComponentOptCAM(
        model=model,  # Teacher for feature extraction (masks)
        device=device,
        target_layers=target_layers,
        num_components=FLAGS.num_masks,
        max_iter=FLAGS.max_iter,
        learning_rate=FLAGS.learning_rate,
        name_norm=FLAGS.name_norm,
        cat_indices=idx_cat,
        dog_indices=idx_dog,
        lambda_consistency=FLAGS.lambda_consistency,
        mask_scaling=FLAGS.mask_scaling,
        learn_scalars=True,
        min_orig=FLAGS.min_orig,
        monitoring_metric=FLAGS.monitoring_metric if FLAGS.monitoring_metric != 'all' else 'abs',
        use_lambda_scheduling=FLAGS.use_lambda_scheduling,
        lambda_start=FLAGS.lambda_start,
        lambda_end=FLAGS.lambda_end,
        use_mixed_precision=FLAGS.use_mixed_precision,
        max_batch_size=max_batch_param,  # OPTIMIZATION 1: Auto-detect or user override
        init_method=FLAGS.init_method,  # NEW: Initialization strategy
        inference_model=student_model  # Optional: Student for fast inference during optimization
    )
    
    # Log optimization settings
    if OptCAM.max_batch_size > 0:
        print(f"[Optimization] Batched forward enabled (max_batch_size={OptCAM.max_batch_size})")
    else:
        print(f"[Optimization] Batched forward disabled (using sequential mode)")
    
    # Log initialization strategy
    print(f"[Initialization] Method: {FLAGS.init_method}")
    if FLAGS.init_method == 'adaptive':
        if FLAGS.num_masks == 1:
            print(f"  → K=1: Baseline-compatible constant init (0.5)")
        else:
            print(f"  → K={FLAGS.num_masks}: Constant (0.5) + tiny noise (1e-4) for symmetry breaking")
    elif FLAGS.init_method == 'random':
        print(f"  → Gaussian random init (σ=0.01) - not baseline-compatible")
    elif FLAGS.init_method == 'constant':
        print(f"  → Pure constant init (0.5)")
        if FLAGS.num_masks > 1:
            print(f"  ⚠️  WARNING: K={FLAGS.num_masks}>1 with constant init may cause symmetry problem!")
    
    if not running_all:
        OptCAM.monitoring_metric = FLAGS.monitoring_metric

    plotting_enabled = FLAGS.save_loss_plot
    if plotting_enabled:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib failed: {e}; disable plotting.")
            plotting_enabled = False
    save_npy_flag = bool(FLAGS.save_loss_npy)

    # Accumulators (primary metrics) — shared for all losses (Paper Equations 13-15)
    raw_samples = 0
    counted_samples = 0  # N in paper formulas
    sum_ad = 0.0  # Σ |p-o|+ / p (ALL samples)
    sum_ai = 0  # Σ 1_{o>p} (count increases)
    sum_ag = 0.0  # Σ |o-p|+ / (1-p) (ALL samples)
    per_image_rows = []
    total_saliency_time = 0.0
    total_original_infer_time = 0.0
    timed_batches = 0
    total_wall_time = 0.0
    cons_err_accum = 0.0
    cons_n = 0

    # Loss histories - NEW: component-level tracking
    batch_loss_histories_by = {}
    batch_final_losses_by = {}
    batch_times = []

    # Advanced metrics sums
    sum_auc_ins = 0.0
    sum_auc_del = 0.0
    sum_aopc_ins = 0.0
    sum_aopc_del = 0.0
    am_count_used = 0

    # Load checkpoint if resume requested
    start_batch_idx = FLAGS.start_batch
    if FLAGS.resume_checkpoint and os.path.exists(FLAGS.resume_checkpoint):
        print(f"[Checkpoint] Loading from {FLAGS.resume_checkpoint}")
        try:
            ckpt = torch.load(FLAGS.resume_checkpoint, map_location=device)
            start_batch_idx = ckpt.get('batch_idx', 0) + 1  # Resume from next batch
            raw_samples = ckpt.get('raw_samples', 0)
            counted_samples = ckpt.get('counted_samples', 0)
            sum_ad = ckpt.get('sum_ad', 0.0)
            sum_ai = ckpt.get('sum_ai', 0)
            sum_ag = ckpt.get('sum_ag', 0.0)
            per_image_rows = ckpt.get('per_image_rows', [])
            total_saliency_time = ckpt.get('total_saliency_time', 0.0)
            total_original_infer_time = ckpt.get('total_original_infer_time', 0.0)
            timed_batches = ckpt.get('timed_batches', 0)
            total_wall_time = ckpt.get('total_wall_time', 0.0)
            cons_err_accum = ckpt.get('cons_err_accum', 0.0)
            cons_n = ckpt.get('cons_n', 0)
            sum_auc_ins = ckpt.get('sum_auc_ins', 0.0)
            sum_auc_del = ckpt.get('sum_auc_del', 0.0)
            sum_aopc_ins = ckpt.get('sum_aopc_ins', 0.0)
            sum_aopc_del = ckpt.get('sum_aopc_del', 0.0)
            am_count_used = ckpt.get('am_count_used', 0)
            batch_loss_histories_by = ckpt.get('batch_loss_histories_by', {})
            batch_final_losses_by = ckpt.get('batch_final_losses_by', {})
            batch_times = ckpt.get('batch_times', [])
            print(f"[Checkpoint] Resuming from batch {start_batch_idx} (processed {counted_samples} samples)")
        except Exception as e:
            print(f"[Checkpoint] Failed to load: {e}. Starting from batch {start_batch_idx}")

    for batch_idx, (images, labels, file_names) in enumerate(val_loader):
        # Skip batches already processed
        if batch_idx < start_batch_idx:
            continue
        t_batch0 = time.time()
        images = images.to(device); labels = labels.to(device)

        # Original forward - consistent with _score() method
        t0 = time.time()
        with torch.no_grad():
            logits_orig = model(images)
            labels = labels.long()
            
            # Get predictions (argmax for all cases)
            if idx_cat.numel()>0 and idx_dog.numel()>0:
                # Multi-species ImageNet mode (sum logits/probs per species)
                if FLAGS.use_logit:
                    p_cat = logits_orig[:, idx_cat].sum(dim=1)
                    p_dog = logits_orig[:, idx_dog].sum(dim=1)
                else:
                    probs_orig = torch.softmax(logits_orig, dim=1)
                    p_cat = probs_orig[:, idx_cat].sum(dim=1)
                    p_dog = probs_orig[:, idx_dog].sum(dim=1)
                preds = torch.where(p_cat >= p_dog,
                                    torch.zeros_like(p_cat,dtype=torch.long),
                                    torch.ones_like(p_dog,dtype=torch.long))
                orig_scores = torch.where(labels==0, p_cat, p_dog)
            else:
                # Binary/single-class mode (match _score() logic)
                preds = torch.argmax(logits_orig, dim=1)
                if FLAGS.use_logit:
                    # Use logits for target class (unbounded)
                    orig_scores = logits_orig[range(images.size(0)), labels]
                else:
                    # Use probabilities for target class (bounded [0,1])
                    probs_orig = torch.softmax(logits_orig, dim=1)
                    orig_scores = probs_orig[range(images.size(0)), labels]
        total_original_infer_time += (time.time()-t0)
        raw_samples += images.size(0)

        if only_correct:
            mask = (preds == labels)
            if mask.sum()==0:
                print(f"[Batch {batch_idx}] skip (no correct)")
                continue
            images = images[mask]; labels = labels[mask]
            orig_scores = orig_scores[mask]
            file_names = [fn for i,fn in enumerate(file_names) if mask[i]]

        if len(file_names)==0:
            continue

        # Optimize once
        sal_start = time.time()
        masks = OptCAM(images, labels)  # (B,K,1,H,W); histories stored inside
        total_saliency_time += (time.time()-sal_start)

        # NEW: Extract detailed component histories (replaces old abs/mse tracking)
        if getattr(OptCAM, 'last_loss_histories', None):
            histories = OptCAM.last_loss_histories
            # Store per-batch histories for all components
            for key in ['total', 'fidelity', 'consistency', 'lambda', 'violation']:
                hist = list(histories.get(key, []))
                if len(hist) > 0:
                    if key not in batch_loss_histories_by:
                        batch_loss_histories_by[key] = []
                        batch_final_losses_by[key] = []
                    batch_loss_histories_by[key].append(hist)
                    batch_final_losses_by[key].append(hist[-1])

        B = images.size(0); K = masks.size(1); timed_batches += 1

        # Vectorized masked forwards
        masks_apply_batch = masks.repeat(1,1,images.size(1),1,1)
        masked_images_batch = masks_apply_batch * images.unsqueeze(1)
        Bk = B*K
        masked_images_flat_batch = masked_images_batch.view(Bk, images.size(1), images.size(2), images.size(3))
        labels_rep_batch = labels.unsqueeze(1).repeat(1,K).view(Bk)

        # Masked forward - consistent with _score() and respect use_logit flag
        with torch.no_grad():
            logits_masked_all = model(masked_images_flat_batch)
            
            if idx_cat.numel()>0 and idx_dog.numel()>0:
                # Multi-species ImageNet mode
                if FLAGS.use_logit:
                    p_cat_masked_all = logits_masked_all[:, idx_cat].sum(dim=1)
                    p_dog_masked_all = logits_masked_all[:, idx_dog].sum(dim=1)
                else:
                    probs_masked_all = torch.softmax(logits_masked_all, dim=1)
                    p_cat_masked_all = probs_masked_all[:, idx_cat].sum(dim=1)
                    p_dog_masked_all = probs_masked_all[:, idx_dog].sum(dim=1)
                s_all = torch.where(labels_rep_batch==0, p_cat_masked_all, p_dog_masked_all)
            else:
                # Binary/single-class mode (match _score() logic)
                if FLAGS.use_logit:
                    # Use logits (unbounded)
                    s_all = logits_masked_all[range(Bk), labels_rep_batch]
                else:
                    # Use probabilities (bounded [0,1])
                    probs_masked_all = torch.softmax(logits_masked_all, dim=1)
                    s_all = probs_masked_all[range(Bk), labels_rep_batch]
        s_all = s_all.view(B,K)

        # Debug: compute reconstruction breakdown if requested
        if FLAGS.dbg_recon:
            try:
                with torch.no_grad():
                    # attempt to get learned scalars from OptCAM if present
                    c_pos = getattr(OptCAM, 'last_component_weights', None)
                    if c_pos is None:
                        c_pos = torch.ones((B, K), dtype=torch.float, device=device) / float(K)
                    if FLAGS.mask_scaling:
                        c_resh_dbg = c_pos.view(B, K, 1, 1, 1)
                        masks_for_dbg = (c_resh_dbg * masks)
                    else:
                        masks_for_dbg = masks
                    # weighted masks and residual (use clamped combined mask)
                    try:
                        weighted_masks_dbg = torch.clamp((c_resh_dbg * masks).sum(dim=1), min=0.0, max=1.0)
                    except Exception:
                        weighted_masks_dbg = torch.clamp(masks.sum(dim=1), min=0.0, max=1.0)
                    residual_mask_dbg = torch.clamp(1.0 - weighted_masks_dbg, min=0.0, max=1.0)
                    x_res_dbg = residual_mask_dbg * images
                    s_res_dbg = OptCAM._score(x_res_dbg, labels)
                    # combined-based score
                    try:
                        x_combined_dbg = weighted_masks_dbg * images
                        x_combined_dbg = x_combined_dbg.view(B, images.size(1), images.size(2), images.size(3))
                    except Exception:
                        x_combined_dbg = weighted_masks_dbg * images
                    s_combined_dbg = OptCAM._score(x_combined_dbg, labels)
                    # weighted_sum for reference (per-component scores computed earlier)
                    if FLAGS.mask_scaling:
                        weighted_sum_dbg = s_all.sum(dim=1)
                    else:
                        weighted_sum_dbg = (s_all * c_pos).sum(dim=1)
                    # saturation-aware reconstruction
                    recon_dbg = s_combined_dbg + (1.0 - s_combined_dbg) * s_res_dbg
                    # residual target (conditional fraction)
                    eps_dbg = 1e-6
                    den_dbg = (1.0 - s_combined_dbg.detach()).clamp(min=eps_dbg)
                    target_res_dbg = ((orig_scores - s_combined_dbg.detach()) / den_dbg).clamp(min=0.0, max=1.0)
                    # losses (debug compute with MSE for comparability)
                    loss_cons_dbg = torch.nn.functional.mse_loss(recon_dbg, orig_scores)
                    loss_res_dbg = torch.nn.functional.mse_loss(s_res_dbg, target_res_dbg)
                    print(f"[DBG_RECON] batch={batch_idx} orig_mean={float(orig_scores.mean().cpu()):.6f} "
                          f"s_combined_mean={float(s_combined_dbg.mean().cpu()):.6f} weighted_sum_mean={float(weighted_sum_dbg.mean().cpu()):.6f} "
                          f"s_res_mean={float(s_res_dbg.mean().cpu()):.6f} recon_mean={float(recon_dbg.mean().cpu()):.6f} "
                          f"tres_mean={float(target_res_dbg.mean().cpu()):.6f} loss_cons={float(loss_cons_dbg.cpu()):.6f} "
                          f"loss_res={float(loss_res_dbg.cpu()):.6f}")
            except Exception as e:
                print(f"[DBG_RECON] failed to compute diagnostics: {e}")

        for i in range(B):
            # CRITICAL: For AD/AI/AG metrics, y and o MUST be probabilities (per paper)
            # even if optimization used logits (--use_logit flag)
            if FLAGS.use_logit:
                # orig_scores are logits, convert to probability for metrics
                with torch.no_grad():
                    logit_val = orig_scores[i].item()
                    # Convert single logit to probability using softmax with dummy class
                    # This is approximate; ideally should use full model output
                    # But for consistency, we recompute from original image
                    logits_single = model(images[i:i+1])
                    probs_single = torch.softmax(logits_single, dim=1)
                    if idx_cat.numel()>0 and idx_dog.numel()>0:
                        p_cat_single = probs_single[0, idx_cat].sum().item()
                        p_dog_single = probs_single[0, idx_dog].sum().item()
                        y = p_cat_single if int(labels[i].item())==0 else p_dog_single
                    else:
                        y = probs_single[0, labels[i].item()].item()
            else:
                # Already probability
                y = float(orig_scores[i].item())
            
            img_uint8 = np.array(to_pil(images[i].detach().cpu()))
            img_float = img_uint8.astype(np.float32)/255.0

            masks_np_list = []
            for k in range(K):
                m = masks[i,k,0].detach().cpu().numpy()
                masks_np_list.append(m)
                np.save(os.path.join(masks_dir, f"{sanitize_name(file_names[i])}_m{k+1}.npy"), m)
            masks_np = np.stack(masks_np_list, axis=0)

            s_list = [float(x) for x in s_all[i].detach().cpu().tolist()]
            if len(s_list)!=K:
                s_list = (s_list + [0.0]*(K-len(s_list)))[:K]

            # Combined mask visualization
            m_combined_viz = _build_combined_mask(masks_np, mode=FLAGS.combine_rule, scores=s_list, temp=FLAGS.weighted_temp)
            m_combined_viz = _normalize_map01(m_combined_viz)
            cam_rgb = show_cam_on_image(img_float, m_combined_viz, use_rgb=True)
            if cam_rgb.dtype!=np.uint8:
                cam_rgb=(np.clip(cam_rgb,0,1)*255).astype(np.uint8)
            cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

            # Component overlays
            sum_scores = float(np.sum(s_list))+1e-6
            for k in range(K):
                share = float(s_list[k])/sum_scores
                m_disp = _make_comp_display_map(masks_np[k], share, m_combined_viz, FLAGS.comp_viz_mode)
                comp_rgb = show_cam_on_image(img_float, m_disp, use_rgb=True)
                if comp_rgb.dtype!=np.uint8:
                    comp_rgb=(np.clip(comp_rgb,0,1)*255).astype(np.uint8)
                out_name = (f"{sanitize_name(file_names[i])}_m{k+1}_share{share:.2f}_low.png"
                            if FLAGS.comp_viz_mode!='raw' and share<FLAGS.min_comp_share
                            else f"{sanitize_name(file_names[i])}_m{k+1}_share{share:.2f}.png")
                cv2.imwrite(os.path.join(images_dir,out_name), cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))
            if FLAGS.viz_from_combined:
                cv2.imwrite(os.path.join(images_dir, f"{sanitize_name(file_names[i])}_mall.png"), cam_bgr)

            # Reduced score cho metrics
            # CRITICAL: For AD/AI/AG metrics, ALWAYS use probabilities (as defined in Chattopadhay et al. 2018)
            # regardless of --use_logit flag (which controls optimization only)
            if FLAGS.eval_reduce=="avg":
                # If s_list are logits, need to convert to probs first for proper averaging
                if FLAGS.use_logit:
                    # Convert component logits to probs for meaningful average
                    with torch.no_grad():
                        s_logits = torch.tensor(s_list, device=device)
                        s_probs = torch.softmax(torch.stack([s_logits, torch.zeros_like(s_logits)], dim=-1), dim=-1)[:, 0]
                        o = float(s_probs.mean().cpu().item())
                else:
                    o = float(np.mean(s_list))
            elif FLAGS.eval_reduce=="sum":
                o = float(np.sum(s_list))
            else:
                m_combined_metric = _build_combined_mask(masks_np, mode=FLAGS.combine_rule, scores=s_list, temp=FLAGS.weighted_temp)
                m_combined_metric = _normalize_map01(m_combined_metric)
                x_combined_np = img_float * m_combined_metric[:, :, None]
                x_combined_tensor = torch.from_numpy(x_combined_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits_combined = model(x_combined_tensor)
                    # ALWAYS convert to probability for AD/AI/AG metrics (per paper definition)
                    probs_combined = torch.softmax(logits_combined, dim=1)
                    if idx_cat.numel()>0 and idx_dog.numel()>0:
                        p_cat_combined = probs_combined[0, idx_cat].sum().item()
                        p_dog_combined = probs_combined[0, idx_dog].sum().item()
                        o = p_cat_combined if int(labels[i].item())==0 else p_dog_combined
                    else:
                        o = probs_combined[0, labels[i].item()].item()

            # Consistency error calculation (MUST match training formula)
            # When mask_scaling=True: training uses Σ(β_k * c_k), so evaluation should too
            # When mask_scaling=False: training uses Σc_k (no weighting)
            if FLAGS.mask_scaling:
                # Get learned beta weights from optimizer
                try:
                    if hasattr(OptCAM, 'last_component_weights') and OptCAM.last_component_weights is not None:
                        beta_weights = OptCAM.last_component_weights[i].detach().cpu().numpy()
                        weighted_sum = float(np.sum(np.array(s_list) * beta_weights))
                        cons_err = abs(weighted_sum - y)
                    else:
                        # Fallback: unweighted sum (should not happen in normal execution)
                        cons_err = abs(float(np.sum(s_list)) - y)
                except Exception:
                    # Fallback on error
                    cons_err = abs(float(np.sum(s_list)) - y)
            else:
                # No mask scaling: use unweighted sum (matches training)
                cons_err = abs(float(np.sum(s_list)) - y)
            
            cons_err_accum += cons_err
            cons_n += 1

            ad_value, ai_flag, ag_value, used_flag = compute_metrics_per_sample(y, o, min_orig=min_orig)
            if used_flag:
                counted_samples += 1
                sum_ad += ad_value  # Accumulate AD for ALL samples
                sum_ai += ai_flag   # Count increases
                sum_ag += ag_value  # Accumulate AG for ALL samples

            # Advanced metrics
            am = {}
            if compute_adv_master:
                m_combined_for_metric = _build_combined_mask(masks_np, mode=FLAGS.combine_rule, scores=s_list, temp=FLAGS.weighted_temp)
                m_combined_for_metric = _normalize_map01(m_combined_for_metric)
                idx_cat_pass = idx_cat if idx_cat.numel()>0 else None
                idx_dog_pass = idx_dog if idx_dog.numel()>0 else None
                am = advanced_metrics(
                    img_float=img_float,
                    union_mask01=m_combined_for_metric,
                    model=model,
                    device=device,
                    label_scalar=int(labels[i].item()),
                    idx_cat=idx_cat_pass,
                    idx_dog=idx_dog_pass,
                    use_logit=FLAGS.use_logit,
                    steps=ID_STEPS,
                    baseline=ID_BASELINE,
                    blur_ksize=ID_BLUR_KSIZE,
                    enable_ins_auc=FLAGS.am_enable_ins_auc,
                    enable_del_auc=FLAGS.am_enable_del_auc,
                    enable_aopc_ins=FLAGS.am_enable_aopc_ins,
                    enable_aopc_del=FLAGS.am_enable_aopc_del,
                    return_curves=False
                )
                if used_flag:
                    if FLAGS.am_enable_ins_auc and 'auc_insertion' in am: sum_auc_ins += am['auc_insertion']
                    if FLAGS.am_enable_del_auc and 'auc_deletion' in am: sum_auc_del += am['auc_deletion']
                    if FLAGS.am_enable_aopc_ins and 'aopc_insertion' in am: sum_aopc_ins += am['aopc_insertion']
                    if FLAGS.am_enable_aopc_del and 'aopc_deletion' in am: sum_aopc_del += am['aopc_deletion']
                    am_count_used += 1

            # Lưu row per-image (CSV)
            row = {
                'file_name': file_names[i],
                'label': int(labels[i].item()),
                'orig_score': y,
                'reduced_score': o,
                'sum_component_scores': float(np.sum(s_list)),
                'consistency_error_abs': cons_err,
                'ad_component': ad_value,
                'ai_flag': ai_flag,
                'ag_component': ag_value,
                'used_for_metrics': used_flag,
                'auc_insertion': float(am['auc_insertion']) if compute_adv_master and 'auc_insertion' in am else '',
                'auc_deletion': float(am['auc_deletion']) if compute_adv_master and 'auc_deletion' in am else '',
                'aopc_insertion': float(am['aopc_insertion']) if compute_adv_master and 'aopc_insertion' in am else '',
                'aopc_deletion': float(am['aopc_deletion']) if compute_adv_master and 'aopc_deletion' in am else ''
            }
            for k_idx, sval in enumerate(s_list):
                row[f's_{k_idx+1}'] = float(sval)
            # Scalars
            try:
                if hasattr(OptCAM,'last_component_weights') and OptCAM.last_component_weights is not None:
                    c_batch = OptCAM.last_component_weights.detach().cpu().numpy()
                    for k in range(c_batch.shape[1]):
                        row[f'c_{k+1}'] = float(c_batch[i,k])
                    np.save(os.path.join(scalars_dir, f"{sanitize_name(file_names[i])}_c.npy"),
                            c_batch[i], allow_pickle=False)
            except Exception:
                pass
            per_image_rows.append(row)

            # Per-image loss histories (save all components as separate .npy files)
            if save_npy_flag and getattr(OptCAM, 'last_loss_histories', None):
                for loss_key, hist_list in OptCAM.last_loss_histories.items():
                    if len(hist_list) > 0:
                        filename = f"{sanitize_name(file_names[i])}_{loss_key.capitalize()}_Loss.npy"
                        np.save(os.path.join(plot_dir, filename),
                                np.array(hist_list, dtype=np.float32), allow_pickle=False)
        
        t_batch = time.time() - t_batch0
        total_wall_time += t_batch
        batch_times.append(float(t_batch))
        print(f"\n----[Batch {batch_idx}] sal_opt_accum={total_saliency_time:.3f}s | wall={t_batch:.3f}s raw={raw_samples} used={counted_samples}----")
        
        # Save checkpoint after every 10 batches or last batch
        if batch_idx % 10 == 0 or batch_idx == len(val_loader) - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_batch_{batch_idx}.pt')
            try:
                torch.save({
                    'batch_idx': batch_idx,
                    'raw_samples': raw_samples,
                    'counted_samples': counted_samples,
                    'sum_ad': sum_ad,
                    'sum_ai': sum_ai,
                    'sum_ag': sum_ag,
                    'per_image_rows': per_image_rows,
                    'total_saliency_time': total_saliency_time,
                    'total_original_infer_time': total_original_infer_time,
                    'timed_batches': timed_batches,
                    'total_wall_time': total_wall_time,
                    'cons_err_accum': cons_err_accum,
                    'cons_n': cons_n,
                    'sum_auc_ins': sum_auc_ins,
                    'sum_auc_del': sum_auc_del,
                    'sum_aopc_ins': sum_aopc_ins,
                    'sum_aopc_del': sum_aopc_del,
                    'am_count_used': am_count_used,
                    'batch_loss_histories_by': batch_loss_histories_by,
                    'batch_final_losses_by': batch_final_losses_by,
                    'batch_times': batch_times,
                    'flags': {
                        'max_iter': FLAGS.max_iter,
                        'learning_rate': FLAGS.learning_rate,
                        'batch_size': FLAGS.batch_size,
                        'num_masks': FLAGS.num_masks,
                        'lambda_consistency': FLAGS.lambda_consistency,
                        'use_logit': FLAGS.use_logit,
                        'name_path': FLAGS.name_path
                    }
                }, checkpoint_path)
                print(f"[Checkpoint] Saved to {checkpoint_path}")
                # Keep only last 3 checkpoints to save space
                if batch_idx >= 30:
                    old_ckpt = os.path.join(checkpoint_dir, f'checkpoint_batch_{batch_idx - 30}.pt')
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)
            except Exception as e:
                print(f"[Checkpoint] Save failed: {e}")

    # Primary aggregates (Paper Equations 13-15)
    if counted_samples>0:
        AD = sum_ad / counted_samples  # (1/N) Σ |p-o|+ / p
        AI = (sum_ai / counted_samples) * 100.0  # (1/N) Σ 1_{o>p} × 100
        AG = sum_ag / counted_samples  # (1/N) Σ |o-p|+ / (1-p)
    else:
        AD = AI = AG = 0.0

    def avg_or_na(total, count, enabled):
        if not enabled: return "N/A"
        return f"{(total / count):.6f}" if count>0 else "N/A"

    AUC_INS_str = avg_or_na(sum_auc_ins, am_count_used, FLAGS.am_enable_ins_auc)
    AUC_DEL_str = avg_or_na(sum_auc_del, am_count_used, FLAGS.am_enable_del_auc)
    AOPC_INS_str = avg_or_na(sum_aopc_ins, am_count_used, FLAGS.am_enable_aopc_ins)
    AOPC_DEL_str = avg_or_na(sum_aopc_del, am_count_used, FLAGS.am_enable_aopc_del)

    avg_saliency_per_batch = total_saliency_time / timed_batches if timed_batches>0 else 0.0
    avg_saliency_per_used_image = total_saliency_time / counted_samples if counted_samples>0 else 0.0
    avg_orig_infer_per_raw_image = total_original_infer_time / raw_samples if raw_samples>0 else 0.0
    global_runtime = time.time() - global_start_time
    avg_elapsed_time_per_batch = total_wall_time / timed_batches if timed_batches>0 else 0.0
    cons_err_mean = (cons_err_accum / cons_n) if cons_n>0 else 0.0

    # Write outputs - NEW: component-based tracking
    component_keys = ['total', 'fidelity', 'consistency', 'lambda', 'violation']
    
    # Create component subdirectories and save .npy files only
    for comp_key in component_keys:
        comp_dir = os.path.join(metrics_root, comp_key)
        ensure_dir(comp_dir)
        
        # Save batch-level histories as .npy for plotting (no individual summary files)
        if comp_key in batch_loss_histories_by and len(batch_loss_histories_by[comp_key]) > 0:
            np.save(os.path.join(comp_dir, f"batch_histories_{comp_key}.npy"),
                    np.array(batch_loss_histories_by[comp_key], dtype=object), allow_pickle=True)
            np.save(os.path.join(comp_dir, f"batch_finals_{comp_key}.npy"),
                    np.array(batch_final_losses_by[comp_key], dtype=np.float32), allow_pickle=False)
            print(f"[{comp_key}] Saved .npy data -> {comp_dir}")
    
    # Write ONE consolidated metrics_summary.txt at metrics root
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CONSOLIDATED METRICS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # === GLOBAL INFO (once) ===
    summary_lines.append("-- Run Configuration --")
    summary_lines.append(f"Samples (raw/used)               : {raw_samples}/{counted_samples}")
    summary_lines.append(f"Min original confidence          : {FLAGS.min_orig}")
    summary_lines.append(f"Only correct predictions         : {FLAGS.only_correct}")
    summary_lines.append(f"Batch size                       : {FLAGS.batch_size}")
    summary_lines.append(f"Number of masks (K)              : {FLAGS.num_masks}")
    summary_lines.append(f"Max iterations per batch         : {FLAGS.max_iter}")
    summary_lines.append(f"Lambda consistency (λ)           : {FLAGS.lambda_consistency}")
    summary_lines.append(f"Lambda scheduling enabled        : {FLAGS.use_lambda_scheduling}")
    if FLAGS.use_lambda_scheduling:
        summary_lines.append(f"Lambda start → end               : {FLAGS.lambda_start} → {FLAGS.lambda_end}")
    summary_lines.append(f"Mixed precision (FP16)           : {FLAGS.use_mixed_precision}")
    summary_lines.append("")
    
    # === PRIMARY STATISTICS (once) ===
    summary_lines.append("-- Primary Metrics (Paper Equations 13-15) --")
    summary_lines.append(f"AD (Average Drop %, Eq 13)       : {AD:.6f}")
    summary_lines.append(f"AI (Average Increase %, Eq 14)   : {AI:.6f}")
    summary_lines.append(f"AG (Average Gain %, Eq 15)       : {AG:.6f}")
    summary_lines.append("")
    
    # === ADVANCED METRICS (once) ===
    summary_lines.append("-- Advanced Metrics --")
    summary_lines.append(f"AUC Insertion                    : {AUC_INS_str}")
    summary_lines.append(f"AUC Deletion                     : {AUC_DEL_str}")
    summary_lines.append(f"AOPC Insertion                   : {AOPC_INS_str}")
    summary_lines.append(f"AOPC Deletion                    : {AOPC_DEL_str}")
    summary_lines.append(f"Advanced metrics config:")
    summary_lines.append(f"  - ID steps                     : {ID_STEPS}")
    summary_lines.append(f"  - ID baseline                  : {ID_BASELINE}")
    summary_lines.append(f"  - Blur kernel size             : {ID_BLUR_KSIZE}")
    summary_lines.append("")
    
    # === TIMING (once) ===
    summary_lines.append("-- Timing (seconds) --")
    summary_lines.append(f"Global total runtime             : {global_runtime:.3f}")
    summary_lines.append(f"Avg saliency per batch           : {avg_saliency_per_batch:.3f}")
    summary_lines.append(f"Avg saliency per used image      : {avg_saliency_per_used_image:.3f}")
    summary_lines.append(f"Avg original infer per raw img   : {avg_orig_infer_per_raw_image:.6f}")
    summary_lines.append(f"Avg elapsed time per batch       : {avg_elapsed_time_per_batch:.3f}")
    summary_lines.append("")
    
    # === CONSISTENCY CONSTRAINT (once) ===
    summary_lines.append("-- Consistency Constraint (Thesis Objective) --")
    summary_lines.append(f"Consistency error |Σc_k - c|     : {cons_err_mean:.6f}")
    summary_lines.append(f"  Per-image average              : {cons_err_mean/max(1, counted_samples):.6f}")
    summary_lines.append(f"  Accuracy (1 - error rate)           : {100.0 * (1.0 - cons_err_mean/max(1, counted_samples)):.2f}%")
    summary_lines.append("")
    
    # === COMPONENT STATISTICS (all components in one section) ===
    summary_lines.append("=" * 80)
    summary_lines.append("COMPONENT LOSS STATISTICS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    for comp_key in component_keys:
        if comp_key not in batch_final_losses_by or len(batch_final_losses_by[comp_key]) == 0:
            summary_lines.append(f"-- {comp_key.upper()} --")
            summary_lines.append("  No data available")
            summary_lines.append("")
            continue
            
        finals = batch_final_losses_by[comp_key]
        all_vals = [v for batch in batch_loss_histories_by[comp_key] for v in batch]
        
        final_val = finals[-1] if len(finals) > 0 else float('nan')
        mean_val = np.mean(all_vals) if len(all_vals) > 0 else float('nan')
        std_val = np.std(all_vals) if len(all_vals) > 0 else float('nan')
        min_val = np.min(all_vals) if len(all_vals) > 0 else float('nan')
        max_val = np.max(all_vals) if len(all_vals) > 0 else float('nan')
        
        summary_lines.append(f"-- {comp_key.upper()} --")
        summary_lines.append(f"  Final (last batch)             : {final_val:.6f}")
        summary_lines.append(f"  Mean (all steps)               : {mean_val:.6f}")
        summary_lines.append(f"  Std deviation                  : {std_val:.6f}")
        summary_lines.append(f"  Min value                      : {min_val:.6f}")
        summary_lines.append(f"  Max value                      : {max_val:.6f}")
        summary_lines.append("")
    
    # Write consolidated summary
    summary_path = os.path.join(metrics_root, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[Summary] Consolidated metrics summary written to: {summary_path}")
    
    # Write per-image CSV (only once, not per-component)
    main_metrics_dir = os.path.join(metrics_root, 'total')
    ensure_dir(main_metrics_dir)
    comp_score_fields = [f's_{k+1}' for k in range(FLAGS.num_masks)]
    scalar_fields = [f'c_{k+1}' for k in range(FLAGS.num_masks)]
    base_fields = [
        'file_name','label','orig_score','reduced_score','sum_component_scores',
        'consistency_error_abs','ad_component','ai_flag','ag_component','used_for_metrics',
        'auc_insertion','auc_deletion','aopc_insertion','aopc_deletion'
    ]
    fieldnames = base_fields + comp_score_fields + scalar_fields
    csv_path = os.path.join(main_metrics_dir, "metrics_per_image.csv")
    with open(csv_path,"w",newline='',encoding='utf-8') as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames); w.writeheader()
        for row in per_image_rows:
            for k in range(FLAGS.num_masks):
                row.setdefault(f's_{k+1}',''); row.setdefault(f'c_{k+1}','')
            w.writerow(row)

    # ==== Save batch histories (.npy) per component ====
    if save_npy_flag:
        for loss_key in ['total', 'fidelity', 'consistency', 'lambda', 'violation']:
            comp_metrics_dir = os.path.join(metrics_root, loss_key)
            ensure_dir(comp_metrics_dir)
            try:
                np.save(os.path.join(comp_metrics_dir, f"batch_loss_histories_{loss_key}.npy"),
                        np.array(batch_loss_histories_by.get(loss_key, []), dtype=object), allow_pickle=True)
                np.save(os.path.join(comp_metrics_dir, f"batch_final_losses_{loss_key}.npy"),
                        np.array(batch_final_losses_by.get(loss_key, []), dtype=np.float32))
                np.save(os.path.join(comp_metrics_dir, "batch_times.npy"),
                        np.array(batch_times, dtype=np.float32))
            except Exception as e:
                print(f"[WARN] Could not save batch histories for {loss_key}: {e}")
    
    # ==== Append one row to single table (TXT) per component ====
    for loss_key in ['total', 'fidelity', 'consistency', 'lambda', 'violation']:
        try:
            finals = batch_final_losses_by.get(loss_key, [])
            final_last = finals[-1] if finals else None
            mean_final = (sum(finals) / len(finals)) if finals else None
            histories = batch_loss_histories_by.get(loss_key, [])
            best_val = min((min(h) for h in histories), default=None) if histories else None
            first_val = histories[0][0] if (histories and histories[0]) else None

            append_row_from_flags(
                base_dir=base_dir,
                FLAGS=FLAGS,
                OptCAM=OptCAM,
                loss_suffix=loss_key,
                raw_samples=raw_samples,
                counted_samples=counted_samples,
                AD=AD, AI=AI, AG=AG,
                AUC_INS_str=AUC_INS_str,
                AUC_DEL_str=AUC_DEL_str,
                AOPC_INS_str=AOPC_INS_str,
                AOPC_DEL_str=AOPC_DEL_str,
                avg_saliency_per_batch=avg_saliency_per_batch,
                avg_elapsed_time_per_batch=avg_elapsed_time_per_batch,
                global_runtime=global_runtime,
                script_name="generate_opticam_multi.py",
                log_name="run_log",
                use_unicode=True
            )
            print(f"[Run log] appended -> {loss_key}")
        except Exception as e:
            print(f"[Run log] append failed for {loss_key}: {e}")

    # Nếu chạy 'all', thêm 1 hàng tổng hợp có đầy đủ số liệu của cả 5 components
    if running_all:
        try:
            append_summary_row_from_multi(
                base_dir=base_dir,
                FLAGS=FLAGS,
                OptCAM=OptCAM,
                raw_samples=raw_samples,
                counted_samples=counted_samples,
                AD=AD, AI=AI, AG=AG,
                AUC_INS_str=AUC_INS_str, AUC_DEL_str=AUC_DEL_str,
                AOPC_INS_str=AOPC_INS_str, AOPC_DEL_str=AOPC_DEL_str,
                avg_saliency_per_batch=avg_saliency_per_batch,
                avg_elapsed_time_per_batch=avg_elapsed_time_per_batch,
                global_runtime=global_runtime,
                batch_final_losses_by=batch_final_losses_by,
                batch_loss_histories_by=batch_loss_histories_by,
                script_name="generate_opticam_multi.py",
                log_name="run_log",
                use_unicode=True
            )
            print("[Run log] appended all_summary row")
        except Exception as e:
            print(f"[Run log] summary append failed: {e}")

    migrate_misplaced_files(base_dir, losses=['total', 'fidelity', 'consistency', 'lambda', 'violation'])
    print("[DONE] Component metrics written: total, fidelity, consistency, lambda, violation")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Speed Optimization: Optional student model for faster inference
    parser.add_argument("--student_path", type=str,
                        help="[OPTIONAL] Path to student model checkpoint for speed optimization. "
                             "If provided, student model will be used for fast inference during optimization, "
                             "while teacher (ResNet50) is used for masks and metrics. Expected 2-3x speedup.")
    parser.add_argument("--student_arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet34"],
                        help="Student model architecture (only used if --student_path is provided).")
    
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--name_path", type=str, default="OptiCamMulti_PureProb")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_masks", type=int, default=3)
    parser.add_argument("--name_norm", type=str, default="max_min", choices=["max_min","sigmoid","max"])
    parser.add_argument("--min_orig", type=float, default=0.05)
    parser.add_argument("--only_correct", action="store_true",
                        help="Only process correctly classified images (same as generate_opticam.py)")
    # NOTE: This flag ONLY affects metrics computation (AD/AG/AI), NOT optimization.
    # Optimization is hardcoded to pure probability space in MultiComponentOptCAM.
    parser.add_argument("--use_logit", action="store_true",
                        help="[METRICS ONLY] Use logits for AD/AG/AI metrics. Does NOT affect optimization (always uses probabilities). Default: False (use probabilities).")
    parser.add_argument("--eval_reduce", type=str, default="combined", choices=["combined","avg","sum"],
                        help="How to reduce component metrics: 'combined' uses aggregated mask, 'avg' averages components, 'sum' sums them.")
    parser.add_argument("--combine_rule", type=str, default="max", choices=["max","prob_or","weighted"],
                        help="Rule for combining component masks: 'max' (pixel-wise max), 'prob_or' (probabilistic), 'weighted' (score-weighted).")
    parser.add_argument("--weighted_temp", type=float, default=1.0,
                        help="Temperature for softmax weighting when combine_rule='weighted'.")
    parser.add_argument("--viz_from_combined", action="store_true",
                        help="Generate visualization from combined/aggregated mask.")
    parser.add_argument("--comp_viz_mode", type=str, default="importance", choices=["raw","importance","combined_clip"],
                        help="Component visualization mode: 'raw' (no scaling), 'importance' (scale by share), 'combined_clip' (clip by combined mask).")
    parser.add_argument("--min_comp_share", type=float, default=0.05)
    parser.add_argument("--save_loss_plot", action="store_true")
    parser.add_argument("--save_loss_npy", action="store_true", default=True)
    parser.add_argument("--no_save_loss_npy", action="store_true")
    parser.add_argument("--dbg_recon", action="store_true",
                        help="[DEBUG ONLY] Print detailed reconstruction diagnostics per-batch. Adds compute overhead. Default: False.")
    parser.add_argument("--mask_scaling", action="store_true", default=True)
    parser.add_argument("--no_mask_scaling", action="store_true")
    parser.add_argument("--am_enable_ins_auc", action="store_true", default=True)
    parser.add_argument("--am_disable_ins_auc", action="store_true")
    parser.add_argument("--am_enable_del_auc", action="store_true", default=True)
    parser.add_argument("--am_disable_del_auc", action="store_true")
    parser.add_argument("--am_enable_aopc_ins", action="store_true", default=True)
    parser.add_argument("--am_disable_aopc_ins", action="store_true")
    parser.add_argument("--am_enable_aopc_del", action="store_true", default=True)
    parser.add_argument("--am_disable_aopc_del", action="store_true")
    parser.add_argument("--no_adv_metrics", action="store_true",
                        help="Disable all advanced metrics")
    parser.add_argument("--monitoring_metric", type=str, default="mse",
                        choices=["abs","mse","all"],
                        help="Choose abs|mse or 'all' to compute all in one pass (for monitoring/logging only).")
    parser.add_argument("--lambda_consistency", type=float, default=1.0,
                        help="Lambda (λ) for consistency constraint weight. "
                             "Higher value → stronger enforcement of Σf(x⊙mask_k)=f(x). "
                             "Default=1.0. Try 0.0 (no consistency), 0.5, 1.0, 2.0, 5.0")
    parser.add_argument("--use_lambda_scheduling", action="store_true",
                        help="Enable adaptive lambda scheduling (linear from lambda_start to lambda_end).")
    parser.add_argument("--lambda_start", type=float, default=1.0,
                        help="Starting lambda value for scheduling (default: 0.1).")
    parser.add_argument("--lambda_end", type=float, default=0.3,
                        help="Ending lambda value for scheduling (default: 0.3, reduced from 0.5 to prevent loss explosion).")
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Enable mixed precision training (FP16) for memory and speed optimization.")
    
    # Initialization strategy
    parser.add_argument("--init_method", type=str, default="adaptive",
                        choices=["adaptive", "random", "constant"],
                        help="Weight initialization strategy: "
                             "'adaptive' (RECOMMENDED - baseline-compatible for K=1, symmetry-breaking for K>1), "
                             "'random' (original Gaussian noise, not baseline-compatible), "
                             "'constant' (pure constant 0.5, only safe for K=1).")
    
    # Batched forward pass
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Max batch size for batched forward optimization (auto-detect by default). "
                             "Set to 0 to disable batching. Reduce if OOM occurs (e.g., 60 for 8GB GPU, 80 for 11GB). ")
    
    parser.add_argument("--start_batch", type=int, default=0,
                        help="Starting batch index (for manual resume without checkpoint file).")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to checkpoint file to resume from (e.g., results/OptiCamMulti/checkpoints/checkpoint_batch_10.pt).")
    args = parser.parse_args()
    
    # Handle negation flags
    if args.no_save_loss_npy: args.save_loss_npy = False
    if args.no_mask_scaling: args.mask_scaling = False
    if args.am_disable_ins_auc: args.am_enable_ins_auc = False
    if args.am_disable_del_auc: args.am_enable_del_auc = False
    if args.am_disable_aopc_ins: args.am_enable_aopc_ins = False
    if args.am_disable_aopc_del: args.am_enable_aopc_del = False
    if args.no_adv_metrics:
        args.am_enable_ins_auc = False
        args.am_enable_del_auc = False
        args.am_enable_aopc_ins = False
        args.am_enable_aopc_del = False

    # absl flags registration
    flags.DEFINE_string('student_path', args.student_path, '')
    flags.DEFINE_string('student_arch', args.student_arch, '')
    flags.DEFINE_integer('max_iter', args.max_iter, '')
    flags.DEFINE_float('learning_rate', args.learning_rate, '')
    flags.DEFINE_string('name_path', args.name_path, '')
    flags.DEFINE_integer('batch_size', args.batch_size, '')
    flags.DEFINE_integer('num_masks', args.num_masks, '')
    flags.DEFINE_string('name_norm', args.name_norm, '')
    flags.DEFINE_float('min_orig', args.min_orig, '')
    flags.DEFINE_float('lambda_consistency', args.lambda_consistency, '')
    flags.DEFINE_boolean('only_correct', args.only_correct, '')
    flags.DEFINE_boolean('use_logit', args.use_logit, '')
    flags.DEFINE_string('eval_reduce', args.eval_reduce, '')
    flags.DEFINE_string('combine_rule', args.combine_rule, '')
    flags.DEFINE_float('weighted_temp', args.weighted_temp, '')
    flags.DEFINE_boolean('viz_from_combined', args.viz_from_combined, '')
    flags.DEFINE_string('comp_viz_mode', args.comp_viz_mode, '')
    flags.DEFINE_float('min_comp_share', args.min_comp_share, '')
    flags.DEFINE_boolean('save_loss_plot', args.save_loss_plot, '')
    flags.DEFINE_boolean('save_loss_npy', args.save_loss_npy, '')
    flags.DEFINE_boolean('mask_scaling', args.mask_scaling, '')
    flags.DEFINE_boolean('dbg_recon', args.dbg_recon, '')
    flags.DEFINE_boolean('am_enable_ins_auc', args.am_enable_ins_auc, '')
    flags.DEFINE_boolean('am_enable_del_auc', args.am_enable_del_auc, '')
    flags.DEFINE_boolean('am_enable_aopc_ins', args.am_enable_aopc_ins, '')
    flags.DEFINE_boolean('am_enable_aopc_del', args.am_enable_aopc_del, '')
    flags.DEFINE_boolean('no_adv_metrics', args.no_adv_metrics, '')
    flags.DEFINE_string('monitoring_metric', args.monitoring_metric, '')
    flags.DEFINE_boolean('use_lambda_scheduling', args.use_lambda_scheduling, '')
    flags.DEFINE_float('lambda_start', args.lambda_start, '')
    flags.DEFINE_float('lambda_end', args.lambda_end, '')
    flags.DEFINE_boolean('use_mixed_precision', args.use_mixed_precision, '')
    flags.DEFINE_string('init_method', args.init_method, '')
    flags.DEFINE_integer('max_batch_size', args.max_batch_size if args.max_batch_size is not None else -1, '')
    flags.DEFINE_integer('start_batch', args.start_batch, '')
    flags.DEFINE_string('resume_checkpoint', args.resume_checkpoint, '')

    app.run(main)