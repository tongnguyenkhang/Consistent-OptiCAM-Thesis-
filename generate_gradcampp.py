"""
Grad-CAM++ Implementation using pytorch_grad_cam library
=========================================================
Uses official Grad-CAM++ from pytorch_grad_cam for better performance and reliability.
Includes checkpointing, plotting, and advanced metrics.

Note: Grad-CAM++ is a single-pass method (1 forward + 1 backward), NO iterations needed.

Usage:
    python generate_gradcampp.py --name_path "GradCAMPP_Baseline" --batch_size 5
"""

import os
import time
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from absl import flags, app
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append("pytorch_grad_cam")
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from imagenet_loader import ImageNetLoader
from util import Preprocessing_Layer

from tools.compute_metrics import (
    build_cat_dog_index_sets,
    compute_metrics_per_sample,
    _normalize_map01,
    advanced_metrics,
    ID_STEPS,
    ID_BASELINE,
    ID_BLUR_KSIZE,
)

FLAGS = flags.FLAGS

# ==================== Flags ====================
flags.DEFINE_string('name_path', 'GradCAMPP', 'Name for results directory')
flags.DEFINE_integer('batch_size', 5, 'Batch size for processing (lower to avoid OOM)')
flags.DEFINE_float('min_orig', 0.05, 'Minimum original confidence threshold')
flags.DEFINE_bool('only_correct', False, 'Only process correctly predicted images')

# Advanced metrics flags
flags.DEFINE_bool('am_enable_ins_auc', True, 'Enable AUC Insertion metric')
flags.DEFINE_bool('am_enable_del_auc', True, 'Enable AUC Deletion metric')
flags.DEFINE_bool('am_enable_aopc_ins', True, 'Enable AOPC Insertion metric')
flags.DEFINE_bool('am_enable_aopc_del', True, 'Enable AOPC Deletion metric')
flags.DEFINE_bool('am_disable_ins_auc', False, 'Disable AUC Insertion')
flags.DEFINE_bool('am_disable_del_auc', False, 'Disable AUC Deletion')
flags.DEFINE_bool('am_disable_aopc_ins', False, 'Disable AOPC Insertion')
flags.DEFINE_bool('am_disable_aopc_del', False, 'Disable AOPC Deletion')
flags.DEFINE_bool('no_adv_metrics', False, 'Disable all advanced metrics')

# Checkpointing and plotting flags
flags.DEFINE_string('resume_checkpoint', None, 'Path to checkpoint file to resume from')
flags.DEFINE_integer('start_batch', 0, 'Batch index to start from (for manual resume)')
flags.DEFINE_bool('save_loss_plot', False, 'Generate plots for per-image metrics')
flags.DEFINE_bool('save_loss_npy', True, 'Save per-image metrics as .npy files')
flags.DEFINE_bool('no_save_loss_npy', False, 'Disable saving .npy files')

# ==================== Helper Functions ====================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def sanitize_name(name: str) -> str:
    base = os.path.basename(str(name))
    return os.path.splitext(base.replace('/', '_').replace('\\', '_'))[0]

# ==================== Main Processing ====================
def main(_):
    global_start_time = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Check advanced metrics flags
    enable_ins_auc = FLAGS.am_enable_ins_auc and not FLAGS.am_disable_ins_auc and not FLAGS.no_adv_metrics
    enable_del_auc = FLAGS.am_enable_del_auc and not FLAGS.am_disable_del_auc and not FLAGS.no_adv_metrics
    enable_aopc_ins = FLAGS.am_enable_aopc_ins and not FLAGS.am_disable_aopc_ins and not FLAGS.no_adv_metrics
    enable_aopc_del = FLAGS.am_enable_aopc_del and not FLAGS.am_disable_aopc_del and not FLAGS.no_adv_metrics
    
    any_adv_metric = enable_ins_auc or enable_del_auc or enable_aopc_ins or enable_aopc_del
    
    print(f"\n{'='*60}")
    print(f"Grad-CAM++ Configuration (pytorch_grad_cam library)")
    print(f"{'='*60}")
    print(f"Results directory: {FLAGS.name_path}")
    print(f"Batch size: {FLAGS.batch_size}")
    print(f"Min original confidence: {FLAGS.min_orig}")
    print(f"Only correct predictions: {FLAGS.only_correct}")
    print(f"\nAdvanced Metrics:")
    print(f"  AUC Insertion: {enable_ins_auc}")
    print(f"  AUC Deletion: {enable_del_auc}")
    print(f"  AOPC Insertion: {enable_aopc_ins}")
    print(f"  AOPC Deletion: {enable_aopc_del}")
    print(f"\nNote: Grad-CAM++ is single-pass (no iterations)")
    print(f"{'='*60}\n")
    
    # Setup directories
    base_dir = os.path.join("results", FLAGS.name_path)
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    metrics_dir = os.path.join(base_dir, "metrics")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    ensure_dir(base_dir)
    ensure_dir(images_dir)
    ensure_dir(masks_dir)
    ensure_dir(metrics_dir)
    ensure_dir(checkpoint_dir)
    
    # Load model
    print("Loading ResNet50 model...")
    try:
        from torchvision.models import ResNet50_Weights
        model_core = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError):
        model_core = models.resnet50(pretrained=True)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess_layer = Preprocessing_Layer(mean, std)
    model = nn.Sequential(preprocess_layer, model_core).to(device).eval()
    
    # Initialize Grad-CAM++ (using pytorch_grad_cam library)
    target_layers = [model[1].layer4[-1]]  # Last convolutional layer
    gradcampp = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # Data loader setup
    print("Loading ImageNet validation data...")
    valdir = './images/'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = ImageNetLoader(valdir, './revisited_imagenet_2012_val.csv', transform)
    pin_memory = use_cuda
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=pin_memory
    )
    
    # Preprocessing
    to_pil = transforms.ToPILImage()
    
    # Cat/Dog indices
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V1.meta.get('categories', None)
    except Exception:
        categories = None
    
    if categories is not None:
        cat_indices, dog_indices = build_cat_dog_index_sets(categories)
    else:
        cat_indices, dog_indices = [], []
    
    idx_cat = torch.tensor(cat_indices, dtype=torch.long, device=device) if len(cat_indices) > 0 else torch.tensor([], dtype=torch.long, device=device)
    idx_dog = torch.tensor(dog_indices, dtype=torch.long, device=device) if len(dog_indices) > 0 else torch.tensor([], dtype=torch.long, device=device)
    
    # Metrics tracking
    raw_samples = 0
    counted_samples = 0
    sum_ad = 0.0
    sum_ai = 0
    sum_ag = 0.0
    sum_auc_ins = 0.0
    sum_auc_del = 0.0
    sum_aopc_ins = 0.0
    sum_aopc_del = 0.0
    
    per_image_rows = []
    
    total_saliency_time = 0.0
    batch_times = []
    
    # Load checkpoint if resume requested
    start_batch_idx = FLAGS.start_batch
    if FLAGS.resume_checkpoint and os.path.exists(FLAGS.resume_checkpoint):
        print(f"\nLoading checkpoint from: {FLAGS.resume_checkpoint}")
        checkpoint = torch.load(FLAGS.resume_checkpoint, map_location=device)
        start_batch_idx = checkpoint.get('batch_idx', 0) + 1
        raw_samples = checkpoint.get('raw_samples', 0)
        counted_samples = checkpoint.get('counted_samples', 0)
        sum_ad = checkpoint.get('sum_ad', 0.0)
        sum_ai = checkpoint.get('sum_ai', 0)
        sum_ag = checkpoint.get('sum_ag', 0.0)
        sum_auc_ins = checkpoint.get('sum_auc_ins', 0.0)
        sum_auc_del = checkpoint.get('sum_auc_del', 0.0)
        sum_aopc_ins = checkpoint.get('sum_aopc_ins', 0.0)
        sum_aopc_del = checkpoint.get('sum_aopc_del', 0.0)
        total_saliency_time = checkpoint.get('total_saliency_time', 0.0)
        per_image_rows = checkpoint.get('per_image_rows', [])
        print(f"Resumed from batch {start_batch_idx}, {counted_samples} samples processed so far\n")
    
    print(f"\nProcessing {len(loader)} batches...")
    print(f"Note: Using pytorch_grad_cam library (optimized Grad-CAM++)")
    print(f"{'='*60}\n")
    
    for batch_idx, (images, labels, file_names) in enumerate(loader):
        # Skip batches if resuming
        if batch_idx < start_batch_idx:
            continue
            
        batch_start_time = time.time()
        
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)
        
        # Get original predictions
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
        
        # Generate Grad-CAM++ saliency maps (using library)
        saliency_start = time.time()
        targets = [ClassifierOutputTarget(int(label.item())) for label in labels]
        grayscale_cam = gradcampp(input_tensor=images, targets=targets)  # Returns (B, H, W) normalized [0,1]
        saliency_elapsed = time.time() - saliency_start
        total_saliency_time += saliency_elapsed
        
        # Process each image
        for i in range(B):
            raw_samples += 1
            
            # Get original probability
            if idx_cat.numel() > 0 and idx_dog.numel() > 0:
                p_cat = probs[i, idx_cat].sum().item()
                p_dog = probs[i, idx_dog].sum().item()
                y = p_cat if int(labels[i].item()) == 0 else p_dog
                pred_class = 0 if p_cat > p_dog else 1
            else:
                y = probs[i, labels[i]].item()
                pred_class = probs[i].argmax().item()
            
            correct = (pred_class == labels[i].item())
            
            # Filter: only_correct and min_orig
            if FLAGS.only_correct and not correct:
                continue
            if y < FLAGS.min_orig:
                continue
            
            # Get saliency map
            saliency_map = grayscale_cam[i]  # (H, W) already in [0, 1]
            img_name = sanitize_name(file_names[i])
            
            # Save mask
            mask_path = os.path.join(masks_dir, f"{img_name}_mask.npy")
            np.save(mask_path, saliency_map)
            
            # Create masked image and compute score
            with torch.no_grad():
                mask_tensor = torch.from_numpy(saliency_map).float().to(device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                masked_img = images[i:i+1] * mask_tensor
                
                masked_output = model(masked_img)
                masked_probs = torch.softmax(masked_output, dim=1)
                
                if idx_cat.numel() > 0 and idx_dog.numel() > 0:
                    p_cat_masked = masked_probs[0, idx_cat].sum().item()
                    p_dog_masked = masked_probs[0, idx_dog].sum().item()
                    o = p_cat_masked if int(labels[i].item()) == 0 else p_dog_masked
                else:
                    o = masked_probs[0, labels[i]].item()
            
            # Compute primary metrics
            ad_val, ai_val, ag_val, used_flag = compute_metrics_per_sample(
                y, o, min_orig=FLAGS.min_orig
            )
            
            # Only accumulate if sample is used (passed min_orig threshold)
            if used_flag:
                counted_samples += 1
                sum_ad += ad_val
                sum_ai += ai_val
                sum_ag += ag_val  # Accumulate AG from primary metrics (Paper Eq. 15)
            
            # Compute advanced metrics if enabled
            if any_adv_metric and used_flag:
                img_np = images[i].detach().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                adv_results = advanced_metrics(
                    img_float=img_np,
                    union_mask01=saliency_map,
                    model=model,
                    device=device,
                    label_scalar=labels[i].item(),
                    idx_cat=idx_cat if idx_cat.numel() > 0 else None,
                    idx_dog=idx_dog if idx_dog.numel() > 0 else None,
                    steps=ID_STEPS,
                    baseline=ID_BASELINE,
                    blur_ksize=ID_BLUR_KSIZE,
                    enable_ins_auc=enable_ins_auc,
                    enable_del_auc=enable_del_auc,
                    enable_aopc_ins=enable_aopc_ins,
                    enable_aopc_del=enable_aopc_del
                )
                
                ins_auc = adv_results.get('auc_insertion', 0.0)
                del_auc = adv_results.get('auc_deletion', 0.0)
                aopc_ins = adv_results.get('aopc_insertion', 0.0)
                aopc_del = adv_results.get('aopc_deletion', 0.0)
                
                # Only accumulate if sample is used
                sum_auc_ins += ins_auc
                sum_auc_del += del_auc
                sum_aopc_ins += aopc_ins
                sum_aopc_del += aopc_del
            else:
                ins_auc = del_auc = aopc_ins = aopc_del = 0.0
            
            # Save visualization
            img_uint8 = np.array(to_pil(images[i].detach().cpu()))
            img_float = img_uint8.astype(np.float32) / 255.0
            
            cam_rgb = show_cam_on_image(img_float, saliency_map, use_rgb=True)
            if cam_rgb.dtype != np.uint8:
                cam_rgb = (np.clip(cam_rgb, 0, 1) * 255).astype(np.uint8)
            cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
            
            overlay_path = os.path.join(images_dir, f"{img_name}_overlay.png")
            cv2.imwrite(overlay_path, cam_bgr)
            
            # Save per-image metrics
            row = {
                'image': file_names[i],
                'label': int(labels[i].item()),
                'pred': pred_class,
                'correct': correct,
                'y_orig': f"{y:.6f}",
                'y_masked': f"{o:.6f}",
                'AD': f"{ad_val:.4f}",
                'AI': ai_val,
                'AG': f"{ag_val:.4f}",
                'AUC_ins': f"{ins_auc:.6f}" if any_adv_metric else "N/A",
                'AUC_del': f"{del_auc:.6f}" if any_adv_metric else "N/A",
                'AOPC_ins': f"{aopc_ins:.6f}" if any_adv_metric else "N/A",
                'AOPC_del': f"{aopc_del:.6f}" if any_adv_metric else "N/A",
            }
            per_image_rows.append(row)
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{batch_idx}.pt")
            checkpoint = {
                'batch_idx': batch_idx,
                'raw_samples': raw_samples,
                'counted_samples': counted_samples,
                'sum_ad': sum_ad,
                'sum_ai': sum_ai,
                'sum_ag': sum_ag,
                'sum_auc_ins': sum_auc_ins,
                'sum_auc_del': sum_auc_del,
                'sum_aopc_ins': sum_aopc_ins,
                'sum_aopc_del': sum_aopc_del,
                'total_saliency_time': total_saliency_time,
                'per_image_rows': per_image_rows,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  [Checkpoint] Saved to {checkpoint_path}")
        
        # Progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
            avg_time = np.mean(batch_times[-5:]) if batch_times else 0
            print(f"Batch {batch_idx+1}/{len(loader)} | "
                  f"Samples: {counted_samples}/{raw_samples} | "
                  f"Time: {batch_time:.2f}s | "
                  f"Avg: {avg_time:.2f}s/batch")
    
    total_wall_time = time.time() - global_start_time
    
    # Compute final metrics
    if counted_samples > 0:
        avg_ad = sum_ad / counted_samples
        avg_ai = (sum_ai / counted_samples) * 100.0
        avg_ag = sum_ag / counted_samples  # Always from primary metrics (Paper Eq. 15)
        avg_auc_ins = sum_auc_ins / counted_samples if any_adv_metric else 0.0
        avg_auc_del = sum_auc_del / counted_samples if any_adv_metric else 0.0
        avg_aopc_ins = sum_aopc_ins / counted_samples if any_adv_metric else 0.0
        avg_aopc_del = sum_aopc_del / counted_samples if any_adv_metric else 0.0
    else:
        avg_ad = avg_ai = avg_ag = 0.0
        avg_auc_ins = avg_auc_del = avg_aopc_ins = avg_aopc_del = 0.0
    
    # Save summary
    summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Grad-CAM++ Metrics Summary (pytorch_grad_cam library)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of images processed: {raw_samples}\n")
        f.write(f"Number of images counted: {counted_samples}\n")
        f.write(f"Total wall time: {total_wall_time:.2f}s\n")
        f.write(f"Total saliency time: {total_saliency_time:.2f}s\n\n")
        f.write("Primary Metrics:\n")
        f.write(f"  Average Drop (AD):       {avg_ad:.4f}%\n")
        f.write(f"  Average Increase (AI):   {avg_ai:.4f}%\n")
        f.write(f"  Average Gain (AG):       {avg_ag:.4f}%\n\n")
        
        if any_adv_metric:
            f.write("Advanced Metrics:\n")
            f.write(f"  AUC Insertion:           {avg_auc_ins:.6f}\n")
            f.write(f"  AUC Deletion:            {avg_auc_del:.6f}\n")
            f.write(f"  AOPC Insertion:          {avg_aopc_ins:.6f}\n")
            f.write(f"  AOPC Deletion:           {avg_aopc_del:.6f}\n")
    
    # Save per-image CSV
    csv_path = os.path.join(metrics_dir, "metrics_per_image.csv")
    if per_image_rows:
        fieldnames = list(per_image_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_rows)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Grad-CAM++ Complete!")
    print(f"{'='*60}")
    print(f"Processed: {counted_samples}/{raw_samples} images")
    print(f"Average Drop (AD): {avg_ad:.4f}%")
    print(f"Average Increase (AI): {avg_ai:.4f}%")
    print(f"Average Gain (AG): {avg_ag:.4f}%")
    if any_adv_metric:
        print(f"AUC Insertion: {avg_auc_ins:.6f}")
        print(f"AUC Deletion: {avg_auc_del:.6f}")
        print(f"AOPC Insertion: {avg_aopc_ins:.6f}")
        print(f"AOPC Deletion: {avg_aopc_del:.6f}")
    print(f"Total time: {total_wall_time:.2f}s")
    print(f"Results saved to: {base_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    app.run(main)
