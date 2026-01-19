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

# Nếu compute_metrics.py ở thư mục tools thì sửa thành from tools.compute_metrics import ...
from tools.compute_metrics import build_cat_dog_index_sets, compute_metrics_per_sample, ID_STEPS, ID_BASELINE, ID_BLUR_KSIZE
# Local helpers
from util import Preprocessing_Layer, Basic_OptCAM, advanced_metrics
from imagenet_loader import ImageNetLoader
from cam_utils.eval_utils import ensure_dir, sanitize_name, show_cam_on_image

FLAGS = flags.FLAGS


def main(_):
    global_start_time = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    min_orig = FLAGS.min_orig
    only_correct = FLAGS.only_correct
    print(f"[Metrics] min_orig={min_orig}  only_correct={only_correct}")
    print(f"[AdvMetrics] toggles: insAUC={FLAGS.am_enable_ins_auc} delAUC={FLAGS.am_enable_del_auc} "
          f"AOPCins={FLAGS.am_enable_aopc_ins} AOPCdel={FLAGS.am_enable_aopc_del}  ID_STEPS={ID_STEPS} baseline={ID_BASELINE}")

    try:
        from torchvision.models import ResNet50_Weights
        model_core = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError):
        model_core = models.resnet50(pretrained=True)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    preprocess_layer = Preprocessing_Layer(mean, std)
    model = nn.Sequential(preprocess_layer, model_core).to(device).eval()

    target_layers = [model[1].layer4[-1]]

    valdir = './images/'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = ImageNetLoader(valdir, './revisited_imagenet_2012_val.csv', transform)
    pin_memory = use_cuda
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=pin_memory
    )
    to_pil = transforms.ToPILImage()

    base_dir = os.path.join('./results', FLAGS.name_path)
    images_dir = os.path.join(base_dir, 'images')
    metrics_dir = os.path.join(base_dir, 'metrics')
    plot_dir = os.path.join(base_dir, 'plot')
    weights_dir = os.path.join(base_dir, 'weights') if getattr(FLAGS, 'save_best_weights', False) else None
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    ensure_dir(base_dir); ensure_dir(images_dir); ensure_dir(metrics_dir); ensure_dir(plot_dir); ensure_dir(checkpoint_dir)
    if weights_dir is not None:
        ensure_dir(weights_dir)
    # Per-loss metrics folder (to match multi-run layout)
    loss_key = FLAGS.canonical_loss if FLAGS.canonical_loss is not None else 'abs'
    metrics_loss_dir = os.path.join(metrics_dir, loss_key)
    ensure_dir(metrics_loss_dir)

    print(f"Overlays will be saved to: {images_dir}")
    print(f"Metrics will be saved to: {metrics_dir}")

    OptCAM = Basic_OptCAM(
        model=model,
        device=device,
        target_layer=target_layers,
        max_iter=FLAGS.max_iter,
        learning_rate=FLAGS.learning_rate,
        # `name_f` was a legacy CLI label; use a fixed internal label instead
        name_f='predict',
        name_loss=FLAGS.canonical_loss,
        name_norm=FLAGS.name_norm,
            name_mode='resnet',
            # match paper/default: objective controlled by CLI flag
            objective=FLAGS.objective,
            use_prob=(not FLAGS.use_logit),
            min_orig=FLAGS.min_orig,
            delta_change_threshold=getattr(FLAGS, 'delta_change_threshold', 0.0),
            save_best_weights=getattr(FLAGS, 'save_best_weights', False),
            save_best_weights_dir=weights_dir
    )
    # Apply canonical loss choice to the optimizer (abs|rel|mse|'all')
    OptCAM.canonical_loss = FLAGS.canonical_loss

    # Plotting (optional)
    plotting_enabled = FLAGS.save_loss_plot
    if plotting_enabled:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib import failed: {e}. Disabling plotting.")
            plotting_enabled = False
    save_npy_flag = bool(FLAGS.save_loss_npy) and (not FLAGS.no_save_loss_npy)

    # Primary metrics accumulators (Paper Equations 13-15)
    raw_samples = 0
    counted_samples = 0  # N in paper formulas
    sum_ad = 0.0  # Σ |p-o|+ / p (ALL samples)
    sum_ai = 0  # Σ 1_{o>p} (count increases)
    sum_ag = 0.0  # Σ |o-p|+ / (1-p) (ALL samples)
    per_image_rows = []

    # Timing accumulators
    total_saliency_time = 0.0
    total_mask_infer_time = 0.0
    total_original_infer_time = 0.0
    timed_batches = 0

    # Advanced metrics accumulators
    sum_auc_ins = 0.0
    sum_auc_del = 0.0
    sum_aopc_ins = 0.0
    sum_aopc_del = 0.0
    am_count_used = 0

    # Loss histories (aggregated) - only internal loss
    batch_internal_final_losses = []
    batch_internal_loss_histories = []
    batch_times = []

    # Build species index sets (cat/dog)
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

    # Attach to OptCAM for species probability aggregation
    if idx_cat.numel() > 0 and idx_dog.numel() > 0:
        OptCAM.idx_cat = idx_cat
        OptCAM.idx_dog = idx_dog

    # Helper: kiểm tra có cần compute advanced metrics không
    compute_adv = any([
        FLAGS.am_enable_ins_auc,
        FLAGS.am_enable_del_auc,
        FLAGS.am_enable_aopc_ins,
        FLAGS.am_enable_aopc_del
    ])
    if not compute_adv:
        print("[AdvMetrics] All advanced metric toggles disabled -> skipping advanced metrics computation.")
    
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
            total_mask_infer_time = ckpt.get('total_mask_infer_time', 0.0)
            total_original_infer_time = ckpt.get('total_original_infer_time', 0.0)
            timed_batches = ckpt.get('timed_batches', 0)
            sum_auc_ins = ckpt.get('sum_auc_ins', 0.0)
            sum_auc_del = ckpt.get('sum_auc_del', 0.0)
            sum_aopc_ins = ckpt.get('sum_aopc_ins', 0.0)
            sum_aopc_del = ckpt.get('sum_aopc_del', 0.0)
            am_count_used = ckpt.get('am_count_used', 0)
            batch_internal_final_losses = ckpt.get('batch_internal_final_losses', [])
            batch_internal_loss_histories = ckpt.get('batch_internal_loss_histories', [])
            batch_times = ckpt.get('batch_times', [])
            print(f"[Checkpoint] Resuming from batch {start_batch_idx} (processed {counted_samples} samples)")
        except Exception as e:
            print(f"[Checkpoint] Failed to load: {e}. Starting from batch {start_batch_idx}")

    for batch_idx, (images, labels, file_names) in enumerate(val_loader):
        # Skip batches already processed
        if batch_idx < start_batch_idx:
            continue
        
        batch_start_wall = time.time()
        images = images.to(device)
        labels = labels.to(device)

        # Original inference
        t0 = time.time()
        with torch.no_grad():
            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)
            # For prediction/thresholding we may use aggregated species probs or logits
            if idx_cat.numel() > 0 and idx_dog.numel() > 0:
                # aggregate per-species both for logits and probs
                p_cat_logits = logits_orig[:, idx_cat].sum(dim=1)
                p_dog_logits = logits_orig[:, idx_dog].sum(dim=1)
                p_cat_probs = probs_orig[:, idx_cat].sum(dim=1)
                p_dog_probs = probs_orig[:, idx_dog].sum(dim=1)
                # Use probabilities to decide predicted species (more stable)
                preds = torch.where(p_cat_probs >= p_dog_probs,
                                    torch.zeros_like(p_cat_probs, dtype=torch.long),
                                    torch.ones_like(p_dog_probs, dtype=torch.long))
                labels = labels.long()
                # Keep both representations: orig_scores (for logging/legacy) and
                # orig_scores_metric (for metric computations which expect probs)
                if FLAGS.use_logit:
                    orig_scores = torch.where(labels == 0, p_cat_logits, p_dog_logits)
                else:
                    orig_scores = torch.where(labels == 0, p_cat_probs, p_dog_probs)
                orig_scores_metric = torch.where(labels == 0, p_cat_probs, p_dog_probs)
            else:
                # No species aggregation: pick per-sample
                preds = torch.argmax(probs_orig, dim=1)
                labels = labels.long()
                if FLAGS.use_logit:
                    orig_scores = logits_orig[range(images.size(0)), labels]
                else:
                    orig_scores = probs_orig[range(images.size(0)), labels]
                # For metrics we always use probabilities (compute_metrics expects probs)
                orig_scores_metric = probs_orig[range(images.size(0)), labels]
        original_infer_time = time.time() - t0
        total_original_infer_time += original_infer_time
        raw_samples += images.size(0)

        if only_correct:
            correct_mask = (preds == labels)
            if correct_mask.sum() == 0:
                print(f"[Batch {batch_idx}] skipped (no correct predictions)")
                continue
            images = images[correct_mask]
            labels = labels[correct_mask]
            orig_scores = orig_scores[correct_mask]
            # also filter metric scores (probabilities) if present
            try:
                orig_scores_metric = orig_scores_metric[correct_mask]
            except Exception:
                pass
            file_names = [fn for i, fn in enumerate(file_names) if correct_mask[i]]

        if len(file_names) == 0:
            continue

        # Optimize saliency
        sal_start = time.time()
        saliency_map, masked_images = OptCAM(images, labels)  # saliency_map: (B,1,H,W)
        sal_time = time.time() - sal_start
        total_saliency_time += sal_time

        # Record internal loss history
        if hasattr(OptCAM, 'last_internal_loss_history') and OptCAM.last_internal_loss_history is not None:
            hist_int = list(OptCAM.last_internal_loss_history)
            if len(hist_int) > 0:
                batch_internal_final_losses.append(hist_int[-1])
                batch_internal_loss_histories.append(hist_int)
                batch_times.append(sal_time)

        # Inference on masked images
        mask_infer_start = time.time()
        with torch.no_grad():
            logits_masked = model(masked_images.to(device))
            probs_masked = torch.softmax(logits_masked, dim=1)
            if idx_cat.numel() > 0 and idx_dog.numel() > 0:
                p_cat_logits_masked = logits_masked[:, idx_cat].sum(dim=1)
                p_dog_logits_masked = logits_masked[:, idx_dog].sum(dim=1)
                p_cat_probs_masked = probs_masked[:, idx_cat].sum(dim=1)
                p_dog_probs_masked = probs_masked[:, idx_dog].sum(dim=1)
                if FLAGS.use_logit:
                    masked_scores = torch.where(labels == 0, p_cat_logits_masked, p_dog_logits_masked)
                else:
                    masked_scores = torch.where(labels == 0, p_cat_probs_masked, p_dog_probs_masked)
                masked_scores_metric = torch.where(labels == 0, p_cat_probs_masked, p_dog_probs_masked)
            else:
                if FLAGS.use_logit:
                    masked_scores = logits_masked[range(images.size(0)), labels]
                else:
                    masked_scores = probs_masked[range(images.size(0)), labels]
                # Metrics always use probs
                masked_scores_metric = probs_masked[range(images.size(0)), labels]
        mask_infer_time = time.time() - mask_infer_start
        total_mask_infer_time += mask_infer_time

        saliency_np = saliency_map.detach().cpu().numpy()  # (B,1,H,W)
        B_used = images.size(0)
        timed_batches += 1

        for i in range(B_used):
            # For metrics we use probability-space scores (orig_scores_metric / masked_scores_metric)
            try:
                y_metric = float(orig_scores_metric[i].item())
            except Exception:
                # fallback if metric tensor missing
                y_metric = float(orig_scores[i].item())
            try:
                o_metric = float(masked_scores_metric[i].item())
            except Exception:
                o_metric = float(masked_scores[i].item())
            ad_value, ai_flag, ag_value, used_flag = compute_metrics_per_sample(
                y_metric, o_metric, min_orig=min_orig
            )

            if used_flag:
                counted_samples += 1
                sum_ad += ad_value  # Accumulate AD for ALL samples
                sum_ai += ai_flag   # Count increases
                sum_ag += ag_value  # Accumulate AG for ALL samples

            # Advanced metrics (nên dùng saliency mask đã chuẩn hoá: saliency_np[i,0])
            am = {}
            if compute_adv:
                union_mask01 = saliency_np[i, 0]  # (H,W)
                am = advanced_metrics(
                    img_float=(np.array(to_pil(images[i].cpu())).astype(np.float32) / 255.0),
                    union_mask01=union_mask01,
                    model=model,
                    device=device,
                    label_scalar=int(labels[i].item()),
                    idx_cat=idx_cat if idx_cat.numel() > 0 else None,
                    idx_dog=idx_dog if idx_dog.numel() > 0 else None,
                    use_logit=False,  # giữ tương thích với xác suất
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
                    if FLAGS.am_enable_ins_auc and 'auc_insertion' in am:
                        sum_auc_ins += float(am['auc_insertion'])
                    if FLAGS.am_enable_del_auc and 'auc_deletion' in am:
                        sum_auc_del += float(am['auc_deletion'])
                    if FLAGS.am_enable_aopc_ins and 'aopc_insertion' in am:
                        sum_aopc_ins += float(am['aopc_insertion'])
                    if FLAGS.am_enable_aopc_del and 'aopc_deletion' in am:
                        sum_aopc_del += float(am['aopc_deletion'])
                    am_count_used += 1

            row = {
                'file_name': file_names[i],
                'label': int(labels[i].item()),
                'orig_score': y_metric,
                'masked_score': o_metric,
                'ad_component': ad_value,
                'ai_flag': ai_flag,
                'ag_component': ag_value,
                'used_for_metrics': used_flag,
                'auc_insertion': float(am['auc_insertion']) if (FLAGS.am_enable_ins_auc and 'auc_insertion' in am) else '',
                'auc_deletion': float(am['auc_deletion']) if (FLAGS.am_enable_del_auc and 'auc_deletion' in am) else '',
                'aopc_insertion': float(am['aopc_insertion']) if (FLAGS.am_enable_aopc_ins and 'aopc_insertion' in am) else '',
                'aopc_deletion': float(am['aopc_deletion']) if (FLAGS.am_enable_aopc_del and 'aopc_deletion' in am) else ''
            }
            per_image_rows.append(row)

            # Save per-image loss history (internal loss only)
            if hasattr(OptCAM, 'last_internal_loss_history') and OptCAM.last_internal_loss_history is not None and len(OptCAM.last_internal_loss_history) > 0:
                hist_int = np.array(list(OptCAM.last_internal_loss_history), dtype=np.float32)
                fname = sanitize_name(file_names[i])
                out_npy = os.path.join(plot_dir, f"{fname}_Loss_{OptCAM.canonical_loss}.npy")
                try:
                    if os.path.exists(out_npy):
                        os.remove(out_npy)
                except Exception:
                    pass
                if save_npy_flag:
                    np.save(out_npy, hist_int)

            # Overlay
            sal_map = np.transpose(saliency_np[i], (1, 2, 0))  # (H,W,1)
            img_uint8 = np.array(to_pil(images[i].cpu()))
            img_float = img_uint8 / 255.0
            overlay = show_cam_on_image(img_float, sal_map)
            out_name = f"{sanitize_name(file_names[i])}_{labels[i].item()}_Smap.png"
            cv2.imwrite(os.path.join(images_dir, out_name), np.uint8(overlay))

        batch_wall_time = time.time() - batch_start_wall
        batch_times.append(batch_wall_time)
        print(
            f"[Batch {batch_idx}] orig={original_infer_time:.3f}s sal={sal_time:.3f}s "
            f"masked={mask_infer_time:.3f}s wall={batch_wall_time:.3f}s raw={raw_samples} used={counted_samples}"
        )
        
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
                    'total_mask_infer_time': total_mask_infer_time,
                    'total_original_infer_time': total_original_infer_time,
                    'timed_batches': timed_batches,
                    'sum_auc_ins': sum_auc_ins,
                    'sum_auc_del': sum_auc_del,
                    'sum_aopc_ins': sum_aopc_ins,
                    'sum_aopc_del': sum_aopc_del,
                    'am_count_used': am_count_used,
                    'batch_internal_final_losses': batch_internal_final_losses,
                    'batch_internal_loss_histories': batch_internal_loss_histories,
                    'batch_times': batch_times,
                    'flags': {
                        'max_iter': FLAGS.max_iter,
                        'learning_rate': FLAGS.learning_rate,
                        'objective': FLAGS.objective,
                        'batch_size': FLAGS.batch_size,
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
                print(f"[Checkpoint] Failed to save: {e}")

    # Primary summary (Paper Equations 13-15)
    if counted_samples > 0:
        AD = sum_ad / counted_samples  # (1/N) Σ |p-o|+ / p
        AI = (sum_ai / counted_samples) * 100.0  # (1/N) Σ 1_{o>p} × 100
        AG = sum_ag / counted_samples  # (1/N) Σ |o-p|+ / (1-p)
    else:
        AD = AI = AG = 0.0

    # Advanced summary
    def avg_or_na(total, count, enabled):
        if not enabled:
            return "N/A"
        if count > 0:
            return f"{(total / count):.6f}"
        return "N/A"

    AUC_INS_str = avg_or_na(sum_auc_ins, am_count_used, FLAGS.am_enable_ins_auc)
    AUC_DEL_str = avg_or_na(sum_auc_del, am_count_used, FLAGS.am_enable_del_auc)
    AOPC_INS_str = avg_or_na(sum_aopc_ins, am_count_used, FLAGS.am_enable_aopc_ins)
    AOPC_DEL_str = avg_or_na(sum_aopc_del, am_count_used, FLAGS.am_enable_aopc_del)

    avg_saliency_per_batch = total_saliency_time / timed_batches if timed_batches > 0 else 0.0
    avg_saliency_per_used_image = total_saliency_time / counted_samples if counted_samples > 0 else 0.0
    avg_mask_infer_per_used_image = total_mask_infer_time / counted_samples if counted_samples > 0 else 0.0
    avg_orig_infer_per_raw_image = total_original_infer_time / raw_samples if raw_samples > 0 else 0.0
    global_runtime = time.time() - global_start_time

    summary_path = os.path.join(metrics_loss_dir, f"metrics_summary_{OptCAM.canonical_loss}.txt")
    with open(summary_path, "w") as f:
        f.write("===== Metrics Summary =====\n")
        f.write(f"Samples raw/used: {raw_samples}/{counted_samples}\n")
        f.write(f"Settings: min_orig={min_orig}  only_correct={only_correct}\n")
        f.write("\n-- Primary statistics --\n")
        f.write(f"AD (Average Drop %)      : {AD:.6f}\n")
        f.write(f"AI (Average Increase %)  : {AI:.6f}\n")
        f.write(f"AG (Average Gain %)      : {AG:.6f}\n")
        f.write("\n-- Advanced metrics --\n")
        f.write(f"AUC Insertion                   : {AUC_INS_str}\n")
        f.write(f"AUC Deletion                    : {AUC_DEL_str}\n")
        f.write(f"AOPC Insertion                  : {AOPC_INS_str}\n")
        f.write(f"AOPC Deletion                   : {AOPC_DEL_str}\n")
        f.write("\n-- Timing (seconds) --\n")
        f.write(f"Original inference total       : {total_original_infer_time:.6f}\n")
        f.write(f"Saliency optimization total    : {total_saliency_time:.6f}\n")
        f.write(f"Masked inference total         : {total_mask_infer_time:.6f}\n")
        f.write(f"Avg saliency per batch         : {avg_saliency_per_batch:.6f}\n")
        f.write(f"Avg saliency per used image    : {avg_saliency_per_used_image:.6f}\n")
        f.write(f"Avg masked infer per used image: {avg_mask_infer_per_used_image:.6f}\n")
        f.write(f"Avg original infer per raw img : {avg_orig_infer_per_raw_image:.6f}\n")
        f.write(f"Global total runtime           : {global_runtime:.6f}\n")
    print(f"Saved metrics summary to: {summary_path}")

    csv_path = os.path.join(metrics_loss_dir, f"metrics_per_image_{OptCAM.canonical_loss}.csv")
    with open(csv_path, "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            'file_name', 'label', 'orig_score', 'masked_score',
            'ad_component', 'ai_flag', 'ag_component', 'used_for_metrics',
            'auc_insertion', 'auc_deletion', 'aopc_insertion', 'aopc_deletion'
        ])
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

    # Optional plotting of aggregated internal losses
    if plotting_enabled and len(batch_internal_loss_histories) > 0:
        try:
            import matplotlib.pyplot as plt
            # Plot internal final losses per batch
            if len(batch_internal_final_losses) > 0:
                finals = np.array(batch_internal_final_losses, dtype=np.float32)
                out_name = os.path.join(metrics_loss_dir, f'overall_loss_batches_{OptCAM.canonical_loss}.png')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(np.arange(1, len(finals) + 1), finals, s=10, alpha=0.6,
                           color='tab:purple', label='Final loss per batch')
                window = max(1, int(len(finals) / 50))
                if window > 1 and len(finals) >= window:
                    kernel = np.ones(window) / float(window)
                    ma = np.convolve(finals, kernel, mode='valid')
                    ma_x = np.arange(1 + window // 2, 1 + window // 2 + len(ma))
                    ax.plot(ma_x, ma, color='tab:red', linewidth=2, label=f'{window}-pt MA')
                ax.set_xlabel('Batch index')
                ax.set_ylabel(f'Loss ({OptCAM.canonical_loss})')
                ax.set_title(f'Final loss per batch [{OptCAM.canonical_loss}]')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                try:
                    if os.path.exists(out_name):
                        os.remove(out_name)
                except Exception:
                    pass
                plt.savefig(out_name, dpi=100)
                plt.close()
            # Plot internal loss histories per iteration
            if len(batch_internal_loss_histories) > 0:
                max_len = max(len(h) for h in batch_internal_loss_histories)
                arr = np.full((len(batch_internal_loss_histories), max_len), np.nan, dtype=np.float32)
                for i, h in enumerate(batch_internal_loss_histories):
                    arr[i, :len(h)] = np.array(h, dtype=np.float32)
                mean_loss = np.nanmean(arr, axis=0)
                std_loss = np.nanstd(arr, axis=0)
                counts = np.sum(~np.isnan(arr), axis=0)
                iters = np.arange(1, max_len + 1)
                out_name2 = os.path.join(metrics_loss_dir, f'overall_loss_iterations_{OptCAM.canonical_loss}.png')
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(iters, mean_loss, marker='o', color='tab:purple', label='Mean loss')
                ax1.fill_between(iters, mean_loss - std_loss, mean_loss + std_loss,
                                 color='tab:purple', alpha=0.2, label='±1 std')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel(f'Mean loss ({OptCAM.canonical_loss})')
                ax1.grid(True)
                ax2 = ax1.twinx()
                ax2.bar(iters, counts, alpha=0.15, color='grey', label='Num batches', width=0.8)
                ax2.set_ylabel('Number of batches contributing')
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, loc='upper right')
                plt.title(f'Mean loss per iteration [{OptCAM.canonical_loss}]')
                plt.tight_layout()
                try:
                    if os.path.exists(out_name2):
                        os.remove(out_name2)
                except Exception:
                    pass
                plt.savefig(out_name2, dpi=100)
                plt.close()
        except Exception:
            pass

    if save_npy_flag and len(batch_internal_loss_histories) > 0:
        try:
                np.save(os.path.join(metrics_loss_dir, f'batch_internal_loss_histories_{OptCAM.canonical_loss}.npy'),
                        np.array(batch_internal_loss_histories, dtype=object), allow_pickle=True)
                np.save(os.path.join(metrics_loss_dir, f'batch_internal_final_losses_{OptCAM.canonical_loss}.npy'),
                        np.array(batch_internal_final_losses, dtype=np.float32))
                np.save(os.path.join(metrics_loss_dir, f'batch_times_{OptCAM.canonical_loss}.npy'),
                        np.array(batch_times, dtype=np.float32))
        except Exception:
            pass

    print(f"Saved overlay images to: {images_dir}")
    print(f"Saved metrics to: {metrics_dir}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--target_layer", default='42')
    # legacy name_f removed; use --use_logit to control logits vs probabilities
    parser.add_argument("--name_norm", default='max_min')
    parser.add_argument("--canonical_loss", type=str, default="mse", choices=["abs","mse","all"],
                        help="Choose abs|mse or 'all' to compute all in one pass. Default 'mse' for smooth gradients (recommended).")
    parser.add_argument("--name_path", default='OptiCam')
    parser.add_argument("--batch_size", type=int, default=16)
    
    # Control logits vs probabilities: Default is probabilities (use_logit=False)
    # Use --use_logit to enable logits for optimization
    parser.add_argument("--use_logit", action="store_true",
                        help="Use logits for optimization (unbounded). Default: False (use probabilities).")
    # Objective selection: 'mask' (maximize class score) or 'diff' (preserve score)
    parser.add_argument("--objective", type=str, choices=["mask", "diff"], default="diff",
                        help="Objective to optimize per-image: 'mask' (maximize class score) or 'diff' (preserve score).")

    parser.add_argument("--min_orig", type=float, default=0.05)
    parser.add_argument("--only_correct", action="store_true",
                        help="Only process samples with correct predictions (pred == label). Recommended for XAI evaluation.")

    parser.add_argument("--save_loss_plot", action="store_true")
    parser.add_argument("--save_loss_npy", action="store_true", default=True)
    parser.add_argument("--no_save_loss_npy", action="store_true")

    # Advanced metrics toggles (giống file multi)
    parser.add_argument("--am_enable_ins_auc", action="store_true", default=True)
    parser.add_argument("--am_disable_ins_auc", action="store_true")
    parser.add_argument("--am_enable_del_auc", action="store_true", default=True)
    parser.add_argument("--am_disable_del_auc", action="store_true")
    parser.add_argument("--am_enable_aopc_ins", action="store_true", default=True)
    parser.add_argument("--am_disable_aopc_ins", action="store_true")
    parser.add_argument("--am_enable_aopc_del", action="store_true", default=True)
    parser.add_argument("--am_disable_aopc_del", action="store_true")

    # Early stopping & best weights
    parser.add_argument("--delta_change_threshold", type=float, default=0.0,
                        help="Early stop when |loss_t - loss_{t-1}| < threshold (0 disables).")
    parser.add_argument("--save_best_weights", action="store_true",
                        help="Save best optimization weights (alpha) per batch.")
    
    # Checkpoint resume
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume from.")
    parser.add_argument("--start_batch", type=int, default=0,
                        help="Batch index to start from (for manual resume).")

    args = parser.parse_args()

    # Resolve disable toggles
    if args.am_disable_ins_auc:
        args.am_enable_ins_auc = False
    if args.am_disable_del_auc:
        args.am_enable_del_auc = False
    if args.am_disable_aopc_ins:
        args.am_enable_aopc_ins = False
    if args.am_disable_aopc_del:
        args.am_enable_aopc_del = False
    if args.no_save_loss_npy:
        args.save_loss_npy = False

    flags.DEFINE_integer('max_iter', args.max_iter, '')
    flags.DEFINE_float('learning_rate', args.learning_rate, '')
    flags.DEFINE_string('target_layer', args.target_layer, '')
    # legacy name_f flag removed; use --use_logit to select logits vs probs
    # legacy name_loss flag removed; use 'canonical_loss' instead
    flags.DEFINE_string('name_norm', args.name_norm, '')
    flags.DEFINE_string('name_path', args.name_path, '')
    flags.DEFINE_string('canonical_loss', args.canonical_loss, '')
    flags.DEFINE_integer('batch_size', args.batch_size, '')

    flags.DEFINE_float('min_orig', args.min_orig, '')
    flags.DEFINE_boolean('only_correct', args.only_correct, '')
    flags.DEFINE_boolean('save_loss_plot', args.save_loss_plot, '')
    flags.DEFINE_boolean('save_loss_npy', args.save_loss_npy, '')
    flags.DEFINE_boolean('no_save_loss_npy', args.no_save_loss_npy, '')
    flags.DEFINE_boolean('use_logit', args.use_logit, '')
    flags.DEFINE_string('objective', args.objective, '')

    # Advanced metrics flags
    flags.DEFINE_boolean('am_enable_ins_auc', args.am_enable_ins_auc, '')
    flags.DEFINE_string('resume_checkpoint', args.resume_checkpoint if args.resume_checkpoint else '', '')
    flags.DEFINE_integer('start_batch', args.start_batch, '')
    flags.DEFINE_boolean('am_enable_del_auc', args.am_enable_del_auc, '')
    flags.DEFINE_boolean('am_enable_aopc_ins', args.am_enable_aopc_ins, '')
    flags.DEFINE_boolean('am_enable_aopc_del', args.am_enable_aopc_del, '')

    # Early stopping flags
    flags.DEFINE_float('delta_change_threshold', args.delta_change_threshold, '')
    flags.DEFINE_boolean('save_best_weights', args.save_best_weights, '')

    app.run(main)