import os
import time
import csv
import numpy as np
import cv2
import torch

# Đảm bảo có thể import pytorch_grad_cam khi chạy từ repo root
import sys
if "pytorch_grad_cam" not in sys.path:
    sys.path.append("pytorch_grad_cam")

from pytorch_grad_cam.utils.image import show_cam_on_image
from util import Basic_OptCAM

# ---------- io_utils ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_name(name: str) -> str:
    base = os.path.basename(str(name))
    return os.path.splitext(base.replace('/', '_').replace('\\', '_'))[0]


# ---------- metrics ----------

def compute_metrics_per_sample(y,
                               o,
                               min_orig=0.05,
                               mode="abs",
                               eps=1e-12):
    """
    Trả về: drop_percent, increase_flag, gain_value, used_flag
    """
    y = max(float(y), eps)
    if y < min_orig:
        return 0.0, 0, 0.0, 0
    if o <= y:
        drop = (y - o) / y * 100.0
        return drop, 0, 0.0, 1
    if mode == "rel":
        gain = (o - y) / y * 100.0
    else:
        gain = (o - y) * 100.0
    return 0.0, 1, gain, 1

def _split_into_patches(h, w, ph, pw):
    patches = []
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            ys, ye = y, min(y + ph, h)
            xs, xe = x, min(x + pw, w)
            patches.append((ys, ye, xs, xe))
    return patches

def _compute_patch_importance(saliency_map, patches):
    # saliency_map: (1,H,W) hoặc (H,W)
    if saliency_map.dim() == 3:
        sal = saliency_map[0]
    else:
        sal = saliency_map
    scores = []
    for i, (ys, ye, xs, xe) in enumerate(patches):
        patch_val = sal[ys:ye, xs:xe].mean().item()
        scores.append((i, patch_val))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _prepare_replacement(image, method):
    c, h, w = image.shape
    out = image.clone()
    if method == 'zero':
        out.zero_()
    elif method == 'mean':
        mean_vals = image.view(c, -1).mean(dim=1)
        out[:] = mean_vals.view(c, 1, 1)
    else:
        raise ValueError(f"Unsupported id_replacement: {method}")
    return out

@torch.no_grad()
def _score_model(model, img_batch, labels, use_logit: bool):
    logits = model(img_batch)
    if use_logit:
        return logits[range(len(labels)), labels]
    probs = torch.softmax(logits, dim=1)
    return probs[range(len(labels)), labels]

def insertion_deletion_auc(model,
                           image,
                           saliency_map,
                           label,
                           device,
                           patch_h=16,
                           patch_w=16,
                           steps=20,
                           replacement='mean',
                           use_logit=False):
    image = image.to(device).clone()
    c, h, w = image.shape
    sal = saliency_map[0].to(device) if saliency_map.dim() == 3 else saliency_map.to(device)
    patches = _split_into_patches(h, w, patch_h, patch_w)
    ordered = _compute_patch_importance(sal, patches)
    N = len(patches)
    steps = max(1, min(steps, N))
    group = max(1, N // steps)

    ins_scores, del_scores = [], []
    ins_img = _prepare_replacement(image, replacement)
    del_img = image.clone()
    base = _prepare_replacement(image, replacement)

    s0_ins = _score_model(model, ins_img.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
    s0_del = _score_model(model, del_img.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
    ins_scores.append(s0_ins)
    del_scores.append(s0_del)

    applied = 0
    for _ in range(steps):
        start, end = applied, min(N, applied + group)
        for t in range(start, end):
            idx = ordered[t][0]
            ys, ye, xs, xe = patches[idx]
            ins_img[:, ys:ye, xs:xe] = image[:, ys:ye, xs:xe]
            del_img[:, ys:ye, xs:xe] = base[:, ys:ye, xs:xe]
        applied = end
        s_ins = _score_model(model, ins_img.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
        s_del = _score_model(model, del_img.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
        ins_scores.append(s_ins)
        del_scores.append(s_del)
        if applied >= N:
            break

    x = np.linspace(0, 1, len(ins_scores))
    try:
        integ = np.trapezoid  # numpy mới
    except AttributeError:
        integ = np.trapz
    ins_auc = float(integ(ins_scores, x))
    del_auc = float(integ(del_scores, x))
    return ins_auc, del_auc, ins_scores, del_scores

def compute_morf_single(model,
                        image,
                        saliency_map,
                        label,
                        device,
                        patch_h=16,
                        patch_w=16,
                        replacement='mean',
                        use_logit=False,
                        max_fraction=1.0):
    image = image.clone().to(device)
    c, h, w = image.shape
    sal_map = saliency_map.to(device)
    if sal_map.dim() == 4:
        sal_map = sal_map[0]
    patches = _split_into_patches(h, w, patch_h, patch_w)
    ordered = _compute_patch_importance(sal_map, patches)
    max_patches = int(len(patches) * float(max_fraction) + 1e-8)
    max_patches = max(1, min(len(patches), max_patches))

    s0 = _score_model(model, image.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
    morf_curve = [s0]
    base = _prepare_replacement(image, replacement)
    work = image.clone()
    for k in range(1, max_patches + 1):
        idx = ordered[k-1][0]
        ys, ye, xs, xe = patches[idx]
        work[:, ys:ye, xs:xe] = base[:, ys:ye, xs:xe]
        s = _score_model(model, work.unsqueeze(0), torch.tensor([label], device=device), use_logit)[0].item()
        morf_curve.append(s)

    s0_val = morf_curve[0]
    diffs = [(s0_val - morf_curve[k]) for k in range(1, len(morf_curve))]
    aopc = 0.0 if len(diffs) == 0 else float(np.mean(diffs))
    final_score = morf_curve[-1]
    return morf_curve, aopc, final_score, s0_val


# ---------- evaluator ----------

def run_opticam(model_seq, device, images, labels, target_layers, optcam_cfg):
    """
    Thực thi Opti-CAM cho một batch.
    Trả về: saliency_maps (B,1,H,W), masked_images (B,3,H,W), last_feature (B,C,Hf,Wf)
    """
    optcam = Basic_OptCAM(
        model=model_seq,
        device=device,
        target_layer=target_layers,
        max_iter=optcam_cfg['max_iter'],
        learning_rate=optcam_cfg['learning_rate'],
        name_f=optcam_cfg['name_f'],
        name_loss=optcam_cfg['name_loss'],
        name_norm=optcam_cfg['name_norm'],
        name_mode='resnet'
    )
    saliency_maps, masked_images = optcam(images, labels)
    last_feature = optcam.last_feature
    return saliency_maps, masked_images, last_feature

def eval_and_dump(model_seq,
                  tag,
                  device,
                  val_loader,
                  base_dirs,
                  metric_cfg,
                  optcam_cfg):
    """
    Thực thi Opti-CAM + xuất artifacts + metrics cho một model.
    """
    images_dir = base_dirs['images_dir']
    sal_dir = base_dirs['sal_dir']
    feat_dir = base_dirs['feat_dir']
    metrics_dir = base_dirs['metrics_dir']
    for d in [images_dir, sal_dir, feat_dir, metrics_dir]:
        ensure_dir(d)

    target_layers = [model_seq[1].layer4[-1]]

    metric_mode = metric_cfg['metric_mode']
    min_orig = metric_cfg['min_orig']
    only_correct = metric_cfg['only_correct']

    raw_samples = 0
    counted_samples = 0
    sum_drop = 0.0
    increase_count = 0
    sum_gain = 0.0
    per_image_rows = []

    ins_aucs, del_aucs, morf_aopcs = [], [], []

    total_saliency_time = 0.0
    total_mask_infer_time = 0.0
    total_original_infer_time = 0.0
    timed_batches = 0
    total_saved = 0

    for batch_idx, (images, labels, file_names) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Original inference
        t0 = time.time()
        with torch.no_grad():
            logits_orig = model_seq(images)
            probs_orig = torch.softmax(logits_orig, dim=1)
            preds = probs_orig.argmax(dim=1)
            orig_scores = probs_orig[range(images.size(0)), labels]
        total_original_infer_time += (time.time() - t0)

        raw_samples += images.size(0)

        if only_correct:
            mask = (preds == labels)
            if mask.sum() == 0:
                print(f"[{tag}][Batch {batch_idx}] skipped (no correct predictions)")
                continue
            images = images[mask]
            labels = labels[mask]
            orig_scores = orig_scores[mask]
            file_names = [fn for i, fn in enumerate(file_names) if mask[i]]

        if len(file_names) == 0:
            continue

        # Opti-CAM
        sal_start = time.time()
        saliency_maps, masked_images, last_feature = run_opticam(
            model_seq, device, images, labels, target_layers, optcam_cfg
        )
        sal_time = time.time() - sal_start
        total_saliency_time += sal_time

        # Masked inference
        m0 = time.time()
        with torch.no_grad():
            logits_masked = model_seq(masked_images.to(device))
            probs_masked = torch.softmax(logits_masked, dim=1)
            masked_scores = probs_masked[range(images.size(0)), labels]
        total_mask_infer_time += (time.time() - m0)

        sal_np = saliency_maps.detach().cpu().numpy()
        feat_np = last_feature.detach().cpu().numpy()
        timed_batches += 1

        # Per-image metrics + save
        for i in range(images.size(0)):
            fname = sanitize_name(file_names[i])
            y = float(orig_scores[i].item())
            o = float(masked_scores[i].item())
            drop_percent, inc_flag, gain_value, used_flag = compute_metrics_per_sample(
                y, o, min_orig=min_orig, mode=metric_mode
            )
            if used_flag:
                counted_samples += 1
                sum_drop += drop_percent
                if inc_flag:
                    increase_count += 1
                    sum_gain += gain_value

            # Insertion/Deletion AUC
            ins_auc, del_auc, _, _ = insertion_deletion_auc(
                model=model_seq,
                image=images[i].detach(),
                saliency_map=saliency_maps[i].detach(),
                label=int(labels[i].item()),
                device=device,
                patch_h=optcam_cfg['id_patch_h'],
                patch_w=optcam_cfg['id_patch_w'],
                steps=optcam_cfg['id_steps'],
                replacement=optcam_cfg['id_replacement'],
                use_logit=not optcam_cfg['id_use_prob']
            )
            ins_aucs.append(ins_auc)
            del_aucs.append(del_auc)

            # MoRF AOPC
            _, morf_aopc, _, _ = compute_morf_single(
                model=model_seq,
                image=images[i].detach(),
                saliency_map=saliency_maps[i].detach(),
                label=int(labels[i].item()),
                device=device,
                patch_h=optcam_cfg['id_patch_h'],
                patch_w=optcam_cfg['id_patch_w'],
                replacement=optcam_cfg['id_replacement'],
                use_logit=not optcam_cfg['id_use_prob'],
                max_fraction=optcam_cfg['id_max_fraction']
            )
            morf_aopcs.append(morf_aopc)

            # Save artifacts
            np.save(os.path.join(sal_dir, f"{fname}_{tag}.npy"), sal_np[i])
            np.save(os.path.join(feat_dir, f"{fname}_{tag}.npy"), feat_np[i])

            img_uint8 = (images[i].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            img_float = img_uint8 / 255.0
            sal_map = np.transpose(sal_np[i], (1, 2, 0))  # HxWx1
            overlay = show_cam_on_image(img_float, sal_map)
            cv2.imwrite(os.path.join(images_dir, f"{fname}_Smap_{tag}.png"), overlay)

            total_saved += 1

            per_image_rows.append({
                'file_name': file_names[i],
                'label': int(labels[i].item()),
                'orig_score': y,
                'masked_score': o,
                'drop_percent': drop_percent,
                'increase_flag': inc_flag,
                'gain_value': gain_value,
                'used_for_metrics': used_flag,
                'insertion_auc': ins_auc,
                'deletion_auc': del_auc,
                'morf_aopc': morf_aopc,
                'model_tag': tag
            })

        print(f"[{tag}][Batch {batch_idx}] saved {images.size(0)} items (running total={total_saved}) | saliency={sal_time:.3f}s")

    # Summaries
    if counted_samples > 0:
        AD = sum_drop / counted_samples
        AI = (increase_count / counted_samples) * 100.0
        AG = (sum_gain / increase_count) if increase_count > 0 else 0.0
    else:
        AD = AI = AG = 0.0

    mean_ins_auc = float(np.mean(ins_aucs)) if len(ins_aucs) > 0 else 0.0
    mean_del_auc = float(np.mean(del_aucs)) if len(del_aucs) > 0 else 0.0
    mean_morf_aopc = float(np.mean(morf_aopcs)) if len(morf_aopcs) > 0 else 0.0

    # Timing
    avg_saliency_per_batch = total_saliency_time / timed_batches if timed_batches > 0 else 0.0
    avg_saliency_per_used_image = total_saliency_time / counted_samples if counted_samples > 0 else 0.0
    avg_mask_infer_per_used_image = total_mask_infer_time / counted_samples if counted_samples > 0 else 0.0
    avg_orig_infer_per_raw_image = total_original_infer_time / raw_samples if raw_samples > 0 else 0.0

    # Write files
    summary_path = os.path.join(metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model tag: {tag}\n")
        f.write(f"raw_samples: {raw_samples}\n")
        f.write(f"used_samples: {counted_samples}\n")
        f.write(f"mode: {metric_mode}\n")
        f.write(f"min_orig: {min_orig}\n")
        f.write(f"only_correct: {only_correct}\n")
        f.write(f"AD (Average Drop %)             : {AD:.6f}\n")
        f.write(f"AI (Average Increase %)         : {AI:.6f}\n")
        f.write(f"AG (Average Gain pp)            : {AG:.6f}\n")
        f.write(f"Insertion AUC (mean)            : {mean_ins_auc:.6f}\n")
        f.write(f"Deletion AUC (mean)             : {mean_del_auc:.6f}\n")
        f.write(f"MoRF AOPC (mean)                : {mean_morf_aopc:.6f}\n")
        f.write("# Timing (per-run)\n")
        f.write(f"saliency_total_sec: {total_saliency_time:.6f}\n")
        f.write(f"masked_infer_total_sec: {total_mask_infer_time:.6f}\n")
        f.write(f"original_infer_total_sec: {total_original_infer_time:.6f}\n")
        f.write(f"avg_saliency_per_batch_sec: {avg_saliency_per_batch:.6f}\n")
        f.write(f"avg_saliency_per_used_image_sec: {avg_saliency_per_used_image:.6f}\n")
        f.write(f"avg_masked_infer_per_used_image_sec: {avg_mask_infer_per_used_image:.6f}\n")
        f.write(f"avg_original_infer_per_raw_image_sec: {avg_orig_infer_per_raw_image:.6f}\n")

    per_image_csv = os.path.join(metrics_dir, "metrics_per_image.csv")
    with open(per_image_csv, "w", newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            'file_name', 'label', 'orig_score', 'masked_score',
            'drop_percent', 'increase_flag', 'gain_value', 'used_for_metrics',
            'insertion_auc', 'deletion_auc', 'morf_aopc', 'model_tag'
        ])
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow(row)

    print(f"[{tag}] Summary saved to: {summary_path}")
    print(f"[{tag}] Per-image CSV saved to: {per_image_csv}")