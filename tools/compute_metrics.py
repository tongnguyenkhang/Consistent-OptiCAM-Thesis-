import numpy as np
from typing import List, Tuple, Optional
import torch

# Advanced-metrics defaults (có thể thay đổi tuỳ run nếu cần)
ID_STEPS = 70
ID_BASELINE = "blur"   # "black" | "blur"
ID_BLUR_KSIZE = 11

# --------- Primary helpers (cat/dog, primary metrics) ----------
def build_cat_dog_index_sets(categories: List[str]) -> Tuple[List[int], List[int]]:
    cat_indices = []
    dog_indices = []
    dog_keywords = {"dog", "terrier", "retriever", "shepherd", "bulldog", "pug", "beagle",
                    "spaniel", "rottweiler", "husky", "chihuahua", "doberman", "schnauzer",
                    "mastiff", "akita", "malamute", "airedale", "whippet", "dalmatian",
                    "basset", "papillon", "pointer", "greyhound", "wolfhound", "corgi",
                    "boxer", "pomeranian", "shih", "samoyed", "saluki", "redbone", "bloodhound",
                    "otterhound", "irish", "japanese", "keeshond"}
    for idx, name in enumerate(categories):
        lname = name.lower()
        if 'cat' in lname or 'kitten' in lname or 'tiger cat' in lname:
            cat_indices.append(idx)
        else:
            for kw in dog_keywords:
                if kw in lname:
                    dog_indices.append(idx)
                    break
    return cat_indices, dog_indices

def compute_metrics_per_sample(y, o, min_orig=0.05, eps=1e-12):
    """
    Compute AD, AI, AG metrics per sample following OptiCAM paper (Equations 13-15).
    
    Returns:
        tuple: (ad_value, ai_flag, ag_value, used_flag)
        - ad_value: Average Drop component = |p-o|+ / p (zero if o>p)
        - ai_flag: 1 if o > p (confidence increase), else 0
        - ag_value: Average Gain component = |o-p|+ / (1-p) (zero if o<=p)
        - used_flag: 1 if sample used for metrics (p >= min_orig), else 0
    
    Paper formulas (ALL samples, not just drop/increase):
        AD (Eq 13): (1/N) Σ |p-o|+ / p          (mask outside salient)
        AI (Eq 14): (1/N) Σ 1_{o>p}             (% with increase)
        AG (Eq 15): (1/N) Σ |o-p|+ / (1-p)      (mask inside salient, symmetric to AD)
    
    Where |x|+ = max(0, x) (positive part)
    
    NOTE: Mode parameter removed - AG always uses (1-p) normalization per paper.
    """
    y = max(float(y), eps)
    o = float(o)
    
    # Filter low-confidence samples
    if y < min_orig:
        return 0.0, 0, 0.0, 0
    
    # AD component: |p-o|+ / p (positive part of drop)
    ad_value = max(0.0, y - o) / y * 100.0
    
    # AI flag: 1 if increase
    ai_flag = 1 if o > y else 0
    
    # AG component: |o-p|+ / (1-p) (positive part of gain, symmetric to AD)
    denom = max(1.0 - y, eps)
    ag_value = max(0.0, o - y) / denom * 100.0
    
    return ad_value, ai_flag, ag_value, 1

# --------- Mask utils for multi-component visualization/union ----------
def _normalize_map01(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = m.astype(np.float32)
    m = m - np.min(m)
    denom = float(np.max(m))
    if denom < eps:
        return np.zeros_like(m, dtype=np.float32)
    return m / (denom + eps)

def _build_combined_mask(masks_np: np.ndarray, mode: str = "max", scores: Optional[List[float]] = None, temp: float = 1.0) -> np.ndarray:
    """
    Build combined/aggregated mask from K component masks.
    
    Args:
        masks_np: (K,H,W) component masks in [0,1]
        mode: Aggregation method
            - "max": per-pixel maximum across components
            - "prob_or": probabilistic union (1 - ∏(1 - m_k))
            - "weighted": weighted sum using scores
        scores: Component scores (length K) when mode == "weighted"
        temp: Temperature for softmax weighting in "weighted" mode
    
    Returns:
        Combined mask (H,W) in [0,1]
    """
    K = int(masks_np.shape[0])
    if K == 1:
        return masks_np[0]

    if mode == "prob_or":
        one_minus = 1.0 - np.clip(masks_np, 0.0, 1.0)
        prod = np.prod(one_minus, axis=0)
        combined = 1.0 - prod
        return np.clip(combined, 0.0, 1.0)

    if mode == "weighted" and scores is not None and len(scores) == K:
        s = np.asarray(scores, dtype=np.float32)
        t = max(float(temp), 1e-6)
        w = np.exp(s / t)
        w = w / (np.sum(w) + 1e-6)
        combined = np.sum(w[:, None, None] * masks_np, axis=0)
        return _normalize_map01(combined)

    # default: per-pixel max
    return np.max(masks_np, axis=0)

def _make_comp_display_map(m: np.ndarray, share: float, combined_viz: Optional[np.ndarray], mode: str) -> np.ndarray:
    """
    Create display map for individual component visualization.
    
    Args:
        m: Raw component mask [0,1]
        share: Component's score share (contribution ratio)
        combined_viz: Combined mask for reference (optional)
        mode: Visualization mode
            - "importance": Scale by importance share
            - "combined_clip": Clip by combined mask
            - "raw": Use raw mask
    """
    # m: raw mask [0,1] for one component
    if mode == "importance":
        return _normalize_map01(m * max(share, 0.0))
    if mode == "combined_clip":
        if combined_viz is None:
            return _normalize_map01(m)
        return _normalize_map01(m * combined_viz)
    # raw
    return _normalize_map01(m)

# --------- Advanced metrics (Insertion/Deletion) ----------
def _am_make_baseline(img_float: np.ndarray, mode: str = "black", blur_ksize: int = 11) -> np.ndarray:
    """
    Xây baseline cho insertion/deletion:
      - "black": ảnh 0
      - "blur": Gaussian blur (import cv2 lazy)
    """
    if mode == "black":
        return np.zeros_like(img_float, dtype=np.float32)
    if mode == "blur":
        import cv2
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur((img_float * 255.0).astype(np.uint8), (k, k), 0)
        return blur.astype(np.float32) / 255.0
    raise ValueError(f"Unknown baseline mode: {mode}")

def _am_score_np_image(model, device, x_np: np.ndarray, label_scalar: int,
                       idx_cat=None, idx_dog=None, use_logit: bool = False) -> float:
    """
    Chấm điểm 1 ảnh numpy HxWx3 qua PyTorch model.
    Nếu có idx_cat/idx_dog (ImageNet aggregation), dùng tổng prob theo species.
    """
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
    """
    Tạo danh sách binary masks chọn top (t/steps) pixel theo saliency (giảm dần), t=0..steps.
    """
    H, W = s_map.shape
    flat = s_map.reshape(-1)
    order = np.argsort(flat)[::-1]  # high->low
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
    # alias sang cùng cách normalize chung
    return _normalize_map01(m, eps=eps)

def _am_compute_id_values(
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
    use_logit: bool = False,
    need_ins: bool = True,
    need_del: bool = True,
):
    """
    Tính y, baseline và chỉ tính các curve cần thiết:
      - need_ins: có cần insertion curve không
      - need_del: có cần deletion curve không
    Trả về: (ins_curve or None, del_curve or None, y, bscore)
    """
    union_norm = _am_normalize01(union_mask01)
    topk_masks = _am_build_topk_masks_from_ranking(union_norm, steps)
    baseline = _am_make_baseline(img_float, baseline_mode, blur_ksize)

    y = _am_score_np_image(model, device, img_float, label_scalar, idx_cat, idx_dog, use_logit)
    bscore = _am_score_np_image(model, device, baseline, label_scalar, idx_cat, idx_dog, use_logit)

    ins_vals = [] if need_ins else None
    del_vals = [] if need_del else None

    for Mt in topk_masks:
        Mt3 = Mt[:, :, None]
        if need_ins:
            x_ins = baseline * (1.0 - Mt3) + img_float * Mt3
            s_ins = _am_score_np_image(model, device, x_ins, label_scalar, idx_cat, idx_dog, use_logit)
            ins_vals.append(s_ins)
        if need_del:
            x_del = img_float * (1.0 - Mt3) + baseline * Mt3
            s_del = _am_score_np_image(model, device, x_del, label_scalar, idx_cat, idx_dog, use_logit)
            del_vals.append(s_del)

    ins_curve = np.array(ins_vals, dtype=np.float32) if need_ins else None
    del_curve = np.array(del_vals, dtype=np.float32) if need_del else None
    return ins_curve, del_curve, y, bscore

def advanced_metrics(
    img_float: np.ndarray,
    union_mask01: np.ndarray,
    model,
    device,
    label_scalar: int,
    *,
    # species aggregation (optional)
    idx_cat=None,
    idx_dog=None,
    # scoring space
    use_logit: bool = False,
    # ID settings
    steps: int = 100,
    baseline: str = "black",      # "black" | "blur"
    blur_ksize: int = 11,
    # toggles
    enable_ins_auc: bool = True,
    enable_del_auc: bool = True,
    enable_aopc_ins: bool = True,
    enable_aopc_del: bool = True,
    return_curves: bool = False
) -> dict:
    """
    Tính Insertion/Deletion AUC và AOPC (có toggle bật/tắt từng metric).
    Tối ưu: chỉ tính curve cần thiết. Nếu tất cả enable_* = False và return_curves=False,
    trả về {} ngay để không tốn compute.
    """
    if not (enable_ins_auc or enable_del_auc or enable_aopc_ins or enable_aopc_del or return_curves):
        return {}

    need_ins = enable_ins_auc or enable_aopc_ins or return_curves
    need_del = enable_del_auc or enable_aopc_del or return_curves

    union_mask01 = _am_normalize01(union_mask01)

    ins_curve, del_curve, y, bscore = _am_compute_id_values(
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
        use_logit=bool(use_logit),
        need_ins=need_ins,
        need_del=need_del,
    )

    out = {
        "orig_score": y,
        "baseline_score": bscore,
        "id_steps": int(steps),
        "id_baseline": str(baseline),
        "id_blur_ksize": int(blur_ksize),
    }

    if need_ins:
        xs_ins = np.linspace(0.0, 1.0, num=len(ins_curve), dtype=np.float32)
        if enable_ins_auc:
            out["auc_insertion"] = float(np.trapz(ins_curve, xs_ins))
        if enable_aopc_ins:
            out["aopc_insertion"] = float(np.mean(ins_curve[1:] - bscore)) if len(ins_curve) > 1 else 0.0

    if need_del:
        xs_del = np.linspace(0.0, 1.0, num=len(del_curve), dtype=np.float32)
        if enable_del_auc:
            out["auc_deletion"] = float(np.trapz(del_curve, xs_del))
        if enable_aopc_del:
            out["aopc_deletion"] = float(np.mean(y - del_curve[1:])) if len(del_curve) > 1 else 0.0

    if return_curves:
        if need_ins:
            out["insertion_curve"] = ins_curve
        if need_del:
            out["deletion_curve"] = del_curve

    return out