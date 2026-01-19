#!/usr/bin/env python3
"""
generate_log_tables.py

- Ghi bảng log duy nhất dạng .txt và LUÔN append, không tạo .tsv, không render phụ.
- Dùng trong generate_opticam_multi để ghi:
  + 1 hàng cho mỗi canonical loss (abs|rel|mse).
  + (tuỳ chọn) 1 hàng tổng hợp 'all_summary' khi chạy đồng thời (canonical_loss=all).

Định dạng:
- Header được viết 1 lần khi file chưa tồn tại.
- Mỗi dòng dữ liệu căn theo độ rộng cột "cố định" (width hint), dài quá sẽ rút gọn với '…'.
- Có dòng gạch ngang sau mỗi hàng để dễ nhìn.
"""
import os
from typing import Dict, List, Optional, Iterable, Tuple

# ====================== Public API ======================

def default_columns() -> List[str]:
    # Cột mở rộng có các thông số tổng hợp cho cả 3 loss
    return [
        "run_no",
        "script",
        "name_path",
        "canonical_loss",
        "use_union_loss",
        "union_rule",
        "num_masks",
        "max_iter",
        "learning_rate",
        "batch_size",
        "metric_mode",
        "min_orig",
        "only_correct",
        "residual_weight",
        "c_reg",
        # summaries per-loss (điền ở hàng loss tương ứng hoặc ở all_summary)
        "final_abs", "final_rel", "final_mse",
        "mean_final_abs", "mean_final_rel", "mean_final_mse",
        "best_abs", "best_rel", "best_mse",
        "first_abs", "first_rel", "first_mse",
        # primary metrics
        "raw_samples",
        "used_samples",
        "AD",
        "AI",
        "AG",
        "AUC_Insertion",
        "AUC_Deletion",
        "AOPC_Insertion",
        "AOPC_Deletion",
        "avg_saliency_per_batch",
        "avg_elapsed_time_per_batch",
        "global_runtime",
    ]

def append_row_from_flags(
    *,
    base_dir: str,
    FLAGS,
    OptCAM,
    loss_suffix: str,
    raw_samples: int,
    counted_samples: int,
    AD, AI, AG,
    AUC_INS_str: str,
    AUC_DEL_str: str,
    AOPC_INS_str: str,
    AOPC_DEL_str: str,
    avg_saliency_per_batch: float,
    avg_elapsed_time_per_batch: float,
    global_runtime: float,
    script_name: str = "generate_opticam_multi.py",
    log_name: str = "run_log",
    use_unicode: bool = True,   # giữ tham số để tương thích; hiện tại chỉ dùng ASCII
    # summaries (tuỳ chọn) — điền cho loss hiện tại hoặc cho hàng all_summary
    final_abs: Optional[float] = None,
    final_rel: Optional[float] = None,
    final_mse: Optional[float] = None,
    mean_final_abs: Optional[float] = None,
    mean_final_rel: Optional[float] = None,
    mean_final_mse: Optional[float] = None,
    best_abs: Optional[float] = None,
    best_rel: Optional[float] = None,
    best_mse: Optional[float] = None,
    first_abs: Optional[float] = None,
    first_rel: Optional[float] = None,
    first_mse: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Ghi 1 hàng vào BẢNG TXT DUY NHẤT (append-only).
    Trả về (path_txt, path_txt) để giữ tương thích chữ ký cũ.
    """
    path_txt = _to_txt_path(os.path.join(base_dir, log_name))
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)

    # Chuẩn bị header nếu file chưa có
    if not os.path.exists(path_txt) or os.path.getsize(path_txt) == 0:
        _write_header(path_txt, default_columns())

    # Lấy run_no kế tiếp từ file .txt hiện tại
    next_run_no = _table_next_run_no(path_txt)

    # Chỉ điền giá trị tóm tắt đúng loss tương ứng hoặc all_summary
    def pick_for_loss(val, key_suffix: str):
        if loss_suffix == "all_summary":
            return "" if (val is None) else val
        if loss_suffix == key_suffix:
            return "" if (val is None) else val
        return ""

    row = {
        "run_no": next_run_no,
        "script": script_name,
        "name_path": getattr(FLAGS, "name_path", ""),
        "canonical_loss": loss_suffix,
        "use_union_loss": str(getattr(FLAGS, "use_union_loss", "")),
        "union_rule": getattr(FLAGS, "union_rule", ""),
        "num_masks": getattr(FLAGS, "num_masks", ""),
        "max_iter": getattr(FLAGS, "max_iter", ""),
        "learning_rate": getattr(FLAGS, "learning_rate", ""),
        "batch_size": getattr(FLAGS, "batch_size", ""),
        "metric_mode": getattr(FLAGS, "metric_mode", ""),
        "min_orig": getattr(FLAGS, "min_orig", ""),
        "only_correct": str(getattr(FLAGS, "only_correct", "")),
        "residual_weight": getattr(OptCAM, "residual_weight", ""),
        "c_reg": getattr(OptCAM, "c_reg", ""),
        # summaries
        "final_abs": pick_for_loss(final_abs, "abs"),
        "final_rel": pick_for_loss(final_rel, "rel"),
        "final_mse": pick_for_loss(final_mse, "mse"),
        "mean_final_abs": pick_for_loss(mean_final_abs, "abs"),
        "mean_final_rel": pick_for_loss(mean_final_rel, "rel"),
        "mean_final_mse": pick_for_loss(mean_final_mse, "mse"),
        "best_abs": pick_for_loss(best_abs, "abs"),
        "best_rel": pick_for_loss(best_rel, "rel"),
        "best_mse": pick_for_loss(best_mse, "mse"),
        "first_abs": pick_for_loss(first_abs, "abs"),
        "first_rel": pick_for_loss(first_rel, "rel"),
        "first_mse": pick_for_loss(first_mse, "mse"),
        # primary
        "raw_samples": raw_samples,
        "used_samples": counted_samples,
        "AD": AD,
        "AI": AI,
        "AG": AG,
        "AUC_Insertion": AUC_INS_str if AUC_INS_str != "N/A" else "",
        "AUC_Deletion": AUC_DEL_str if AUC_DEL_str != "N/A" else "",
        "AOPC_Insertion": AOPC_INS_str if AOPC_INS_str != "N/A" else "",
        "AOPC_Deletion": AOPC_DEL_str if AOPC_DEL_str != "N/A" else "",
        "avg_saliency_per_batch": avg_saliency_per_batch,
        "avg_elapsed_time_per_batch": avg_elapsed_time_per_batch,
        "global_runtime": global_runtime,
    }

    _append_row_txt(path_txt, row, default_columns())
    return path_txt, path_txt  # giữ tương thích API cũ


def append_summary_row_from_multi(
    *,
    base_dir: str,
    FLAGS,
    OptCAM,
    raw_samples: int,
    counted_samples: int,
    AD, AI, AG,
    AUC_INS_str: str,
    AUC_DEL_str: str,
    AOPC_INS_str: str,
    AOPC_DEL_str: str,
    avg_saliency_per_batch: float,
    avg_elapsed_time_per_batch: float,
    global_runtime: float,
    batch_final_losses_by: Dict[str, List[float]],
    batch_loss_histories_by: Dict[str, List[List[float]]],
    script_name: str = "generate_opticam_multi.py",
    log_name: str = "run_log",
    use_unicode: bool = True
) -> Tuple[str, str]:
    """
    Ghi 1 hàng tổng hợp ('all_summary') chứa đủ final/mean/best/first cho abs, rel, mse.
    """
    def stats(loss_key: str):
        finals = batch_final_losses_by.get(loss_key, [])
        histories = batch_loss_histories_by.get(loss_key, [])
        final_last = finals[-1] if finals else None
        mean_final = (sum(finals) / len(finals)) if finals else None
        best_val = min((min(h) for h in histories), default=None) if histories else None
        first_val = histories[0][0] if (histories and histories[0]) else None
        return final_last, mean_final, best_val, first_val

    final_abs, mean_abs, best_abs, first_abs = stats("abs")
    final_rel, mean_rel, best_rel, first_rel = stats("rel")
    final_mse, mean_mse, best_mse, first_mse = stats("mse")

    return append_row_from_flags(
        base_dir=base_dir,
        FLAGS=FLAGS,
        OptCAM=OptCAM,
        loss_suffix="all_summary",
        raw_samples=raw_samples,
        counted_samples=counted_samples,
        AD=AD, AI=AI, AG=AG,
        AUC_INS_str=AUC_INS_str, AUC_DEL_str=AUC_DEL_str,
        AOPC_INS_str=AOPC_INS_str, AOPC_DEL_str=AOPC_DEL_str,
        avg_saliency_per_batch=avg_saliency_per_batch,
        avg_elapsed_time_per_batch=avg_elapsed_time_per_batch,
        global_runtime=global_runtime,
        script_name=script_name,
        log_name=log_name,
        use_unicode=use_unicode,
        final_abs=final_abs, final_rel=final_rel, final_mse=final_mse,
        mean_final_abs=mean_abs, mean_final_rel=mean_rel, mean_final_mse=mean_mse,
        best_abs=best_abs, best_rel=best_rel, best_mse=best_mse,
        first_abs=first_abs, first_rel=first_rel, first_mse=first_mse
    )

# ====================== Internal helpers (append-only TXT) ======================

def _to_txt_path(base_path: str) -> str:
    return base_path if base_path.endswith(".txt") else base_path + ".txt"

def _width_hints(columns: List[str]) -> List[int]:
    # Độ rộng cố định hợp lý cho từng cột (giữ bảng gọn, không re-render)
    hint = {
        "run_no": 6, "script": 22, "name_path": 20, "canonical_loss": 14,
        "use_union_loss": 5, "union_rule": 10, "num_masks": 9, "max_iter": 8,
        "learning_rate": 12, "batch_size": 10, "metric_mode": 10, "min_orig": 9,
        "only_correct": 12, "residual_weight": 15, "c_reg": 8,
        "final_abs": 10, "final_rel": 10, "final_mse": 10,
        "mean_final_abs": 14, "mean_final_rel": 14, "mean_final_mse": 14,
        "best_abs": 10, "best_rel": 10, "best_mse": 10,
        "first_abs": 10, "first_rel": 10, "first_mse": 10,
        "raw_samples": 11, "used_samples": 12,
        "AD": 8, "AI": 8, "AG": 8,
        "AUC_Insertion": 14, "AUC_Deletion": 14,
        "AOPC_Insertion": 15, "AOPC_Deletion": 15,
        "avg_saliency_per_batch": 22, "avg_elapsed_time_per_batch": 26, "global_runtime": 14,
    }
    return [max(len(c), hint.get(c, 10)) for c in columns]

def _is_number_str(s: str) -> bool:
    if s is None or s == "":
        return False
    try:
        float(str(s).replace(",", ""))
        return True
    except Exception:
        return False

def _format_cell(val, width: int) -> str:
    s = "" if val is None else str(val)
    # Truncate nếu quá dài
    if len(s) > width:
        s = s[: max(0, width - 1)] + "…"
    pad = width - len(s)
    # số: căn phải, text: căn trái
    return (" " * pad + s) if _is_number_str(s) else (s + " " * pad)

def _write_header(path_txt: str, columns: List[str]) -> None:
    widths = _width_hints(columns)
    header_cells = [_format_cell(c, w) for c, w in zip(columns, widths)]
    line = " | ".join(header_cells)
    sep = "-+-".join(["-" * w for w in widths])
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(line + "\n")
        f.write(sep + "\n")

def _table_next_run_no(path_txt: str) -> int:
    """
    Tìm run_no kế tiếp bằng cách đọc dòng dữ liệu cuối cùng.
    Giả định run_no là cột đầu tiên.
    """
    if not os.path.exists(path_txt) or os.path.getsize(path_txt) == 0:
        return 1
    try:
        with open(path_txt, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        # Bỏ header (2 dòng đầu) và các dòng phân cách '-----'
        data_lines = [ln for ln in lines[2:] if not set(ln) <= set("-+ ")]
        if not data_lines:
            return 1
        last = data_lines[-1]
        first_col = last.split("|", 1)[0].strip()
        return int(first_col) + 1
    except Exception:
        # Fallback: đếm số dòng dữ liệu
        try:
            with open(path_txt, "r", encoding="utf-8") as f:
                cnt = sum(1 for i, ln in enumerate(f) if i >= 2 and ln.strip() and not set(ln.strip()) <= set("-+"))
            return cnt + 1
        except Exception:
            return 1

def _append_row_txt(path_txt: str, row: Dict[str, object], columns: List[str]) -> None:
    widths = _width_hints(columns)
    values = [_format_cell(row.get(col, ""), w) for col, w in zip(columns, widths)]
    line = " | ".join(values)
    divider = "-" * len(line)
    with open(path_txt, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.write(divider + "\n")