import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from train_student_classifier import ImageFolderAll, kd_loss
from tools.compute_metrics import build_cat_dog_index_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Thư mục chứa ảnh để đánh nhãn.")
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Checkpoint .pth của student ResNet18.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature dùng so sánh KD (giống khi train).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Teacher: ResNet50 (cùng weights như khi train student)
    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    teacher.eval().to(device)

    # Student: ResNet18 + load checkpoint
    student = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if not os.path.isfile(args.student_ckpt):
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student.load_state_dict(ckpt)
    student.eval().to(device)
    print(f"Loaded student checkpoint from {args.student_ckpt}")

    # Build cat/dog indices from ImageNet categories
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V1.meta.get("categories", None)
    except Exception:
        categories = None
    if categories is not None:
        cat_indices, dog_indices = build_cat_dog_index_sets(categories)
    else:
        cat_indices, dog_indices = [], []

    if len(cat_indices) > 0 and len(dog_indices) > 0:
        idx_cat = torch.tensor(cat_indices, dtype=torch.long, device=device)
        idx_dog = torch.tensor(dog_indices, dtype=torch.long, device=device)
        print(f"Cat indices: {len(cat_indices)}, Dog indices: {len(dog_indices)}")
    else:
        idx_cat = None
        idx_dog = None
        print("[Warning] Could not build cat/dog index sets; cat/dog metrics will be skipped.")

    # DataLoader (dùng chung dataset class với script train)
    dataset = ImageFolderAll(args.images_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Found {len(dataset)} images.")

    total_samples = 0
    sum_kd = 0.0
    sum_l2 = 0.0
    sum_cos = 0.0

    top1_matches = 0
    top5_matches = 0

    catdog_matches = 0
    sum_abs_cat = 0.0
    sum_abs_dog = 0.0

    T = float(args.temperature)

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            B = imgs.size(0)

            teacher_logits = teacher(imgs)
            student_logits = student(imgs)

            # KD loss (giống khi train)
            batch_kd = kd_loss(student_logits, teacher_logits, T=T)
            sum_kd += float(batch_kd.item()) * B

            # Phần phân phối softmax với temperature T
            p_t_T = F.softmax(teacher_logits / T, dim=1)
            p_s_T = F.softmax(student_logits / T, dim=1)

            # L2 distance trên phân phối
            l2_batch = torch.norm(p_t_T - p_s_T, p=2, dim=1)
            sum_l2 += float(l2_batch.sum().item())

            # Cosine similarity
            cos_batch = F.cosine_similarity(p_t_T, p_s_T, dim=1)
            sum_cos += float(cos_batch.sum().item())

            # Top-1 / Top-5 match
            teacher_top1 = teacher_logits.argmax(dim=1)
            student_top1 = student_logits.argmax(dim=1)
            top1_matches += int((teacher_top1 == student_top1).sum().item())

            teacher_top5 = teacher_logits.topk(5, dim=1).indices  # (B,5)
            student_top5 = student_logits.topk(5, dim=1).indices  # (B,5)
            # Kiểm tra teacher top-1 có mặt trong student top-5
            in_top5 = (student_top5 == teacher_top1.unsqueeze(1)).any(dim=1)
            top5_matches += int(in_top5.sum().item())

            # Cat/dog group metrics (sử dụng softmax T=1)
            if idx_cat is not None and idx_dog is not None:
                probs_t = F.softmax(teacher_logits, dim=1)
                probs_s = F.softmax(student_logits, dim=1)

                p_cat_t = probs_t[:, idx_cat].sum(dim=1)
                p_dog_t = probs_t[:, idx_dog].sum(dim=1)
                p_cat_s = probs_s[:, idx_cat].sum(dim=1)
                p_dog_s = probs_s[:, idx_dog].sum(dim=1)

                # Group label: 0=cat, 1=dog theo teacher
                group_t = (p_dog_t > p_cat_t).long()
                group_s = (p_dog_s > p_cat_s).long()
                catdog_matches += int((group_t == group_s).sum().item())

                sum_abs_cat += float((p_cat_t - p_cat_s).abs().sum().item())
                sum_abs_dog += float((p_dog_t - p_dog_s).abs().sum().item())

            total_samples += B

    if total_samples == 0:
        print("No samples processed.")
        return

    mean_kd = sum_kd / total_samples
    mean_l2 = sum_l2 / total_samples
    mean_cos = sum_cos / total_samples

    top1_acc = top1_matches / total_samples * 100.0
    top5_acc = top5_matches / total_samples * 100.0

    print("\n=== Teacher vs Student Evaluation ===")
    print(f"Total samples: {total_samples}")
    print(f"Mean KD loss (T={T}): {mean_kd:.4f}")
    print(f"Mean L2 distance (softmax/T): {mean_l2:.6f}")
    print(f"Mean cosine similarity (softmax/T): {mean_cos:.4f}")
    print(f"Top-1 match rate: {top1_acc:.2f}%")
    print(f"Top-5 match rate: {top5_acc:.2f}%")

    if idx_cat is not None and idx_dog is not None and total_samples > 0:
        catdog_acc = catdog_matches / total_samples * 100.0
        mean_abs_cat = sum_abs_cat / total_samples
        mean_abs_dog = sum_abs_dog / total_samples
        print("\n[Cat/Dog group metrics]")
        print(f"Cat/Dog group match rate: {catdog_acc:.2f}%")
        print(f"Mean |P_cat_teacher - P_cat_student|: {mean_abs_cat:.4f}")
        print(f"Mean |P_dog_teacher - P_dog_student|: {mean_abs_dog:.4f}")


if __name__ == "__main__":
    main()
