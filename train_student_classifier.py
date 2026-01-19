import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms


class ImageFolderAll(Dataset):
    """Dataset đơn giản: lấy tất cả ảnh trong 1 thư mục (đệ quy)."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
        if not files:
            raise RuntimeError(f"No images found in {root_dir}")
        self.files = sorted(files)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, path  # path chỉ để debug, không dùng trong loss


def kd_loss(student_logits, teacher_logits, T: float = 1.0):
    """Knowledge distillation loss: KL giữa phân phối teacher và student."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    # batchmean = trung bình trên batch
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Thư mục chứa toàn bộ ảnh (7390 ảnh).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                         help="Optimizer cho student: adam (mặc định) hoặc sgd (momentum=0.9).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Đường dẫn checkpoint student (.pth) để tiếp tục fine-tune.")
    parser.add_argument("--output", type=str,
                        default="student_resnet18_kd.pth",
                        help="Đường dẫn lưu weight student.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Teacher: ResNet50 pretrained, frozen
    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(device)

    # 2) Student: ResNet18 pretrained (nhỏ hơn), fine‑tune toàn bộ
    student = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Nếu có checkpoint, load để tiếp tục từ đó
    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"resume_from checkpoint not found: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu")
        student.load_state_dict(ckpt)
        print(f"Loaded student weights from checkpoint: {args.resume_from}")
    student.train()
    student.to(device)

    # 3) DataLoader
    dataset = ImageFolderAll(args.images_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Found {len(dataset)} images.")
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9)
        print("Using SGD optimizer (momentum=0.9)")
    else:
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
        print("Using Adam optimizer")

    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)
            loss = kd_loss(student_logits, teacher_logits, T=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - KD loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(student.state_dict(), args.output)
    print(f"Saved student weights to: {args.output}")


if __name__ == "__main__":
    main()