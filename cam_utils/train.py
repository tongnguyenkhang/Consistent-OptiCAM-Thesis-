import os
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from imagenet_loader import ImageNetLoader

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_loader(images_root: str,
               csv_path: str,
               transform,
               batch_size: int,
               shuffle: bool = True,
               num_workers: int = 2,
               pin_memory: bool = True) -> DataLoader:
    ds = ImageNetLoader(images_root, csv_path, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

def train_ce(student_seq: nn.Sequential,
             train_loader: DataLoader,
             device: torch.device,
             epochs: int = 5,
             lr: float = 1e-3,
             save_path: Optional[str] = None) -> Dict[str, Any]:
    student_backbone = student_seq[1]
    optimizer = optim.SGD(student_backbone.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    student_seq.train()
    hist = {"epoch_loss": []}
    t0 = time.time()
    for ep in range(epochs):
        ep_loss, n_batches = 0.0, 0
        for images, labels, _ in train_loader:
            images = images.to(device); labels = labels.to(device)
            logits = student_seq(images)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()); n_batches += 1
        avg = ep_loss / max(1, n_batches)
        hist["epoch_loss"].append(avg)
        print(f"[CE][{ep+1}/{epochs}] loss={avg:.4f}")
    hist["wall_sec"] = time.time() - t0
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        torch.save(student_backbone.state_dict(), save_path)
        print(f"[CE][SAVE] -> {save_path}")
    return hist

def train_kd_fixed(teacher_seq: nn.Sequential,
                   student_seq: nn.Sequential,
                   train_loader: DataLoader,
                   device: torch.device,
                   epochs: int = 5,
                   lr: float = 8e-4,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    KD cố định: KL(student || teacher) với nhiệt độ T=1.0, không alpha.
    """
    teacher_seq.eval()
    student_backbone = student_seq[1]
    optimizer = optim.SGD(student_backbone.parameters(), lr=lr, momentum=0.9)
    kldiv = nn.KLDivLoss(reduction="batchmean")
    student_seq.train()
    hist = {"epoch_kd": []}
    t0 = time.time()
    for ep in range(epochs):
        ep_loss, n_batches = 0.0, 0
        for images, _, _ in train_loader:
            images = images.to(device)
            with torch.no_grad():
                t_logits = teacher_seq(images)
                t_prob = torch.softmax(t_logits, dim=1)
            s_logits = student_seq(images)
            s_logprob = torch.log_softmax(s_logits, dim=1)
            loss = kldiv(s_logprob, t_prob)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()); n_batches += 1
        avg = ep_loss / max(1, n_batches)
        hist["epoch_kd"].append(avg)
        print(f"[KD][{ep+1}/{epochs}] kl={avg:.4f}")
    hist["wall_sec"] = time.time() - t0
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        torch.save(student_backbone.state_dict(), save_path)
        print(f"[KD][SAVE] -> {save_path}")
    return hist