import os
import csv
import argparse
from typing import List

import torch
import numpy as np
from PIL import Image

# We import weight enums lazily depending on chosen arch to avoid unnecessary downloads
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def list_image_files(root_dir: str) -> List[str]:
    files = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isfile(path):
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_EXTS:
                files.append(path)
    files.sort()
    return files


def load_and_preprocess(image_path: str, preprocess, device):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor


def build_cat_dog_index_sets(categories: List[str]) -> tuple[set, set]:
    cat_indices = set()
    dog_indices = set()
    dog_keywords = {"dog", "terrier", "retriever", "shepherd", "bulldog", "pug", "beagle",
                    "spaniel", "rottweiler", "husky", "chihuahua", "doberman", "schnauzer",
                    "mastiff", "akita", "malamute", "airedale", "whippet", "dalmatian",
                    "basset", "papillon", "pointer", "greyhound", "wolfhound", "corgi",
                    "boxer", "pomeranian", "shih", "samoyed", "saluki", "redbone", "bloodhound",
                    "otterhound", "irish", "japanese", "keeshond"}
    for idx, name in enumerate(categories):
        lname = name.lower()
        if 'cat' in lname or 'kitten' in lname or 'tiger cat' in lname:
            cat_indices.add(idx)
        else:
            for kw in dog_keywords:
                if kw in lname:
                    dog_indices.add(idx)
                    break
    return cat_indices, dog_indices


def make_model_for_checkpoint(arch: str, device):
    # Build an architecture compatible with a 2-class checkpoint
    if arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif arch == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
    else:
        # resnet50
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Classify images into cat/dog and copy them into subfolders.")
    parser.add_argument("--image_dir", type=str, default="./images", help="Folder containing input images")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet18", "mobilenet_v2"], help="Backbone to use when mapping with ImageNet (resnet50 gives best mapping) or for checkpoint architecture")
    parser.add_argument("--mode", type=str, choices=["map", "checkpoint"], default="map", help="'map' uses ImageNet mapping; 'checkpoint' loads a 2-class checkpoint expecting (cat=0,dog=1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="(optional) path to PyTorch checkpoint for 2-class classifier")
    # We only write a CSV (file_name,label) like `revisited_imagenet_2012_val.csv`.
    parser.add_argument("--output_csv", type=str, default="./revisited_imagenet_2012_val.csv",
                        help="Path to write CSV results (file_name,label)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare model and preprocessing
    if args.mode == 'checkpoint':
        if not args.checkpoint:
            raise ValueError("--checkpoint required when --mode checkpoint")
        model = make_model_for_checkpoint('resnet50' if args.arch == 'resnet50' else args.arch, device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        # use ResNet50 transforms for checkpoint inputs
        preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()
        categories = None
        cat_indices = set()
        dog_indices = set()
    else:
        # mapping: load a pretrained ImageNet model to map classes
        if args.arch == 'mobilenet_v2':
            from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            model = mobilenet_v2(weights=weights).to(device).eval()
            preprocess = weights.transforms()
        elif args.arch == 'resnet18':
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights).to(device).eval()
            preprocess = weights.transforms()
        else:
            weights = ResNet50_Weights.IMAGENET1K_V1
            model = resnet50(weights=weights).to(device).eval()
            preprocess = weights.transforms()
        try:
            categories = weights.meta['categories']
        except Exception:
            categories = None
        if categories is not None:
            cat_indices, dog_indices = build_cat_dog_index_sets(categories)
        else:
            cat_indices, dog_indices = set(), set()

    image_paths = list_image_files(args.image_dir)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} images")

    # We do not copy/move files. Only classify and write CSV.

    rows = []
    batch_tensors = []
    batch_names = []

    def flush_batch():
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

        if args.mode == 'map' and (cat_indices or dog_indices):
            probs_np = probs.cpu().numpy()
            for i_item, fname in enumerate(batch_names):
                p = probs_np[i_item]
                p_cat = float(p[list(cat_indices)].sum()) if cat_indices else 0.0
                p_dog = float(p[list(dog_indices)].sum()) if dog_indices else 0.0
                if p_cat >= p_dog:
                    label = 0  # cat
                    score = p_cat
                else:
                    label = 1  # dog
                    score = p_dog
                rows.append((fname, label, float(score)))
        else:
            # checkpoint mode or fallback: interpret logits/probs as 2-class classifier
            top1 = probs.argmax(dim=1).cpu().tolist()
            maxp = probs.max(dim=1).values.cpu().tolist()
            for fname, idx, pval in zip(batch_names, top1, maxp):
                label = 0 if int(idx) == 0 else 1
                rows.append((fname, label, float(pval)))

        batch_tensors.clear()
        batch_names.clear()

    # build batches
    for i, path in enumerate(image_paths, 1):
        try:
            t = load_and_preprocess(path, preprocess, device)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        batch_tensors.append(t)
        batch_names.append(path)
        if len(batch_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

    # write CSV of results in the simple format used by `revisited_imagenet_2012_val.csv`
    try:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["file_name", "label"])  # label will be 'cat' or 'dog'
            for fname, label, score in rows:
                writer.writerow([os.path.basename(fname), label])
        print(f"CSV saved to {args.output_csv} ({len(rows)} rows)")
    except Exception as e:
        print(f"Failed to write CSV {args.output_csv}: {e}")


if __name__ == '__main__':
    main()
