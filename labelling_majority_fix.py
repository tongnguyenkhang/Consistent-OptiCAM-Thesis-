import os
import csv
import argparse
from collections import Counter, defaultdict


def species_from_filename(fname: str) -> str:
    # Infer species by taking the filename (without extension) and
    # removing a trailing numeric instance id if present.
    # Example: 'american_pit_bull_terrier_123.jpg' -> 'american_pit_bull_terrier'
    # This handles multi-word species names (underscores within species) instead
    # of grouping by only the first token.
    base = os.path.basename(fname)
    name, _ext = os.path.splitext(base)
    parts = name.split('_')
    # If the last token is purely digits, treat it as an instance id and drop it
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    # Rejoin remaining parts as species name
    if len(parts) == 0:
        return name
    return '_'.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Relabel images by majority vote per species (prefix before '_').")
    parser.add_argument("--input_csv", type=str, default="./revisited_imagenet_2012_val.csv",
                        help="CSV file with header file_name,label (label numeric e.g. 0 or 1)")
    parser.add_argument("--output_csv", type=str, default="./revisited_imagenet_2012_val.csv",
                        help="Output CSV with relabeled entries")
    parser.add_argument("--tie_break", type=int, choices=[0, 1], default=0,
                        help="Tie-breaker label to use when counts are equal (default: 0)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the input CSV instead of writing a separate output file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    rows = []
    species_groups = defaultdict(list)

    # read input CSV
    with open(args.input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'file_name' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError("Input CSV must contain header 'file_name,label'")
        for r in reader:
            fname = r['file_name']
            try:
                label = int(r['label'])
            except Exception:
                # try stripping and converting
                label = int(r['label'].strip())
            rows.append((fname, label))
            sp = species_from_filename(fname)
            species_groups[sp].append(label)

    # compute majority per species
    species_majority = {}
    for sp, labels in species_groups.items():
        c = Counter(labels)
        # choose label with highest count; tie-break using args.tie_break
        if len(c) == 0:
            continue
        most_common = c.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            majority_label = most_common[0][0]
        else:
            # tie
            majority_label = args.tie_break
        species_majority[sp] = (majority_label, dict(c))

    # apply majority labels and count changes
    fixed_rows = []
    changed = 0
    for fname, orig_label in rows:
        sp = species_from_filename(fname)
        maj_label = species_majority.get(sp, (orig_label, {}))[0]
        fixed_rows.append((fname, maj_label))
        if maj_label != orig_label:
            changed += 1

    out_path = args.input_csv if args.inplace else args.output_csv
    # write output CSV with header file_name,label
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'label'])
        for fname, label in fixed_rows:
            writer.writerow([fname, label])

    # print summary
    print(f"Read {len(rows)} rows from {args.input_csv}")
    print(f"Processed {len(species_groups)} species; changed {changed} labels")
    # print a small per-species summary (only show up to 20 species)
    shown = 0
    for sp, (maj, counts) in sorted(species_majority.items()):
        print(f"{sp}: majority={maj}, counts={counts}")
        shown += 1
        if shown >= 20:
            break

    print(f"Wrote output CSV to {out_path}")


if __name__ == '__main__':
    main()
