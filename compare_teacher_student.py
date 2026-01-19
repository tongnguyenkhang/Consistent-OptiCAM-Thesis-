"""
Comparison Tool: Compare OptiCAM Multi results between Teacher (ResNet50) and Student (Compressed)

Usage:
    python compare_teacher_student.py \\
        --teacher_dir ./results/OptiCamMulti_ResNet50 \\
        --student_dir ./results/OptiCamMulti_MobileNet \\
        --output comparison_report.txt

Generates:
    - Detailed comparison table
    - Metrics difference analysis
    - Speed vs Accuracy trade-off evaluation
"""

import os
import numpy as np
import csv
import argparse
from pathlib import Path

def load_metrics_summary(base_dir):
    """Load metrics from metrics_summary.txt"""
    summary_path = os.path.join(base_dir, 'metrics', 'metrics_summary.txt')
    
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found")
        return None
    
    metrics = {}
    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                metrics[key] = value
    
    return metrics

def parse_metric_value(value_str):
    """Parse metric value from string (handle N/A)"""
    if value_str == 'N/A' or value_str == '':
        return None
    try:
        return float(value_str)
    except:
        return None

def load_per_image_csv(base_dir):
    """Load per-image metrics CSV"""
    csv_path = os.path.join(base_dir, 'metrics', 'total', 'metrics_per_image.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    return rows

def compare_metrics(teacher_metrics, student_metrics):
    """Compare key metrics between teacher and student"""
    
    comparison = {}
    
    # Extract key metrics
    keys_to_compare = [
        'AD (Average Drop %, Eq 13)',
        'AI (Average Increase %, Eq 14)',
        'AG (Average Gain %, Eq 15)',
        'AUC Insertion',
        'AUC Deletion',
        'AOPC Insertion',
        'AOPC Deletion',
        'Global total runtime',
        'Avg saliency per used image',
        'Consistency error |Σc_k - c|'
    ]
    
    for key in keys_to_compare:
        teacher_val = parse_metric_value(teacher_metrics.get(key, 'N/A'))
        student_val = parse_metric_value(student_metrics.get(key, 'N/A'))
        
        if teacher_val is not None and student_val is not None:
            diff = student_val - teacher_val
            diff_pct = (diff / teacher_val * 100) if teacher_val != 0 else 0
            comparison[key] = {
                'teacher': teacher_val,
                'student': student_val,
                'diff': diff,
                'diff_pct': diff_pct
            }
    
    return comparison

def generate_report(teacher_dir, student_dir, output_path):
    """Generate comprehensive comparison report"""
    
    print(f"Loading teacher metrics from: {teacher_dir}")
    teacher_metrics = load_metrics_summary(teacher_dir)
    
    print(f"Loading student metrics from: {student_dir}")
    student_metrics = load_metrics_summary(student_dir)
    
    if teacher_metrics is None or student_metrics is None:
        print("Error: Could not load metrics from one or both directories")
        return
    
    # Load model info for student
    model_info_path = os.path.join(student_dir, 'model_info.txt')
    student_model_info = {}
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    student_model_info[key] = value
    
    comparison = compare_metrics(teacher_metrics, student_metrics)
    
    # Generate report
    lines = []
    lines.append("=" * 100)
    lines.append("OPTICAM MULTI: TEACHER (ResNet50) vs STUDENT (Compressed) COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    
    # Model info
    lines.append("-- Model Information --")
    lines.append(f"Teacher: ResNet50 (pretrained ImageNet1K_V1)")
    lines.append(f"  Directory: {teacher_dir}")
    lines.append("")
    lines.append(f"Student: {student_model_info.get('Architecture', 'Unknown')}")
    lines.append(f"  Directory: {student_dir}")
    lines.append(f"  Checkpoint: {student_model_info.get('Checkpoint', 'N/A')}")
    lines.append(f"  Teacher accuracy: {student_model_info.get('Teacher accuracy', 'N/A')}")
    lines.append(f"  Student accuracy: {student_model_info.get('Student accuracy', 'N/A')}")
    lines.append(f"  Agreement rate: {student_model_info.get('Agreement rate', 'N/A')}")
    lines.append(f"  Compression ratio: {student_model_info.get('Compression ratio', 'N/A')}")
    lines.append("")
    
    # Primary metrics comparison
    lines.append("=" * 100)
    lines.append("PRIMARY METRICS COMPARISON (Paper Equations 13-15)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Metric':<40} {'Teacher':<15} {'Student':<15} {'Diff':<15} {'% Change':<15}")
    lines.append("-" * 100)
    
    primary_keys = [
        'AD (Average Drop %, Eq 13)',
        'AI (Average Increase %, Eq 14)',
        'AG (Average Gain %, Eq 15)'
    ]
    
    for key in primary_keys:
        if key in comparison:
            c = comparison[key]
            lines.append(f"{key:<40} {c['teacher']:>14.6f} {c['student']:>14.6f} "
                        f"{c['diff']:>+14.6f} {c['diff_pct']:>+13.2f}%")
    
    lines.append("")
    
    # Advanced metrics
    lines.append("=" * 100)
    lines.append("ADVANCED METRICS COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Metric':<40} {'Teacher':<15} {'Student':<15} {'Diff':<15} {'% Change':<15}")
    lines.append("-" * 100)
    
    adv_keys = [
        'AUC Insertion',
        'AUC Deletion',
        'AOPC Insertion',
        'AOPC Deletion'
    ]
    
    for key in adv_keys:
        if key in comparison:
            c = comparison[key]
            lines.append(f"{key:<40} {c['teacher']:>14.6f} {c['student']:>14.6f} "
                        f"{c['diff']:>+14.6f} {c['diff_pct']:>+13.2f}%")
    
    lines.append("")
    
    # Consistency
    lines.append("=" * 100)
    lines.append("CONSISTENCY CONSTRAINT COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    
    if 'Consistency error |Σc_k - c|' in comparison:
        c = comparison['Consistency error |Σc_k - c|']
        lines.append(f"Consistency Error (lower is better):")
        lines.append(f"  Teacher: {c['teacher']:.6f}")
        lines.append(f"  Student: {c['student']:.6f}")
        lines.append(f"  Difference: {c['diff']:+.6f} ({c['diff_pct']:+.2f}%)")
    lines.append("")
    
    # Timing
    lines.append("=" * 100)
    lines.append("TIMING COMPARISON")
    lines.append("=" * 100)
    lines.append("")
    
    timing_keys = [
        'Global total runtime',
        'Avg saliency per used image'
    ]
    
    for key in timing_keys:
        if key in comparison:
            c = comparison[key]
            unit = 's'
            lines.append(f"{key}:")
            lines.append(f"  Teacher: {c['teacher']:.3f}{unit}")
            lines.append(f"  Student: {c['student']:.3f}{unit}")
            lines.append(f"  Speedup: {c['teacher']/c['student']:.2f}x" if c['student'] > 0 else "  Speedup: N/A")
            lines.append(f"  Difference: {c['diff']:+.3f}{unit} ({c['diff_pct']:+.2f}%)")
            lines.append("")
    
    # Summary
    lines.append("=" * 100)
    lines.append("SUMMARY & RECOMMENDATIONS")
    lines.append("=" * 100)
    lines.append("")
    
    # Calculate overall quality loss
    quality_metrics = ['AD (Average Drop %, Eq 13)', 'AI (Average Increase %, Eq 14)', 
                       'AG (Average Gain %, Eq 15)']
    avg_quality_change = np.mean([comparison[k]['diff_pct'] for k in quality_metrics if k in comparison])
    
    # Calculate speedup
    if 'Avg saliency per used image' in comparison:
        speedup = comparison['Avg saliency per used image']['teacher'] / comparison['Avg saliency per used image']['student']
    else:
        speedup = None
    
    lines.append(f"Quality Change (avg of AD/AI/AG): {avg_quality_change:+.2f}%")
    if speedup:
        lines.append(f"Speed Improvement: {speedup:.2f}x faster")
    lines.append("")
    
    # Interpretation
    lines.append("Interpretation:")
    if abs(avg_quality_change) < 5:
        lines.append("  ✓ Quality degradation is MINIMAL (<5%)")
        lines.append("    → Compressed model is suitable for thesis demo")
    elif abs(avg_quality_change) < 10:
        lines.append("  ⚠ Quality degradation is MODERATE (5-10%)")
        lines.append("    → Acceptable for speed-critical applications")
    else:
        lines.append("  ✗ Quality degradation is SIGNIFICANT (>10%)")
        lines.append("    → Recommend further tuning or use original ResNet50")
    
    lines.append("")
    if speedup and speedup > 2:
        lines.append(f"  ✓ Speed improvement is SIGNIFICANT ({speedup:.1f}x)")
        lines.append("    → Good trade-off for real-time applications")
    elif speedup:
        lines.append(f"  ⚠ Speed improvement is MODERATE ({speedup:.1f}x)")
        lines.append("    → Consider if quality loss is acceptable")
    
    lines.append("")
    lines.append("Recommendations:")
    lines.append("  1. For thesis presentation: Use original ResNet50 for best quality")
    lines.append("  2. For demo/inference speed: Use compressed model if quality loss < 5%")
    lines.append("  3. For production deployment: Consider training larger student (e.g., ResNet34)")
    lines.append("")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nComparison report generated: {output_path}")
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print(f"  Quality change: {avg_quality_change:+.2f}%")
    if speedup:
        print(f"  Speedup: {speedup:.2f}x")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Compare Teacher vs Student OptiCAM Multi results')
    parser.add_argument('--teacher_dir', type=str, required=True,
                        help='Directory with teacher (ResNet50) results')
    parser.add_argument('--student_dir', type=str, required=True,
                        help='Directory with student (compressed) results')
    parser.add_argument('--output', type=str, default='comparison_report.txt',
                        help='Output report file path')
    
    args = parser.parse_args()
    
    generate_report(args.teacher_dir, args.student_dir, args.output)

if __name__ == '__main__':
    main()
