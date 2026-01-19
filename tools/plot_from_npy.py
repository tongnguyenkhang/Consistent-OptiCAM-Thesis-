"""
OptiCAM Plot Generator - Unified plotting for Baseline and Multi configurations.

Auto-detects run type (Baseline vs Multi) and generates appropriate visualizations:
- Baseline: Single loss component (abs/mse)
- Multi: Multi-component analysis (total, fidelity, consistency, lambda, violation)

File Organization:
results/<run>/
├── metrics/
│   ├── metrics_summary.txt           # Consolidated statistics
│   ├── component_losses_analysis.png # Multi only: 5-panel overview
│   ├── <component>/                  # Per-component folders
│   │   ├── <component>_loss_batch.png       # Batch-level view
│   │   ├── <component>_loss_iteration.png   # Iteration-level detail
│   │   └── batch_*.npy               # Data files
└── plot/
    └── <image>_<Component>_Loss.png  # Per-image plots (if not skipped)
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Component keys for Multi OptiCAM
COMPONENT_KEYS = ('total', 'fidelity', 'consistency', 'lambda', 'violation')

# Baseline loss types
BASELINE_LOSSES = ('abs', 'mse')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate plots from OptiCAM results (auto-detects Baseline vs Multi)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Path to results/<run_name>')
    parser.add_argument('--skip_per_image', action='store_true',
                       help='Skip per-image plots (only generate aggregate views)')
    return parser.parse_args()

def detect_run_type(results_dir: str) -> str:
    """
    Auto-detect run type based on directory structure.
    
    Returns:
        'multi' if multi-component structure detected
        'baseline' if baseline structure detected
        'unknown' if cannot determine
    """
    metrics_dir = Path(results_dir) / 'metrics'
    
    # Check for multi-component folders
    has_components = any(
        (metrics_dir / comp).exists() 
        for comp in COMPONENT_KEYS
    )
    
    if has_components:
        return 'multi'
    
    # Check for baseline structure
    has_baseline = any(
        (metrics_dir / loss).exists() 
        for loss in BASELINE_LOSSES
    )
    
    if has_baseline:
        return 'baseline'
    
    # Check plot folder for hints
    plot_dir = Path(results_dir) / 'plot'
    if plot_dir.exists():
        npy_files = list(plot_dir.glob('*_Loss.npy'))
        if npy_files:
            # Check if multi-component naming
            first_file = npy_files[0].stem
            if any(comp.capitalize() in first_file for comp in COMPONENT_KEYS):
                return 'multi'
            return 'baseline'
    
    return 'unknown'



def load_multi_component_data(results_dir: str) -> Dict[str, np.ndarray]:
    """Load all component histories for Multi OptiCAM."""
    metrics_dir = Path(results_dir) / 'metrics'
    data = {}
    
    for comp in COMPONENT_KEYS:
        comp_dir = metrics_dir / comp
        batch_file = comp_dir / f'batch_histories_{comp}.npy'
        
        if batch_file.exists():
            try:
                batch_histories = np.load(batch_file, allow_pickle=True)
                # Flatten all batch histories into single array
                all_steps = np.concatenate([np.array(h) for h in batch_histories])
                data[comp] = {
                    'all_steps': all_steps,
                    'batch_histories': batch_histories,
                    'num_batches': len(batch_histories)
                }
                print(f"  [Component] Loaded {comp}: {len(batch_histories)} batches, "
                      f"{len(all_steps)} total steps")
            except Exception as e:
                print(f"  [Error] Failed to load {comp}: {e}")
    
    return data

def load_baseline_component_data(results_dir: str) -> Dict[str, np.ndarray]:
    """Load loss histories for Baseline OptiCAM (abs or mse)."""
    metrics_dir = Path(results_dir) / 'metrics'
    data = {}
    
    # Try to find loss type (abs or mse)
    loss_type = None
    for lt in BASELINE_LOSSES:
        loss_dir = metrics_dir / lt
        if loss_dir.exists():
            loss_type = lt
            break
    
    if not loss_type:
        print(f"  [Error] No baseline loss folder found in {metrics_dir}")
        return data
    
    loss_dir = metrics_dir / loss_type
    batch_file = loss_dir / f'batch_internal_loss_histories_{loss_type}.npy'
    
    if batch_file.exists():
        try:
            batch_histories = np.load(batch_file, allow_pickle=True)
            # Flatten all batch histories into single array
            all_steps = np.concatenate([np.array(h) for h in batch_histories])
            data[loss_type] = {
                'all_steps': all_steps,
                'batch_histories': batch_histories,
                'num_batches': len(batch_histories)
            }
            print(f"  [Baseline] Loaded {loss_type}: {len(batch_histories)} batches, "
                  f"{len(all_steps)} total steps")
        except Exception as e:
            print(f"  [Error] Failed to load {loss_type}: {e}")
    else:
        print(f"  [Error] Batch histories not found: {batch_file}")
    
    return data

def plot_dual_view_batch_level(
    comp_key: str,
    batch_histories: List[np.ndarray],
    output_path: Path,
    title: str,
    color: str
):
    """
    Dual-View Batch-Level Plot (View 1).
    
    Shows:
    - Final loss value for each batch
    - Box plot showing distribution
    - Moving average trend
    """
    final_losses = np.array([h[-1] if len(h) > 0 else np.nan for h in batch_histories])
    batch_indices = np.arange(len(final_losses))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Scatter plot of final losses
    ax.scatter(batch_indices, final_losses, s=30, alpha=0.6, color=color, 
              label='Final loss per batch', zorder=3)
    
    # Moving average
    window = max(3, len(final_losses) // 20)
    if window < len(final_losses):
        kernel = np.ones(window) / window
        ma = np.convolve(final_losses, kernel, mode='valid')
        ma_x = np.arange(window//2, window//2 + len(ma))
        ax.plot(ma_x, ma, color='darkred', linewidth=2.5, 
               label=f'{window}-batch moving average', zorder=4)
    
    # Box plot overlay (every 10 batches)
    if len(final_losses) > 10:
        box_positions = []
        box_data = []
        bin_size = max(1, len(final_losses) // 10)
        for i in range(0, len(final_losses), bin_size):
            end_idx = min(i + bin_size, len(final_losses))
            bin_data = final_losses[i:end_idx]
            bin_data = bin_data[~np.isnan(bin_data)]
            if len(bin_data) > 0:
                box_positions.append(i + bin_size // 2)
                box_data.append(bin_data)
        
        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=bin_size*0.6,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color, alpha=0.3),
                           medianprops=dict(color='black', linewidth=2),
                           whiskerprops=dict(alpha=0.5),
                           capprops=dict(alpha=0.5))
    
    ax.set_xlabel('Batch Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title}', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - Batch-Level View', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Statistics box
    mean_val = np.nanmean(final_losses)
    std_val = np.nanstd(final_losses)
    min_val = np.nanmin(final_losses)
    max_val = np.nanmax(final_losses)
    
    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\n'
    stats_text += f'Range: [{min_val:.4f}, {max_val:.4f}]'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved batch-level: {output_path}")

def plot_dual_view_iteration_level(
    comp_key: str,
    batch_histories: List[np.ndarray],
    output_path: Path,
    title: str,
    color: str,
    sample_batches: int = 8
):
    """
    Dual-View Iteration-Level Plot (View 2).
    
    Shows convergence within batches by plotting a sample of representative batches.
    """
    num_batches = len(batch_histories)
    
    # Select representative batches (evenly spaced + edge cases)
    if num_batches <= sample_batches:
        selected_indices = list(range(num_batches))
    else:
        # Include first, last, and evenly spaced middle batches
        step = (num_batches - 2) // (sample_batches - 2)
        selected_indices = [0] + list(range(step, num_batches-1, step)) + [num_batches-1]
        selected_indices = selected_indices[:sample_batches]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color gradient for batches
    cmap = plt.colormaps['viridis']
    colors = [cmap(i / len(selected_indices)) for i in range(len(selected_indices))]
    
    for idx, batch_idx in enumerate(selected_indices):
        history = batch_histories[batch_idx]
        iterations = np.arange(len(history))
        ax.plot(iterations, history, linewidth=2, alpha=0.7, 
               color=colors[idx], label=f'Batch {batch_idx}')
    
    ax.set_xlabel('Iteration (within batch)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title}', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - Iteration-Level Convergence (Sample: {len(selected_indices)} batches)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    # Add convergence metrics
    mean_initial = np.mean([h[0] for h in batch_histories if len(h) > 0])
    mean_final = np.mean([h[-1] for h in batch_histories if len(h) > 0])
    reduction = ((mean_initial - mean_final) / mean_initial * 100) if mean_initial > 0 else 0
    
    conv_text = f'Avg Reduction: {reduction:.1f}%\n'
    conv_text += f'Initial: {mean_initial:.4f} → Final: {mean_final:.4f}'
    
    ax.text(0.02, 0.98, conv_text,
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
           fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved iteration-level: {output_path}")

def plot_multi_comprehensive(data: Dict, results_dir: str):
    """
    Generate 5-panel comprehensive analysis for Multi OptiCAM.
    Uses batch-level final losses with adaptive grouping for scalability.
    """
    
    # Helper: Extract final loss per batch
    def get_batch_finals(batch_histories: List[np.ndarray]) -> np.ndarray:
        """Get final loss value from each batch."""
        return np.array([h[-1] if len(h) > 0 else np.nan for h in batch_histories])
    
    # Helper: Group batches for better visualization
    def group_batches(values: np.ndarray, target_groups: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Group batch values for cleaner visualization.
        
        Returns:
            group_centers: X positions for plotting
            group_means: Mean value per group
            group_stds: Std deviation per group (for error bars if needed)
        """
        n_batches = len(values)
        if n_batches <= target_groups:
            # No grouping needed
            return np.arange(n_batches), values, np.zeros(n_batches)
        
        # Calculate group size to get approximately target_groups
        group_size = max(1, n_batches // target_groups)
        n_groups = (n_batches + group_size - 1) // group_size
        
        group_centers = []
        group_means = []
        group_stds = []
        
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, n_batches)
            group_data = values[start_idx:end_idx]
            group_data = group_data[~np.isnan(group_data)]  # Remove NaNs
            
            if len(group_data) > 0:
                group_centers.append((start_idx + end_idx - 1) / 2)
                group_means.append(np.mean(group_data))
                group_stds.append(np.std(group_data))
        
        return np.array(group_centers), np.array(group_means), np.array(group_stds)
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                          top=0.94, bottom=0.05, left=0.06, right=0.96)
    
    # Determine if grouping is needed
    n_batches = data['total']['num_batches'] if 'total' in data else 0
    use_grouping = n_batches > 150  # Group if more than 150 batches
    group_note = f" (grouped into ~100 groups from {n_batches} batches)" if use_grouping else ""
    
    # 1. Total Loss Evolution (batch-level with grouping)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'total' in data:
        batch_finals = get_batch_finals(data['total']['batch_histories'])
        
        if use_grouping:
            x_vals, y_vals, y_stds = group_batches(batch_finals)
            ax1.plot(x_vals, y_vals, linewidth=2, color='#2E86AB', alpha=0.8)
            ax1.fill_between(x_vals, y_vals - y_stds, y_vals + y_stds, 
                            color='#2E86AB', alpha=0.2, label='Std deviation')
        else:
            x_vals = np.arange(len(batch_finals))
            ax1.plot(x_vals, batch_finals, linewidth=2, marker='o', markersize=3,
                    color='#2E86AB', alpha=0.7)
        
        ax1.set_xlabel('Batch Index', fontsize=12)
        ax1.set_ylabel('Final Total Loss', fontsize=12)
        ax1.set_title(f'Total Loss per Batch (Fidelity + lambda x Consistency){group_note}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if use_grouping:
            ax1.legend(loc='best', fontsize=9)
        
        mean_val = np.nanmean(batch_finals)
        final_val = batch_finals[-1]
        initial_val = batch_finals[0]
        reduction = ((initial_val - final_val) / initial_val * 100) if initial_val > 0 else 0
        
        ax1.text(0.02, 0.98, 
                f'First batch: {initial_val:.3f}\nLast batch: {final_val:.3f}\n'
                f'Mean: {mean_val:.3f}\nReduction: {reduction:.1f}%',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6), 
                fontsize=10, family='monospace')
    
    # 2. Fidelity vs Consistency (Dual-Axis with grouping)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'fidelity' in data and 'consistency' in data:
        ax2_right = ax2.twinx()
        
        fid_finals = get_batch_finals(data['fidelity']['batch_histories'])
        cons_finals = get_batch_finals(data['consistency']['batch_histories'])
        
        if use_grouping:
            x_fid, y_fid, std_fid = group_batches(fid_finals)
            x_cons, y_cons, std_cons = group_batches(cons_finals)
            
            line1 = ax2.plot(x_fid, y_fid, linewidth=2, color='#A23B72', 
                            alpha=0.8, label='Fidelity (logit MSE)')
            ax2.fill_between(x_fid, y_fid - std_fid, y_fid + std_fid,
                            color='#A23B72', alpha=0.2)
            
            line2 = ax2_right.plot(x_cons, y_cons, linewidth=2, color='#F18F01',
                                  alpha=0.8, label='Consistency (prob MSE)')
            ax2_right.fill_between(x_cons, y_cons - std_cons, y_cons + std_cons,
                                  color='#F18F01', alpha=0.2)
        else:
            batch_indices = np.arange(len(fid_finals))
            line1 = ax2.plot(batch_indices, fid_finals, linewidth=2, marker='o', markersize=3,
                            color='#A23B72', alpha=0.7, label='Fidelity (logit MSE)')
            line2 = ax2_right.plot(batch_indices, cons_finals, linewidth=2, marker='s', markersize=3,
                                  color='#F18F01', alpha=0.7, label='Consistency (prob MSE)')
        
        ax2.set_xlabel('Batch Index', fontsize=12)
        ax2.set_ylabel('Fidelity Loss', fontsize=12, color='#A23B72')
        ax2.tick_params(axis='y', labelcolor='#A23B72')
        ax2.set_ylim(bottom=0)
        
        ax2_right.set_ylabel('Consistency Loss', fontsize=12, color='#F18F01')
        ax2_right.tick_params(axis='y', labelcolor='#F18F01')
        
        ax2.set_title('Component Breakdown per Batch (Dual-Axis)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right', fontsize=10)
        
        fid_mean = np.nanmean(fid_finals)
        cons_mean = np.nanmean(cons_finals)
        scale_ratio = cons_mean / fid_mean if fid_mean > 0 else float('inf')
        
        ax2.text(0.02, 0.98, 
                f'Fidelity mean: {fid_mean:.6f}\n'
                f'Consistency mean: {cons_mean:.3f}\n'
                f'Scale ratio: {scale_ratio:.0f}x',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6), 
                fontsize=10, family='monospace')
    
    # 3. Lambda Schedule (with grouping)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'lambda' in data:
        batch_histories = data['lambda']['batch_histories']
        lambda_per_batch = np.array([np.mean(h) if len(h) > 0 else np.nan for h in batch_histories])
        
        if use_grouping:
            x_vals, y_vals, y_stds = group_batches(lambda_per_batch)
            ax3.plot(x_vals, y_vals, linewidth=2.5, color='#06A77D', alpha=0.8)
            ax3.fill_between(x_vals, y_vals - y_stds, y_vals + y_stds,
                            color='#06A77D', alpha=0.2)
        else:
            batch_indices = np.arange(len(lambda_per_batch))
            ax3.plot(batch_indices, lambda_per_batch, linewidth=2.5, marker='d', markersize=4,
                    color='#06A77D', alpha=0.7)
        
        ax3.set_xlabel('Batch Index', fontsize=12)
        ax3.set_ylabel('Lambda (average per batch)', fontsize=12)
        ax3.set_title('Adaptive Lambda Schedule', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        lambda_start = lambda_per_batch[0]
        lambda_end = lambda_per_batch[-1]
        ax3.axhline(lambda_start, color='red', linestyle='--', alpha=0.5, 
                   linewidth=2, label=f'Start: {lambda_start:.3f}')
        ax3.axhline(lambda_end, color='blue', linestyle='--', alpha=0.5, 
                   linewidth=2, label=f'End: {lambda_end:.3f}')
        ax3.legend(loc='best', fontsize=10)
        
        lambda_range = lambda_end - lambda_start
        ax3.text(0.98, 0.02, 
                f'Range: {lambda_range:.3f}\n'
                f'Schedule: {"Linear" if abs(lambda_range) > 0.01 else "Fixed"}',
                transform=ax3.transAxes, 
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6), 
                fontsize=10, family='monospace')
    
    # 4. Constraint Violation (with grouping)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'violation' in data:
        viol_finals = get_batch_finals(data['violation']['batch_histories'])
        
        if use_grouping:
            x_vals, y_vals, y_stds = group_batches(viol_finals)
            ax4.plot(x_vals, y_vals, linewidth=2, color='#C73E1D', alpha=0.8)
            ax4.fill_between(x_vals, y_vals - y_stds, y_vals + y_stds,
                            color='#C73E1D', alpha=0.2)
        else:
            batch_indices = np.arange(len(viol_finals))
            ax4.plot(batch_indices, viol_finals, linewidth=2, marker='x', markersize=4,
                    color='#C73E1D', alpha=0.7)
        
        ax4.set_xlabel('Batch Index', fontsize=12)
        ax4.set_ylabel('|Sum p_k - p_orig|', fontsize=12)
        ax4.set_title('Consistency Constraint Violation per Batch', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Target thresholds
        ax4.axhline(0.1, color='green', linestyle='--', alpha=0.6, 
                   linewidth=2, label='Good: < 0.1')
        ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.6, 
                   linewidth=2, label='Poor: > 1.0')
        ax4.legend(loc='best', fontsize=10)
        
        mean_viol = np.nanmean(viol_finals)
        final_viol = viol_finals[-1]
        initial_viol = viol_finals[0]
        
        status = "GOOD" if final_viol < 0.1 else ("OK" if final_viol < 1.0 else "POOR")
        color_map = {"GOOD": "lightgreen", "OK": "yellow", "POOR": "lightcoral"}
        
        ax4.text(0.02, 0.98, 
                f'First: {initial_viol:.4f}\n'
                f'Last: {final_viol:.4f}\n'
                f'Mean: {mean_viol:.4f}\n'
                f'Status: {status}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color_map[status], alpha=0.7), 
                fontsize=10, family='monospace', fontweight='bold')
    
    # 5. Loss Component Balance (with grouping)
    ax5 = fig.add_subplot(gs[2, :])
    if 'fidelity' in data and 'consistency' in data and 'lambda' in data:
        fid_finals = get_batch_finals(data['fidelity']['batch_histories'])
        cons_finals = get_batch_finals(data['consistency']['batch_histories'])
        lambda_finals = get_batch_finals(data['lambda']['batch_histories'])
        
        lambda_weighted_cons = lambda_finals * cons_finals
        total_finals = fid_finals + lambda_weighted_cons
        
        if use_grouping:
            x_fid, y_fid, std_fid = group_batches(fid_finals)
            x_cons, y_cons, std_cons = group_batches(lambda_weighted_cons)
            x_tot, y_tot, std_tot = group_batches(total_finals)
            
            ax5.plot(x_fid, y_fid, linewidth=2, alpha=0.8, color='#A23B72', label='Fidelity Loss')
            ax5.plot(x_cons, y_cons, linewidth=2, alpha=0.8, color='#F18F01', label='lambda x Consistency')
            ax5.plot(x_tot, y_tot, linewidth=2.5, alpha=0.9, color='#2E86AB', linestyle='--', label='Total (sum)')
        else:
            batch_indices = np.arange(len(fid_finals))
            ax5.plot(batch_indices, fid_finals, linewidth=2, marker='o', markersize=3,
                    alpha=0.7, color='#A23B72', label='Fidelity Loss')
            ax5.plot(batch_indices, lambda_weighted_cons, linewidth=2, marker='s', markersize=3,
                    alpha=0.7, color='#F18F01', label='lambda x Consistency')
            ax5.plot(batch_indices, total_finals, linewidth=2.5, marker='^', markersize=3,
                    alpha=0.9, color='#2E86AB', linestyle='--', label='Total (sum)')
        
        ax5.set_xlabel('Batch Index', fontsize=12)
        ax5.set_ylabel('Loss Value', fontsize=12)
        ax5.set_title('Loss Component Contributions per Batch', 
                     fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best', fontsize=11)
        ax5.set_yscale('log')
        
        # Compute percentage contributions
        fid_contrib = np.nanmean(fid_finals / (total_finals + 1e-8)) * 100
        cons_contrib = np.nanmean(lambda_weighted_cons / (total_finals + 1e-8)) * 100
        
        ax5.text(0.98, 0.02, 
                f'Avg Contribution:\n'
                f'Fidelity: {fid_contrib:.1f}%\n'
                f'lambda x Consistency: {cons_contrib:.1f}%',
                transform=ax5.transAxes, 
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), 
                fontsize=10, family='monospace')
    
    # Main title
    run_name = os.path.basename(os.path.normpath(results_dir))
    fig.suptitle(f'{run_name} - Multi-Component Loss Analysis', 
                fontsize=18, fontweight='bold')
    
    # Save to metrics folder (not plot folder)
    output_path = Path(results_dir) / 'metrics' / 'component_losses_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  [Comprehensive] Saved 5-panel analysis: {output_path}")
    plt.close()

def plot_multi_components(data: Dict, results_dir: str):
    """Generate dual-view plots for each component."""
    
    configs = [
        ('total', 'Total Loss', '#2E86AB'),
        ('fidelity', 'Fidelity Loss (Logit MSE)', '#A23B72'),
        ('consistency', 'Consistency Loss (Prob MSE)', '#F18F01'),
        ('lambda', 'Lambda (λ) Schedule', '#06A77D'),
        ('violation', 'Constraint Violation |Σp_k - p_orig|', '#C73E1D')
    ]
    
    metrics_dir = Path(results_dir) / 'metrics'
    
    for comp_key, title, color in configs:
        if comp_key not in data:
            print(f"  [Warning] Component '{comp_key}' not found in data")
            continue
        
        comp_dir = metrics_dir / comp_key
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        batch_histories = data[comp_key]['batch_histories']
        
        # View 1: Batch-level aggregation
        batch_plot_path = comp_dir / f'{comp_key}_loss_batch.png'
        plot_dual_view_batch_level(
            comp_key, batch_histories, batch_plot_path, title, color
        )
        
        # View 2: Iteration-level detail
        iter_plot_path = comp_dir / f'{comp_key}_loss_iteration.png'
        plot_dual_view_iteration_level(
            comp_key, batch_histories, iter_plot_path, title, color
        )

def plot_baseline_component(data: Dict, results_dir: str):
    """Generate dual-view plots for Baseline OptiCAM (single loss component)."""
    
    # Determine loss type
    loss_type = None
    for lt in BASELINE_LOSSES:
        if lt in data:
            loss_type = lt
            break
    
    if not loss_type:
        print(f"  [Error] No baseline data found")
        return
    
    # Title and color based on loss type
    if loss_type == 'mse':
        title = 'MSE Loss (Probability Space)'
        color = '#A23B72'
    else:  # abs
        title = 'Absolute Loss (Logit Space)'
        color = '#2E86AB'
    
    metrics_dir = Path(results_dir) / 'metrics' / loss_type
    batch_histories = data[loss_type]['batch_histories']
    
    # View 1: Batch-level aggregation
    batch_plot_path = metrics_dir / f'{loss_type}_loss_batch.png'
    plot_dual_view_batch_level(loss_type, batch_histories, batch_plot_path, title, color)
    
    # View 2: Iteration-level detail
    iter_plot_path = metrics_dir / f'{loss_type}_loss_iteration.png'
    plot_dual_view_iteration_level(loss_type, batch_histories, iter_plot_path, title, color)
    
    print(f"  [Baseline] Generated dual-view plots for {loss_type}")

def plot_per_image_losses(results_dir: str, run_type: str):
    """Generate per-image loss plots."""
    plot_dir = Path(results_dir) / 'plot'
    
    if not plot_dir.exists():
        print(f"  [Warning] Plot directory not found: {plot_dir}")
        return
    
    if run_type == 'multi':
        # Multi: Plot all 5 components per image (files named *_Total_Loss.npy, etc.)
        npy_files = sorted(plot_dir.glob('*_Total_Loss.npy'))
        print(f"\n  [Per-Image] Found {len(npy_files)} images")
        
        plot_count = 0
        for total_npy in npy_files:
            base_name = total_npy.stem.replace('_Total_Loss', '')
            
            for comp in COMPONENT_KEYS:
                comp_npy = plot_dir / f'{base_name}_{comp.capitalize()}_Loss.npy'
                if not comp_npy.exists():
                    continue
                
                try:
                    arr = np.load(comp_npy)
                    if arr.size == 0:
                        continue
                    
                    out_png = comp_npy.with_suffix('.png')
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(range(1, len(arr)+1), arr, linewidth=2, 
                           marker='o', markersize=4, alpha=0.7, color='#2E86AB')
                    ax.set_xlabel('Iteration', fontsize=11)
                    ax.set_ylabel(f'{comp.capitalize()} Loss', fontsize=11)
                    ax.set_title(f'{base_name} - {comp.capitalize()}', 
                                fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    mean_val = np.mean(arr)
                    final_val = arr[-1]
                    ax.text(0.98, 0.98, f'Mean: {mean_val:.4f}\nFinal: {final_val:.4f}',
                           transform=ax.transAxes, 
                           horizontalalignment='right', verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                           fontsize=9, family='monospace')
                    
                    plt.tight_layout()
                    plt.savefig(out_png, dpi=150)
                    plt.close()
                    plot_count += 1
                    
                except Exception as e:
                    print(f"    [Error] Failed to plot {comp_npy}: {e}")
        
        print(f"  [Per-Image] Generated {plot_count} plots")
    
    elif run_type == 'baseline':
        # Baseline: Single loss per image (files named *_Loss_mse.npy or *_Loss_abs.npy)
        # Find which loss type exists
        npy_files = []
        loss_type = None
        
        for lt in BASELINE_LOSSES:
            pattern = f'*_Loss_{lt}.npy'
            files = list(plot_dir.glob(pattern))
            if files:
                npy_files = sorted(files)
                loss_type = lt
                break
        
        if not npy_files:
            print(f"  [Warning] No baseline .npy files found in {plot_dir}")
            return
        
        print(f"\n  [Per-Image] Found {len(npy_files)} images (loss type: {loss_type})")
        
        plot_count = 0
        for npy in npy_files:
            try:
                arr = np.load(npy)
                if arr.size == 0:
                    continue
                
                out_png = npy.with_suffix('.png')
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(1, len(arr)+1), arr, linewidth=2, 
                       marker='o', markersize=4, alpha=0.7, color='#A23B72')
                ax.set_xlabel('Iteration', fontsize=11)
                ax.set_ylabel(f'{loss_type.upper()} Loss', fontsize=11)
                
                # Extract image name (remove _Loss_mse or _Loss_abs suffix)
                img_name = npy.stem.replace(f'_Loss_{loss_type}', '')
                ax.set_title(f'{img_name}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                mean_val = np.mean(arr)
                final_val = arr[-1]
                ax.text(0.98, 0.98, f'Mean: {mean_val:.4f}\nFinal: {final_val:.4f}',
                       transform=ax.transAxes, 
                       horizontalalignment='right', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
                       fontsize=9, family='monospace')
                
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()
                plot_count += 1
                
            except Exception as e:
                print(f"    [Error] Failed to plot {npy}: {e}")
        
        print(f"  [Per-Image] Generated {plot_count} plots")

def main():
    args = parse_args()
    results_dir = args.results_dir
    
    print(f"\n{'='*80}")
    print(f"OptiCAM Plot Generator")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    
    # Auto-detect run type
    run_type = detect_run_type(results_dir)
    print(f"Detected run type: {run_type.upper()}")
    
    if run_type == 'unknown':
        print("\n[ERROR] Cannot determine run type. Check directory structure.")
        return
    
    # Generate plots based on run type
    print(f"\n{'='*80}")
    print("Step 1: Generating Aggregate Plots")
    print(f"{'='*80}")
    
    if run_type == 'multi':
        print("[Multi] Loading component data...")
        data = load_multi_component_data(results_dir)
        
        if len(data) == 0:
            print("[ERROR] No component data found!")
            return
        
        print(f"[Multi] Loaded {len(data)} components")
        
        # Comprehensive 5-panel plot
        print("\n[Multi] Generating comprehensive analysis...")
        plot_multi_comprehensive(data, results_dir)
        
        # Individual component dual-view plots
        print("\n[Multi] Generating component dual-view plots...")
        plot_multi_components(data, results_dir)
    
    elif run_type == 'baseline':
        print("[Baseline] Loading baseline data...")
        data = load_baseline_component_data(results_dir)
        
        if len(data) == 0:
            print("[ERROR] No baseline data found!")
            return
        
        print(f"[Baseline] Loaded {len(data)} loss type(s)")
        
        # Generate dual-view plots
        print("\n[Baseline] Generating dual-view plots...")
        plot_baseline_component(data, results_dir)
    
    # Per-image plots
    if not args.skip_per_image:
        print(f"\n{'='*80}")
        print("Step 2: Generating Per-Image Plots")
        print(f"{'='*80}")
        plot_per_image_losses(results_dir, run_type)
    else:
        print(f"\n[Skip] Per-image plots skipped (--skip_per_image flag)")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] All plotting completed!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
