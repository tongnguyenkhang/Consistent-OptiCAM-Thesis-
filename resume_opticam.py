"""
Helper script to resume generate_opticam.py from checkpoint.
Usage:
    python resume_opticam.py --checkpoint_dir ./results/OptiCAM_Iter500_Lr1e-6/checkpoints
"""

import os
import sys
import argparse
import glob

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in directory."""
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find all checkpoint files
    pattern = os.path.join(checkpoint_dir, 'checkpoint_batch_*.pt')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"Error: No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort by batch number (extract from filename)
    def get_batch_num(path):
        basename = os.path.basename(path)
        # Extract number from "checkpoint_batch_123.pt"
        num_str = basename.replace('checkpoint_batch_', '').replace('.pt', '')
        try:
            return int(num_str)
        except:
            return -1
    
    checkpoints.sort(key=get_batch_num)
    latest = checkpoints[-1]
    batch_num = get_batch_num(latest)
    
    print(f"Found latest checkpoint: {latest} (batch {batch_num})")
    return latest

def main():
    parser = argparse.ArgumentParser(description='Resume OptiCAM from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Specific checkpoint file to resume from (overrides auto-detection)')
    
    # Forward all other args to generate_opticam.py
    args, unknown = parser.parse_known_args()
    
    # Find checkpoint
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            sys.exit(1)
    
    # Load checkpoint to get config
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        batch_idx = ckpt.get('batch_idx', 0)
        flags = ckpt.get('flags', {})
        
        print(f"\nCheckpoint info:")
        print(f"  Batch index: {batch_idx}")
        print(f"  Processed samples: {ckpt.get('counted_samples', 0)}")
        print(f"  Config: {flags}")
        print()
        
    except Exception as e:
        print(f"Warning: Could not load checkpoint metadata: {e}")
        flags = {}
    
    # Build command to resume
    cmd_parts = [
        sys.executable,  # python
        'generate_opticam.py',
        f'--resume_checkpoint "{checkpoint_path}"'
    ]
    
    # Add flags from checkpoint if available
    if flags:
        if 'max_iter' in flags:
            cmd_parts.append(f'--max_iter {flags["max_iter"]}')
        if 'learning_rate' in flags:
            cmd_parts.append(f'--learning_rate {flags["learning_rate"]}')
        if 'objective' in flags:
            cmd_parts.append(f'--objective {flags["objective"]}')
        if 'batch_size' in flags:
            cmd_parts.append(f'--batch_size {flags["batch_size"]}')
        if 'name_path' in flags:
            cmd_parts.append(f'--name_path {flags["name_path"]}')
    
    # Add any extra args passed by user
    if unknown:
        cmd_parts.extend(unknown)
    
    cmd = ' '.join(cmd_parts)
    
    print("Command to resume:")
    print(cmd)
    print()
    
    # Ask for confirmation
    response = input("Execute this command? [y/N]: ")
    if response.lower() == 'y':
        os.system(cmd)
    else:
        print("Aborted. You can manually run the command above.")

if __name__ == '__main__':
    main()
