#!/usr/bin/env python3
"""
Real-time training monitor - run this while training is ongoing
"""

import os
import time
import sys
from datetime import datetime, timedelta

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def parse_progress_line(line):
    """Extract progress percentage from training log line"""
    try:
        if '%|' in line and '/' in line:
            parts = line.split('|')[0].strip()
            pct = parts.replace('%', '').strip()
            return float(pct)
    except:
        pass
    return None

def monitor_training():
    """Monitor training progress in real-time"""
    log_file = 'phase2_extended_training_v2.log'
    results_file = 'phase2_extended_v2_results.json'
    
    start_time = datetime.now()
    last_progress = 0
    
    clear_screen()
    print("=" * 80)
    print("🔥 Phase 2 Extended Training - Live Monitor")
    print("=" * 80)
    print()
    print("Press Ctrl+C to exit (training will continue)")
    print()
    
    try:
        while True:
            clear_screen()
            print("=" * 80)
            print("🔥 Phase 2 Extended Training - Live Monitor")
            print("=" * 80)
            print()
            
            # Check if training is complete
            if os.path.exists(results_file):
                print("✅ TRAINING COMPLETE!")
                print()
                print(f"Results saved to: {results_file}")
                print()
                print("Run this to see results:")
                print("  python3 check_training_status.py")
                print()
                break
            
            # Read log file
            if not os.path.exists(log_file):
                print("⏳ Waiting for training to start...")
                print()
                time.sleep(2)
                continue
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find last progress line
            progress_lines = [l for l in lines if '%|' in l and 'it/s' in l]
            
            if progress_lines:
                last_line = progress_lines[-1]
                pct = parse_progress_line(last_line)
                
                if pct is not None:
                    last_progress = pct
                    elapsed = datetime.now() - start_time
                    
                    # Estimate remaining time
                    if pct > 0:
                        total_estimate = elapsed.total_seconds() / (pct / 100)
                        remaining = total_estimate - elapsed.total_seconds()
                        eta = datetime.now() + timedelta(seconds=remaining)
                    else:
                        remaining = 0
                        eta = None
                    
                    # Progress bar
                    bar_width = 50
                    filled = int(bar_width * pct / 100)
                    bar = '█' * filled + '░' * (bar_width - filled)
                    
                    print(f"📊 Progress: {pct:.1f}%")
                    print(f"[{bar}]")
                    print()
                    print(f"⏱️  Elapsed: {elapsed}")
                    if eta:
                        print(f"⏳ ETA: {eta.strftime('%H:%M:%S')} ({timedelta(seconds=int(remaining))} remaining)")
                    print()
                    print(f"💻 Last update:")
                    print(f"   {last_line.strip()}")
                    print()
            else:
                print("🔄 Training started, waiting for progress updates...")
                print()
            
            # Show last few log lines
            print("📝 Recent activity:")
            print("-" * 80)
            for line in lines[-10:]:
                if line.strip() and not line.startswith('['):
                    print(f"   {line.rstrip()}")
            print("-" * 80)
            print()
            
            print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            print()
            print("=" * 80)
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print()
        print()
        print("Monitor stopped. Training continues in background.")
        print()
        print("Check status with: python3 check_training_status.py")
        print()

if __name__ == '__main__':
    monitor_training()
