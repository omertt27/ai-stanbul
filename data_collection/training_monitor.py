"""
Training Progress Monitor for Istanbul Tourism Model
Week 5-8 Implementation: Real-time Training Tracking
"""

import os
import time
import json
import psutil
import torch
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMonitor:
    """Monitor and track training progress"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.log_file = "training_progress.json"
        self.progress_data = []
        
    def check_system_resources(self):
        """Monitor system resources during training"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                gpu_reserved = torch.cuda.memory_reserved(i)
                
                gpu_info[f'gpu_{i}'] = {
                    'total_memory_gb': gpu_memory / (1024**3),
                    'allocated_gb': gpu_allocated / (1024**3),
                    'reserved_gb': gpu_reserved / (1024**3),
                    'utilization_percent': (gpu_allocated / gpu_memory) * 100
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'gpu_info': gpu_info
        }
    
    def check_training_files(self):
        """Check for training output files and progress"""
        training_dir = Path("models/istanbul_tourism_model")
        checkpoint_dir = training_dir / "checkpoints"
        
        files_status = {
            'model_dir_exists': training_dir.exists(),
            'checkpoint_dir_exists': checkpoint_dir.exists(),
            'checkpoint_count': 0,
            'latest_checkpoint': None,
            'final_model_exists': (training_dir / "pytorch_model.bin").exists()
        }
        
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            files_status['checkpoint_count'] = len(checkpoints)
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                files_status['latest_checkpoint'] = str(latest)
        
        return files_status
    
    def estimate_completion_time(self, current_step=None, total_steps=None):
        """Estimate training completion time"""
        elapsed = datetime.now() - self.start_time
        
        if current_step and total_steps and current_step > 0:
            progress_ratio = current_step / total_steps
            total_estimated = elapsed / progress_ratio
            remaining = total_estimated - elapsed
            completion_time = datetime.now() + remaining
        else:
            # Default estimation based on model size and data
            estimated_total = timedelta(minutes=30)  # Conservative estimate
            remaining = estimated_total - elapsed
            completion_time = datetime.now() + remaining
        
        return {
            'elapsed_time': str(elapsed),
            'estimated_remaining': str(remaining) if 'remaining' in locals() else "Calculating...",
            'estimated_completion': completion_time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def log_progress(self):
        """Log current training progress"""
        progress_entry = {
            **self.check_system_resources(),
            **self.check_training_files(),
            **self.estimate_completion_time()
        }
        
        self.progress_data.append(progress_entry)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2, default=str)
        
        return progress_entry
    
    def display_progress(self):
        """Display current progress in terminal"""
        progress = self.log_progress()
        
        print(f"\n{'='*50}")
        print(f"🎓 TRAINING PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        print(f"⏱️  Elapsed Time: {progress['elapsed_time']}")
        print(f"⏳ Estimated Remaining: {progress['estimated_remaining']}")
        print(f"🎯 Est. Completion: {progress['estimated_completion']}")
        
        print(f"\n💻 System Resources:")
        print(f"   CPU Usage: {progress['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {progress['memory_percent']:.1f}%")
        print(f"   Memory Available: {progress['memory_available_gb']:.1f} GB")
        
        if progress['gpu_info']:
            print(f"\n🖥️  GPU Status:")
            for gpu_id, gpu_data in progress['gpu_info'].items():
                print(f"   {gpu_id.upper()}: {gpu_data['utilization_percent']:.1f}% utilized")
                print(f"   Memory: {gpu_data['allocated_gb']:.2f}/{gpu_data['total_memory_gb']:.1f} GB")
        
        print(f"\n📁 Training Files:")
        print(f"   Model Dir: {'✅' if progress['model_dir_exists'] else '❌'}")
        print(f"   Checkpoints: {progress['checkpoint_count']}")
        print(f"   Final Model: {'✅' if progress['final_model_exists'] else '⏳'}")
        
        if progress['latest_checkpoint']:
            print(f"   Latest: {Path(progress['latest_checkpoint']).name}")
        
        print(f"{'='*50}")
    
    def monitor_training(self, interval=30, max_duration=3600):
        """Monitor training for specified duration"""
        print(f"🔍 Starting training monitoring...")
        print(f"   Update interval: {interval} seconds")
        print(f"   Max duration: {max_duration/60:.0f} minutes")
        
        start_monitoring = datetime.now()
        
        while (datetime.now() - start_monitoring).seconds < max_duration:
            self.display_progress()
            
            # Check if training is complete
            progress = self.progress_data[-1] if self.progress_data else {}
            if progress.get('final_model_exists', False):
                print("\n🎉 Training appears to be complete!")
                break
                
            time.sleep(interval)
        
        print("\n📊 Monitoring session complete.")
        return self.progress_data

def main():
    """Main monitoring function"""
    monitor = TrainingMonitor()
    
    print("🔍 Training Progress Monitor Started")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        monitor.monitor_training(interval=60, max_duration=7200)  # Monitor for up to 2 hours
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    print(f"\n📈 Progress data saved to: {monitor.log_file}")

if __name__ == "__main__":
    main()
