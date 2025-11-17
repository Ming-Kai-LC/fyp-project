#!/usr/bin/env python3
"""
GPU Memory Monitor for RTX 4060 8GB VRAM
Tracks and logs GPU memory usage during training

Author: Tan Ming Kai (24PMR12003)
FYP: CrossViT for COVID-19 Classification
"""

import torch
import time
from datetime import datetime


class GPUMemoryMonitor:
    """Monitor and log GPU memory usage"""
    
    def __init__(self, device_id=0):
        """
        Initialize GPU memory monitor
        
        Args:
            device_id: CUDA device ID (default: 0)
        """
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.logs = []
        
        if not torch.cuda.is_available():
            print("⚠️  WARNING: CUDA not available. Running on CPU.")
            return
        
        # Get GPU properties
        self.gpu_name = torch.cuda.get_device_name(device_id)
        self.total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        
        print(f"🖥️  GPU Monitor Initialized")
        print(f"   Device: {self.gpu_name}")
        print(f"   Total VRAM: {self.total_memory:.2f} GB")
        print(f"   Target: < 8.0 GB for RTX 4060")
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1e9
        reserved = torch.cuda.memory_reserved(self.device_id) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1e9
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'utilization_%': (allocated / self.total_memory) * 100
        }
    
    def log_memory(self, step_name="", verbose=True):
        """
        Log current memory usage
        
        Args:
            step_name: Name of current step (e.g., "Forward pass")
            verbose: Print to console if True
        """
        if not torch.cuda.is_available():
            return
        
        stats = self.get_memory_stats()
        stats['timestamp'] = datetime.now().isoformat()
        stats['step'] = step_name
        self.logs.append(stats)
        
        if verbose:
            print(f"[{step_name}] VRAM: {stats['allocated_gb']:.2f}/{self.total_memory:.2f} GB "
                  f"({stats['utilization_%']:.1f}% used)")
        
        # Warning if approaching limit
        if stats['allocated_gb'] > 7.0:  # 7GB threshold for 8GB GPU
            print(f"⚠️  WARNING: High memory usage! {stats['allocated_gb']:.2f} GB / 8.0 GB")
            print("   Consider: Reduce batch_size, enable gradient checkpointing, or clear cache")
    
    def clear_cache(self, verbose=True):
        """Clear CUDA cache"""
        if not torch.cuda.is_available():
            return
        
        before = self.get_memory_stats()['allocated_gb']
        torch.cuda.empty_cache()
        after = self.get_memory_stats()['allocated_gb']
        
        if verbose:
            print(f"🧹 Cache cleared: {before:.2f} GB → {after:.2f} GB (freed {before-after:.2f} GB)")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats(self.device_id)
        print("🔄 Peak memory stats reset")
    
    def print_summary(self):
        """Print summary of logged memory usage"""
        if not self.logs:
            print("No memory logs available")
            return
        
        if not torch.cuda.is_available():
            return
        
        print("\n" + "="*60)
        print("GPU MEMORY USAGE SUMMARY")
        print("="*60)
        
        # Current stats
        current = self.get_memory_stats()
        print(f"\nCurrent Status:")
        print(f"  Allocated:      {current['allocated_gb']:.2f} GB")
        print(f"  Reserved:       {current['reserved_gb']:.2f} GB")
        print(f"  Peak Allocated: {current['max_allocated_gb']:.2f} GB")
        print(f"  Utilization:    {current['utilization_%']:.1f}%")
        
        # Historical stats
        allocated_values = [log['allocated_gb'] for log in self.logs]
        print(f"\nHistorical Statistics:")
        print(f"  Average:        {sum(allocated_values)/len(allocated_values):.2f} GB")
        print(f"  Maximum:        {max(allocated_values):.2f} GB")
        print(f"  Minimum:        {min(allocated_values):.2f} GB")
        
        # Safety check
        if max(allocated_values) > 7.5:
            print("\n⚠️  WARNING: Peak memory usage exceeded 7.5 GB")
            print("   Risk of OOM errors on 8GB VRAM")
            print("   Recommendations:")
            print("   - Reduce batch_size")
            print("   - Enable gradient accumulation")
            print("   - Use gradient checkpointing")
        else:
            print(f"\n✅ Memory usage within safe limits for RTX 4060 8GB")
        
        print("="*60 + "\n")
    
    def save_logs(self, filepath='memory_logs.txt'):
        """Save memory logs to file"""
        if not self.logs:
            print("No memory logs to save")
            return
        
        with open(filepath, 'w') as f:
            f.write("GPU Memory Usage Logs\n")
            f.write(f"GPU: {self.gpu_name}\n")
            f.write(f"Total VRAM: {self.total_memory:.2f} GB\n")
            f.write("="*60 + "\n\n")
            
            for log in self.logs:
                f.write(f"[{log['timestamp']}] {log['step']}\n")
                f.write(f"  Allocated: {log['allocated_gb']:.3f} GB\n")
                f.write(f"  Reserved: {log['reserved_gb']:.3f} GB\n")
                f.write(f"  Utilization: {log['utilization_%']:.1f}%\n\n")
        
        print(f"💾 Memory logs saved to: {filepath}")


def monitor_training_step(model, batch_size=8, device='cuda'):
    """
    Test memory usage of a training step
    
    Args:
        model: PyTorch model
        batch_size: Batch size to test
        device: Device to use
    """
    monitor = GPUMemoryMonitor()
    monitor.log_memory("Initial state")
    
    # Simulate training batch
    dummy_input = torch.randn(batch_size, 3, 240, 240).to(device)
    dummy_target = torch.randint(0, 4, (batch_size,)).to(device)
    
    monitor.log_memory("Data loaded")
    
    # Forward pass
    model.train()
    output = model(dummy_input)
    monitor.log_memory("Forward pass")
    
    # Loss computation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, dummy_target)
    monitor.log_memory("Loss computed")
    
    # Backward pass
    loss.backward()
    monitor.log_memory("Backward pass")
    
    # Cleanup
    del dummy_input, dummy_target, output, loss
    monitor.clear_cache()
    
    monitor.print_summary()


if __name__ == "__main__":
    # Quick test
    print("🔍 GPU Memory Monitor Test\n")
    
    monitor = GPUMemoryMonitor()
    
    if torch.cuda.is_available():
        # Test with dummy tensor
        print("\nAllocating 1GB tensor...")
        x = torch.randn(1024, 1024, 256).cuda()  # ~1GB
        monitor.log_memory("1GB allocated")
        
        print("\nAllocating another 1GB tensor...")
        y = torch.randn(1024, 1024, 256).cuda()  # Another ~1GB
        monitor.log_memory("2GB allocated")
        
        print("\nCleaning up...")
        del x, y
        monitor.clear_cache()
        
        monitor.print_summary()
    else:
        print("CUDA not available - cannot test GPU memory")
