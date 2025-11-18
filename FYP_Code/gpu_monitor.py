"""
Continuous GPU Temperature and Safety Monitor
Ensures RTX 6000 Ada operates within safe limits during training
Temperature threshold: 83°C (professional GPU safe limit)
"""

import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Safety thresholds for RTX 6000 Ada
TEMP_WARN = 80  # Warning threshold
TEMP_CRITICAL = 83  # Critical threshold (will log warning)
TEMP_EMERGENCY = 87  # Emergency threshold (GPU will throttle itself)

MONITOR_INTERVAL = 30  # Check every 30 seconds
LOG_FILE = Path("gpu_monitoring.log")

def get_gpu_stats():
    """Get current GPU statistics"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=timestamp,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            timestamp, temp, mem_used, mem_total, gpu_util, power, power_limit = result.stdout.strip().split(', ')
            return {
                'timestamp': timestamp,
                'temperature': int(temp),
                'memory_used_gb': float(mem_used) / 1024,
                'memory_total_gb': float(mem_total) / 1024,
                'memory_percent': (float(mem_used) / float(mem_total)) * 100,
                'gpu_utilization': int(gpu_util),
                'power_draw': float(power),
                'power_limit': float(power_limit),
                'power_percent': (float(power) / float(power_limit)) * 100
            }
    except Exception as e:
        return None

def log_stats(stats, level="INFO"):
    """Log GPU statistics"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] Temp: {stats['temperature']}C | VRAM: {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f} GB ({stats['memory_percent']:.0f}%) | GPU: {stats['gpu_utilization']}% | Power: {stats['power_draw']:.0f}W ({stats['power_percent']:.0f}%)\n"

    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

    return log_entry.strip()

def check_safety(stats):
    """Check if GPU is operating safely"""
    temp = stats['temperature']

    if temp >= TEMP_EMERGENCY:
        return "EMERGENCY", f"Temperature {temp}C exceeds emergency threshold {TEMP_EMERGENCY}C!"
    elif temp >= TEMP_CRITICAL:
        return "CRITICAL", f"Temperature {temp}C exceeds critical threshold {TEMP_CRITICAL}C - GPU will throttle soon"
    elif temp >= TEMP_WARN:
        return "WARNING", f"Temperature {temp}C approaching critical levels"
    else:
        return "OK", f"Temperature {temp}C within safe range"

def main():
    """Main monitoring loop"""
    print("=" * 80)
    print("GPU SAFETY MONITOR - RTX 6000 Ada Generation")
    print("=" * 80)
    print(f"Temperature Warning: {TEMP_WARN}°C")
    print(f"Temperature Critical: {TEMP_CRITICAL}°C")
    print(f"Temperature Emergency: {TEMP_EMERGENCY}°C")
    print(f"Check Interval: {MONITOR_INTERVAL} seconds")
    print(f"Log File: {LOG_FILE}")
    print("=" * 80)
    print()
    print("Monitoring started... (Press Ctrl+C to stop)")
    print()

    iteration = 0

    try:
        while True:
            stats = get_gpu_stats()

            if stats:
                status, message = check_safety(stats)

                # Log every iteration
                log_entry = log_stats(stats, level=status)

                # Print to console every 10 iterations (5 minutes) or if warning/critical
                if iteration % 10 == 0 or status != "OK":
                    print(log_entry)
                    if status != "OK":
                        print(f"  >> {message}")

                iteration += 1
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to get GPU stats")

            time.sleep(MONITOR_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(f"Total iterations: {iteration}")
        print(f"Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
