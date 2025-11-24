#!/bin/bash
# Auto-monitor script: Start Swin after ViT completes

echo "======================================================================"
echo "AUTO-CONTINUATION MONITOR - ViT -> Swin"
echo "======================================================================"
echo "Started at: $(date)"
echo ""
echo "Waiting for ViT training to complete (5 models)..."
echo "Checking every 5 minutes..."
echo ""

check_count=0
while true; do
    check_count=$((check_count + 1))
    
    # Count ViT models
    vit_count=$(ls experiments/phase2_systematic/models/vit/*.pth 2>/dev/null | wc -l)
    
    # Check if ViT training process is running
    if ps aux | grep "train_all_models_safe.py vit" | grep -v grep > /dev/null; then
        training_status="RUNNING"
    else
        training_status="IDLE"
    fi
    
    echo "[$(date +%H:%M:%S)] Check #${check_count}: ViT ${vit_count}/5 models | Status: ${training_status}"
    
    # If ViT complete and not running, start Swin
    if [ "$vit_count" -ge 5 ] && [ "$training_status" = "IDLE" ]; then
        echo ""
        echo "======================================================================"
        echo "ViT TRAINING COMPLETED! Starting Swin training..."
        echo "======================================================================"
        echo ""
        
        python train_all_models_safe.py swin > logs/swin_training_240_auto.log 2>&1 &
        swin_pid=$!
        
        echo "Swin training started (PID: ${swin_pid})"
        echo "Log file: logs/swin_training_240_auto.log"
        echo "Monitor with: tail -f logs/swin_training_240_auto.log"
        echo ""
        echo "Auto-continuation complete!"
        break
    fi
    
    # Wait 5 minutes
    sleep 300
done
