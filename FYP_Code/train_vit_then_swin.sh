#!/bin/bash
# Auto-train ViT then Swin sequentially

echo "================================"
echo "Training ViT-Base..."
echo "================================"
python train_all_models_safe.py vit 2>&1 | tee logs/vit_live_training.log

echo ""
echo "================================"
echo "ViT Complete! Starting Swin..."
echo "================================"
python train_all_models_safe.py swin 2>&1 | tee logs/swin_live_training.log

echo ""
echo "================================"
echo "ALL TRAINING COMPLETE!"
echo "================================"
echo "✅ ViT-Base: Complete"
echo "✅ Swin-Tiny: Complete"
echo ""
echo "Check results:"
echo "  ls -lh experiments/phase2_systematic/models/vit/"
echo "  ls -lh experiments/phase2_systematic/models/swin/"
