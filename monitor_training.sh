#!/bin/bash
# Monitor the extended training progress

echo "🔍 Phase 2 Extended Training Monitor"
echo "===================================="
echo ""

# Check if training is running
if pgrep -f "phase2_extended_training_v2.py" > /dev/null; then
    echo "✅ Training process is running"
    echo ""
    
    # Show last few lines of log
    if [ -f "phase2_extended_training_v2.log" ]; then
        echo "📊 Recent training output:"
        echo "---"
        tail -30 phase2_extended_training_v2.log
        echo "---"
        echo ""
        
        # Count epochs completed
        epoch_count=$(grep -c "{'loss':" phase2_extended_training_v2.log 2>/dev/null || echo "0")
        echo "📈 Epochs logged: ~$epoch_count"
    else
        echo "⚠️  Log file not found yet"
    fi
else
    echo "⚠️  Training process not found"
    
    # Check if training completed
    if [ -f "phase2_extended_v2_results.json" ]; then
        echo ""
        echo "✅ Training completed! Results:"
        cat phase2_extended_v2_results.json | grep -E '"accuracy_percent"|"avg_latency_ms"|"training_duration_minutes"' | head -3
    fi
fi

echo ""
echo "To view full log: tail -f phase2_extended_training_v2.log"
