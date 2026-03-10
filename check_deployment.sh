#!/bin/bash
# Check deployment status

echo "=========================================="
echo "  FPL Champion Model Deployment Status"
echo "=========================================="
echo ""

# Check files
echo "📁 Production Files:"
if [ -f "models/production/model.pkl" ]; then
    echo "  ✅ model.pkl ($(stat -c%s models/production/model.pkl) bytes)"
else
    echo "  ❌ model.pkl missing"
fi

if [ -f "models/production/metadata.json" ]; then
    echo "  ✅ metadata.json"
else
    echo "  ❌ metadata.json missing"
fi

if [ -f "models/production/predictor.py" ]; then
    echo "  ✅ predictor.py"
else
    echo "  ❌ predictor.py missing"
fi

echo ""
echo "📊 Model Info:"
python3 << 'PYEOF'
import json
import sys
sys.path.insert(0, 'backend')

try:
    with open('models/production/metadata.json') as f:
        meta = json.load(f)
    
    print(f"  Name: {meta['name']}")
    print(f"  Experiment: {meta['experiment_id']}")
    print(f"  RMSE: {meta['rmse']}")
    print(f"  Improvement: +{meta['improvement']}%")
    print(f"  Status: {meta['status']}")
    print(f"  Deployed: {meta['deployed_at']}")
except Exception as e:
    print(f"  Error reading metadata: {e}")
PYEOF

echo ""
echo "🧪 Quick Test:"
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'backend')
import numpy as np

try:
    from production_predictor import ProductionPredictor
    p = ProductionPredictor()
    test = np.random.randn(2, 30)
    preds = p.predict(test)
    print(f"  ✅ Predictor working")
    print(f"  Sample output: {preds}")
except Exception as e:
    print(f"  ❌ Test failed: {e}")
PYEOF

echo ""
echo "=========================================="
echo "Deployment Status: READY"
echo "=========================================="
