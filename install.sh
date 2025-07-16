#!/bin/bash

echo "🚀 Installing RevealGrid..."

# Check if Python 3.10+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version found"
else
    echo "❌ Python 3.10+ required, found $python_version"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

# Create sample model file
echo "🤖 Creating sample model..."
python3 -c "
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a simple classifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 3, 100)
clf.fit(X, y)

# Save the model
with open('models/device_clf_v0.1.pkl', 'wb') as f:
    pickle.dump(clf, f)
print('✅ Sample model created')
"

echo "✅ Installation complete!"
echo ""
echo "🎯 To run RevealGrid:"
echo "   streamlit run app.py"
echo ""
echo "🌐 Open your browser to: http://localhost:8501" 