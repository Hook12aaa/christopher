# Quick Start Guide

## 🚀 Get Started Immediately

This guide gets you up and running with the Field Theory framework as quickly as possible.

## Environment Setup

### Option 1: Basic Setup (Start Here)
```bash
# 1. Install core dependencies
pip install numpy scipy matplotlib

# 2. Test basic functionality
python -c "import numpy, scipy; print('✅ Core math libraries working')"

# 3. Try the minimal example (without BGE model initially)  
python examples/minimal_example.py
```

### Option 2: Full Setup (After Basic Works)
```bash
# Install all dependencies
pip install -r requirements.txt

# If you get TensorFlow/Metal errors on Mac:
pip uninstall tensorflow tensorflow-metal
pip install torch torchvision sentence-transformers --no-deps
pip install transformers tokenizers
```

## Common Setup Issues

### 🔧 TensorFlow/Metal Compatibility (Mac Users)
If you see errors about `libmetal_plugin.dylib`, this is a known compatibility issue:

```bash
# Solution 1: Reinstall without metal
pip uninstall tensorflow tensorflow-metal  
pip install sentence-transformers

# Solution 2: Use CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

### 🔧 Import Errors
If imports fail:
```bash
# Check Python version (need 3.8+)
python --version

# Check if packages are installed
pip list | grep -E "(numpy|scipy|sentence|transformers)"

# Reinstall with specific versions
pip install numpy>=1.24.0 scipy>=1.10.0
```

## Verify Installation

### Test 1: Basic Math Operations
```bash
python -c "
import numpy as np
import scipy
result = np.exp(1j * np.pi) + 1
print(f'Complex math test: {abs(result) < 1e-10}')
print('✅ Basic mathematics working')
"
```

### Test 2: Core Project Structure
```bash
python -c "
import sys
from pathlib import Path
project_dirs = ['core_mathematics', 'embedding_engine', 'product_manifold']
for d in project_dirs:
    if Path(d).exists():
        print(f'✅ {d} directory exists')
    else:
        print(f'❌ {d} missing')
"
```

### Test 3: BGE Model (Optional - May Download ~1GB)
```bash
python -c "
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embedding = model.encode('test')
    print(f'✅ BGE model working, dimension: {len(embedding)}')
except Exception as e:
    print(f'⚠️ BGE model issue: {e}')
    print('This is optional - you can develop without it initially')
"
```

## First Steps

### 1. Run the Minimal Example
```bash
python examples/minimal_example.py
```

**Expected Output:**
```
🚀 Field Theory of Social Constructs - Minimal Example
🔍 Testing basic imports...
  ✅ NumPy and SciPy available
  ✅ Complex mathematics working
🧪 Testing Simple Conceptual Charges...
  ✅ Simple charge system working!
🌍 Testing Simple Manifold Concept...
  ✅ Simple manifold concept working!
🎉 SUCCESS: Minimal example completed successfully!
```

### 2. Run Development Tests
```bash
python tests/test_development.py
```

### 3. Begin Development
Follow the implementation priority in `DEVELOPMENT_GUIDE.md`:

1. **Phase 1**: Core ConceptualCharge implementation
2. **Phase 2**: Basic manifold operations  
3. **Phase 3**: Data pipeline and scaling
4. **Phase 4**: Advanced features

## Development Workflow

### Daily Cycle
```bash
# 1. Run tests to check current status
python tests/test_development.py

# 2. Work on one component at a time
# Edit files in core_mathematics/ or other modules

# 3. Test your changes
python examples/minimal_example.py

# 4. Commit working code
git add . && git commit -m "Working: describe what you implemented"
```

### Key Files to Implement First
1. `core_mathematics/conceptual_charge.py` - Core charge mathematics
2. `embedding_engine/models.py` - BGE integration  
3. `product_manifold/transformation_operator.py` - Charge to manifold conversion
4. `product_manifold/product_manifold.py` - Main orchestration

## Troubleshooting

### "Module not found" errors
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### BGE model download issues
```bash
# Manual download test
python -c "
from huggingface_hub import hf_hub_download
print('Testing Hugging Face access...')
# This will test connectivity without full model download
"
```

### Memory issues
The BGE model requires ~4GB RAM. If you have limited memory:
```bash
# Use smaller model for development
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Much smaller
print('✅ Smaller model working')
"
```

## Next Steps

1. **Get Minimal Example Working**: This proves your environment is ready
2. **Review Mathematical Framework**: Read `MATHEMATICAL_FOUNDATION.md`
3. **Follow Development Guide**: Implement components systematically
4. **Test Frequently**: Run tests after each component

## Getting Help

- **Import/Setup Issues**: Check this guide's troubleshooting section
- **Mathematical Questions**: See `MATHEMATICAL_FOUNDATION.md`
- **Implementation Guidance**: Follow `DEVELOPMENT_GUIDE.md`
- **Development Progress**: Use `tests/test_development.py` to verify

---

**Goal**: Get the minimal example working first. Everything else builds from that foundation.