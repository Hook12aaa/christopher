#!/usr/bin/env python3
"""
Development Setup Script

Helps set up the project for development by:
1. Checking Python environment
2. Installing dependencies
3. Testing basic functionality
4. Running minimal examples

Usage:
    python setup_development.py
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check Python version is adequate"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version >= (3, 8):
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (adequate)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  ✅ Dependencies installed successfully")
            return True
        else:
            print(f"  ❌ Installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ Installation timed out (may need internet connection)")
        return False
    except Exception as e:
        print(f"  ❌ Installation error: {e}")
        return False

def test_basic_imports():
    """Test basic imports work"""
    print("\n🔍 Testing basic imports...")
    
    try:
        import numpy as np
        print("  ✅ NumPy available")
        
        import scipy
        print("  ✅ SciPy available")
        
        from sentence_transformers import SentenceTransformer
        print("  ✅ sentence-transformers available")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_bge_model():
    """Test BGE model can be loaded"""
    print("\n🤖 Testing BGE model (may download ~1GB on first run)...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("  📥 Loading BGE-Large-v1.5...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Test encoding
        test_embedding = model.encode("Test text")
        print(f"  ✅ Model loaded, embedding dimension: {len(test_embedding)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        print("  💡 This may be due to internet connection or disk space")
        return False

def run_development_tests():
    """Run development test suite"""
    print("\n🧪 Running development tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_development.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ Development tests passed")
            return True
        else:
            print(f"  ❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"  ❌ Test execution failed: {e}")
        return False

def run_minimal_example():
    """Run minimal example"""
    print("\n🚀 Running minimal example...")
    
    try:
        result = subprocess.run([
            sys.executable, "examples/minimal_example.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("  ✅ Minimal example completed successfully")
            print("\n📊 Example output (last 10 lines):")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                print(f"    {line}")
            return True
        else:
            print(f"  ❌ Example failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"  ❌ Example execution failed: {e}")
        return False

def create_development_workspace():
    """Create useful development files"""
    print("\n📁 Creating development workspace...")
    
    # Create .gitignore if it doesn't exist
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model cache
model_cache/
*.model

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
env/
venv/

# Output files
output/
results/
*.png
*.jpg
*.pdf

# Logs
*.log
logs/
"""
        gitignore_path.write_text(gitignore_content)
        print("  ✅ Created .gitignore")
    
    # Create development notes file
    notes_path = Path("DEVELOPMENT_NOTES.md")
    if not notes_path.exists():
        notes_content = """# Development Notes

## Current Status
- [ ] Basic imports working
- [ ] BGE model loading
- [ ] Simple conceptual charges
- [ ] Basic manifold operations
- [ ] Full mathematical implementation

## Next Tasks
1. Implement core ConceptualCharge class
2. Add BGE integration to ConceptualChargeGenerator  
3. Build basic transformation operator
4. Create simple manifold field evolution
5. Add visualization capabilities

## Development Commands
```bash
# Run tests
python tests/test_development.py

# Run minimal example  
python examples/minimal_example.py

# Install dependencies
pip install -r requirements.txt

# Format code
black .
isort .
```

## Notes
- Add your development notes here
- Track issues and solutions
- Document design decisions
"""
        notes_path.write_text(notes_content)
        print("  ✅ Created DEVELOPMENT_NOTES.md")
    
    print("  ✅ Development workspace ready")

def main():
    """Run complete setup process"""
    print("🔧 Field Theory Development Setup")
    print("=" * 50)
    
    # Check environment
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        print("💡 Try manually: pip install -r requirements.txt")
        return False
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Setup failed: Import errors")
        return False
    
    # Test BGE model (optional)
    model_success = test_bge_model()
    if not model_success:
        print("⚠️  BGE model loading failed, but continuing...")
    
    # Run tests
    test_success = run_development_tests()
    if not test_success:
        print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    example_success = run_minimal_example()
    if not example_success:
        print("⚠️  Minimal example failed, but setup may still be usable")
    
    # Create workspace
    create_development_workspace()
    
    print("\n" + "=" * 50)
    if model_success and test_success and example_success:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Ready to begin development!")
        print("\nNext steps:")
        print("1. Review DEVELOPMENT_GUIDE.md for implementation roadmap")
        print("2. Start with core_mathematics/conceptual_charge.py")
        print("3. Run tests frequently: python tests/test_development.py")
        print("4. Use examples/minimal_example.py to verify changes")
    else:
        print("⚠️  Setup completed with some issues")
        print("\nYou can still begin development, but some features may not work.")
        print("\nTroubleshooting:")
        if not model_success:
            print("- BGE model: Check internet connection and disk space")
        if not test_success:
            print("- Tests: Review test output for specific issues")
        if not example_success:
            print("- Example: Check example output for error details")
    
    print("\n📚 Key files:")
    print("- DEVELOPMENT_GUIDE.md: Implementation roadmap")
    print("- examples/minimal_example.py: Working demonstration")
    print("- tests/test_development.py: Development tests")
    print("- DEVELOPMENT_NOTES.md: Your notes and progress")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)