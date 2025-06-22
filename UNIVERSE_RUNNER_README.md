# Universe Runner - Safe Universe Management

## Overview

The Universe Runner provides a safe interface for loading and interacting with existing stored liquid universes **WITHOUT** triggering hours of embedding computation.

## Key Distinction

- **`foundation_manifold_builder.py`**: For building NEW universes (requires hours of embedding computation)
- **`universe_runner.py`**: For loading EXISTING universes (fast, no embedding computation)

## Quick Start

### 1. List Available Universes
```bash
python Sysnpire/model/universe_runner.py --list
```

### 2. Load and Reconstruct Universe
```bash
python Sysnpire/model/universe_runner.py --load universe_130103de_1750282761 --device cpu
```

### 3. Load Universe + Run Evolution
```bash
python Sysnpire/model/universe_runner.py --load universe_130103de_1750282761 --device cpu --evolve --steps 50
```

### 4. Interactive Mode
```bash
python Sysnpire/model/universe_runner.py --interactive
```

## Features

### ✅ Universe Discovery
- Automatically scans `liquid_universes/` directory
- Lists available universes with metadata
- Shows agent count, creation time, consistency status

### ✅ Universe Reconstruction  
- Complete mathematical precision restoration
- All 100+ agents reconstructed from HDF5 storage
- Device selection (CPU/MPS/CUDA)
- Validation and consistency checking

### ✅ Evolution Simulation
- Run living evolution on reconstructed universes
- Configurable tau steps and step sizes
- Real-time progress monitoring
- Evolution metrics and analysis

### ✅ Interactive CLI
- Menu-driven interface for universe operations
- Step-by-step universe selection and loading
- Safe exploration without accidental builds

## Command Line Options

```
python universe_runner.py [OPTIONS]

Options:
  --storage-path PATH     Base path to search for stored universes (default: liquid_universes)
  --list                  List all available universes
  --load UNIVERSE_ID      Load specific universe by ID
  --device {cpu,mps,cuda} Device for reconstruction (default: cpu)
  --evolve                Run evolution simulation after loading
  --steps INT             Evolution steps (default: 100)
  --step-size FLOAT       Evolution step size (default: 0.01)
  --interactive           Start interactive menu
```

## Example Output

### Universe Listing
```
🌊 AVAILABLE LIQUID UNIVERSES
================================================================================

1. universe_130103de_1750282761
   📁 Storage: 1750282691
   🤖 Agents: 100
   📅 Created: 1750282770.941987
   🔍 Status: ✅ Consistent

2. universe_d0018b30_1750282354
   📁 Storage: 1750282285
   🤖 Agents: 100
   📅 Created: 1750282364.3636532
   🔍 Status: ⚠️ Inconsistent
```

### Reconstruction Results
```
✅ Universe reconstruction successful!
   Agents: 100
   Field energy: 10.000000
   Time: 15.23s
   Validation: ✅ PASSED
```

### Evolution Summary
```
📊 Evolution Summary:
   final_complexity: 0.856432
   total_Q_energy: 12.345678
   evolution_steps_completed: 50
   convergence_achieved: True
```

## Safety Features

### 🔒 No Accidental Building
- Clear separation from `foundation_manifold_builder.py`
- Only works with existing storage
- No embedding model initialization
- No risk of triggering hours-long computations

### ⚡ Fast Operations
- Quick universe listing and metadata
- Efficient reconstruction using existing pipeline
- Minimal memory footprint
- Optimized for interactive exploration

## Integration Points

### Uses Existing Components
- `FieldUniverse.reconstruct_liquid_universe()`
- `LiquidOrchestrator.orchestrate_living_evolution()`
- `StorageCoordinator` (HDF5 + Arrow storage)
- Complete reconstruction pipeline from previous work

### No Dependencies On
- BGE/MPNet model loading
- ChargeFactory building  
- Embedding computation
- Foundation model downloads

## Architecture

```
universe_runner.py
├── UniverseRunner class
├── discover_universes() → Find stored universes
├── reconstruct_universe() → Load from storage
├── run_evolution_simulation() → Evolve loaded universe
└── interactive_menu() → CLI interface

Integration with:
├── FieldUniverse → High-level reconstruction
├── LiquidOrchestrator → Evolution orchestration  
├── StorageCoordinator → HDF5/Arrow storage
└── ConceptualChargeAgent → Agent reconstruction
```

## Quick Demo Script

For even easier access, use:
```bash
python quick_universe_demo.py
```

This provides a simple menu-driven interface to universe operations.

## Troubleshooting

### No Universes Found
- Check that `liquid_universes/` directory exists
- Verify HDF5 files are present in storage directories
- Run `foundation_manifold_builder.py` first to create universes

### Reconstruction Warnings
- Minor warnings about missing attributes are normal
- Agents will still reconstruct successfully with fallback values
- Check that storage is consistent

### Performance Issues
- Use `--device cpu` for compatibility
- Reduce evolution steps for faster testing
- Check available memory for large universes

## Next Steps

1. **Explore Existing Universes**: Start with `--list` to see what's available
2. **Test Reconstruction**: Load a universe to verify the pipeline works
3. **Run Evolution**: Experiment with different tau steps and parameters
4. **Interactive Mode**: Use the menu system for guided exploration

**Remember**: This tool is for exploring existing universes safely. Use `foundation_manifold_builder.py` only when you want to create new universes and can afford hours of computation time.