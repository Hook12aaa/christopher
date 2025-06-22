#!/usr/bin/env python3
"""
Universe Runner - Load and Interact with Existing Liquid Universes

This script provides a safe interface for loading and working with existing
stored liquid universes WITHOUT triggering hours of embedding computation.

Key Features:
- List available stored universes with metadata
- Reconstruct universes from HDF5/Arrow storage
- Run evolution simulations on reconstructed universes
- Interactive CLI for universe management
- NO embedding computation (foundation_manifold_builder.py is for building)

Usage Examples:
    python universe_runner.py --list
    python universe_runner.py --load universe_130103de_1750282761 --evolve --steps 50
    python universe_runner.py --interactive
"""

# CRITICAL FIX: Set float32 default BEFORE any other imports to prevent MPS float64 errors
import torch
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
    print("🔧 UniverseRunner: MPS detected, setting float32 default dtype")

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Sysnpire.database.field_universe import FieldUniverse, FieldUniverseConfig
from Sysnpire.model.liquid.liquid_orchestrator import LiquidOrchestrator
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class UniverseRunner:
    """
    Main runner for interacting with existing liquid universes.
    
    Provides safe access to stored universes without embedding computation.
    """
    
    def __init__(self, storage_base_path: str = "liquid_universes"):
        """
        Initialize universe runner.
        
        Args:
            storage_base_path: Base path to search for stored universes
        """
        # Handle relative paths by resolving from project root
        if not Path(storage_base_path).is_absolute():
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            self.storage_base_path = project_root / storage_base_path
        else:
            self.storage_base_path = Path(storage_base_path)
        self.field_universes = {}  # Cache for FieldUniverse instances
        
        logger.info(f"🌊 Universe Runner initialized")
        logger.info(f"   Storage base path: {self.storage_base_path}")
    
    def discover_universes(self) -> List[Dict]:
        """
        Discover all available stored universes.
        
        Returns:
            List of universe metadata dictionaries
        """
        logger.info("🔍 Discovering available universes...")
        
        universes = []
        
        if not self.storage_base_path.exists():
            logger.warning(f"Storage path does not exist: {self.storage_base_path}")
            return universes
        
        # Look for universe storage directories
        for storage_dir in self.storage_base_path.iterdir():
            if storage_dir.is_dir():
                try:
                    # Try to create FieldUniverse to validate storage
                    config = FieldUniverseConfig(storage_path=storage_dir)
                    field_universe = FieldUniverse(config)
                    
                    # Get universe list from this storage
                    stored_universes = field_universe.storage_coordinator.list_universes()
                    
                    for universe_info in stored_universes:
                        universe_info["storage_path"] = str(storage_dir)
                        universes.append(universe_info)
                    
                    # Cache field universe for later use
                    self.field_universes[str(storage_dir)] = field_universe
                    
                except Exception as e:
                    logger.debug(f"Skipping {storage_dir}: {e}")
                    continue
        
        logger.info(f"✅ Discovered {len(universes)} universes across {len(self.field_universes)} storage locations")
        return universes
    
    def display_universe_list(self, universes: List[Dict]):
        """Display formatted list of available universes."""
        
        print("\n" + "="*80)
        print("🌊 AVAILABLE LIQUID UNIVERSES")
        print("="*80)
        
        if not universes:
            print("❌ No universes found!")
            print(f"   Check storage path: {self.storage_base_path}")
            return
        
        for i, universe in enumerate(universes, 1):
            print(f"\n{i}. {universe['universe_id']}")
            print(f"   📁 Storage: {Path(universe['storage_path']).name}")
            
            if 'num_agents' in universe:
                print(f"   🤖 Agents: {universe['num_agents']}")
            
            if 'creation_time' in universe:
                print(f"   📅 Created: {universe['creation_time']}")
            
            if 'consistent' in universe:
                status = "✅ Consistent" if universe['consistent'] else "⚠️ Inconsistent"
                print(f"   🔍 Status: {status}")
    
    def reconstruct_universe(self, universe_id: str, device: str = "mps") -> Optional[LiquidOrchestrator]:
        """
        Reconstruct a specific universe from storage.
        
        Args:
            universe_id: Universe identifier to reconstruct
            device: Device for tensor operations (cpu, mps, cuda)
            
        Returns:
            Living LiquidOrchestrator or None if failed
        """
        logger.info(f"🔄 Reconstructing universe: {universe_id}")
        
        # Device compatibility check and automatic fallback
        import torch
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("⚠️ MPS not available, falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            device = "cpu"
        
        logger.info(f"   Target device: {device}")
        
        # Find the storage path for this universe
        storage_path = None
        for path, field_universe in self.field_universes.items():
            universes = field_universe.storage_coordinator.list_universes()
            universe_ids = [u['universe_id'] for u in universes]
            if universe_id in universe_ids:
                storage_path = path
                break
        
        if not storage_path:
            logger.error(f"❌ Universe {universe_id} not found in any storage location")
            return None
        
        try:
            field_universe = self.field_universes[storage_path]
            
            # Reconstruct the universe
            reconstruction_result = field_universe.reconstruct_liquid_universe(
                universe_id=universe_id,
                device=device
            )
            
            if reconstruction_result["status"] == "success":
                logger.info("✅ Universe reconstruction successful!")
                logger.info(f"   Agents: {reconstruction_result['agents_count']}")
                logger.info(f"   Field energy: {reconstruction_result['field_energy']:.6f}")
                logger.info(f"   Time: {reconstruction_result['reconstruction_time']:.2f}s")
                logger.info(f"   Validation: {'✅ PASSED' if reconstruction_result['validation_passed'] else '❌ FAILED'}")
                
                # Additional validation for evolution readiness
                orchestrator = reconstruction_result["orchestrator"]
                if hasattr(orchestrator, 'adaptive_tuning') and orchestrator.adaptive_tuning:
                    listener_count = sum(1 for key in ['eigenvalue_listener', 'breathing_listener', 'interaction_listener', 'cascade_listener', 'phase_listener'] 
                                       if key in orchestrator.adaptive_tuning and hasattr(orchestrator.adaptive_tuning[key], 'listen'))
                    logger.info(f"   Evolution ready: {listener_count}/5 listeners active")
                
                return orchestrator
            else:
                logger.error(f"❌ Reconstruction failed: {reconstruction_result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Reconstruction error: {e}")
            return None
    
    def run_evolution_simulation(self, orchestrator: LiquidOrchestrator, tau_steps: int = 100, tau_step_size: float = 0.01):
        """
        Run evolution simulation on reconstructed universe.
        
        Args:
            orchestrator: Living LiquidOrchestrator
            tau_steps: Number of evolution steps
            tau_step_size: Size of each evolution step
        """
        logger.info("🧬 Starting evolution simulation...")
        logger.info(f"   Steps: {tau_steps}")
        logger.info(f"   Step size: {tau_step_size}")
        logger.info(f"   Active agents: {len(orchestrator.charge_agents)}")
        
        start_time = time.time()
        
        try:
            evolution_results = orchestrator.orchestrate_living_evolution(
                tau_steps=tau_steps,
                tau_step_size=tau_step_size
            )
            
            evolution_time = time.time() - start_time
            
            logger.info("✅ Evolution simulation complete!")
            logger.info(f"   Time: {evolution_time:.2f}s")
            logger.info(f"   Final complexity: {evolution_results.get('final_complexity', 'unknown')}")
            logger.info(f"   Total Q energy: {evolution_results.get('final_total_Q_energy', 'unknown')}")
            
            # Display key metrics
            if 'evolution_summary' in evolution_results:
                summary = evolution_results['evolution_summary']
                print("\n📊 Evolution Summary:")
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
                    else:
                        print(f"   {key}: {value}")
            
            return evolution_results
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Evolution simulation failed: {e}")
            logger.error(f"📍 FULL TRACEBACK:")
            logger.error(traceback.format_exc())
            print(f"\n🔍 DETAILED ERROR LOCATION:\n{traceback.format_exc()}")
            return None
    
    def interactive_menu(self):
        """Interactive CLI menu for universe operations."""
        
        print("\n" + "="*80)
        print("🌊 LIQUID UNIVERSE INTERACTIVE RUNNER")
        print("="*80)
        
        while True:
            print("\nOptions:")
            print("1. List available universes")
            print("2. Reconstruct universe")
            print("3. Run evolution simulation")
            print("4. Exit")
            
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == "1":
                    universes = self.discover_universes()
                    self.display_universe_list(universes)
                
                elif choice == "2":
                    universes = self.discover_universes()
                    if not universes:
                        continue
                    
                    self.display_universe_list(universes)
                    universe_idx = input(f"\nSelect universe (1-{len(universes)}): ").strip()
                    
                    try:
                        idx = int(universe_idx) - 1
                        if 0 <= idx < len(universes):
                            universe_id = universes[idx]['universe_id']
                            device = input("Device (cpu/mps/cuda) [mps]: ").strip() or "mps"
                            
                            orchestrator = self.reconstruct_universe(universe_id, device)
                            if orchestrator:
                                # Store for potential evolution
                                self.current_orchestrator = orchestrator
                        else:
                            print("❌ Invalid selection")
                    except ValueError:
                        print("❌ Invalid input")
                
                elif choice == "3":
                    if not hasattr(self, 'current_orchestrator'):
                        print("❌ No universe loaded. Reconstruct a universe first.")
                        continue
                    
                    steps = input("Evolution steps [100]: ").strip()
                    steps = int(steps) if steps else 100
                    
                    step_size = input("Step size [0.01]: ").strip()
                    step_size = float(step_size) if step_size else 0.01
                    
                    self.run_evolution_simulation(self.current_orchestrator, steps, step_size)
                
                elif choice == "4":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid option")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Universe Runner - Load and interact with existing liquid universes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python universe_runner.py --list
  python universe_runner.py --load universe_130103de_1750282761 --evolve --steps 50
  python universe_runner.py --interactive
        """
    )
    
    parser.add_argument("--storage-path", default="liquid_universes",
                       help="Base path to search for stored universes")
    parser.add_argument("--list", action="store_true",
                       help="List all available universes")
    parser.add_argument("--load", metavar="UNIVERSE_ID",
                       help="Load specific universe by ID")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"],
                       help="Device for reconstruction (default: mps)")
    parser.add_argument("--evolve", action="store_true",
                       help="Run evolution simulation after loading")
    parser.add_argument("--steps", type=int, default=100,
                       help="Evolution steps (default: 100)")
    parser.add_argument("--step-size", type=float, default=0.01,
                       help="Evolution step size (default: 0.01)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive menu")
    
    args = parser.parse_args()
    
    # Initialize runner with proper path handling
    runner = UniverseRunner(storage_base_path=args.storage_path)
    
    try:
        if args.interactive:
            runner.interactive_menu()
        
        elif args.list:
            universes = runner.discover_universes()
            runner.display_universe_list(universes)
        
        elif args.load:
            # Discover universes first
            universes = runner.discover_universes()
            
            # Load specific universe
            orchestrator = runner.reconstruct_universe(args.load, args.device)
            
            if orchestrator and args.evolve:
                runner.run_evolution_simulation(orchestrator, args.steps, args.step_size)
        
        else:
            # Default: show help and start interactive mode
            parser.print_help()
            print("\nStarting interactive mode...")
            runner.interactive_menu()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()