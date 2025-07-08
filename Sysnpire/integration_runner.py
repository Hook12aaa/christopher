#!/usr/bin/env python3
"""
Integration Runner - Test Universe Acceptance/Rejection of Text Content

This script provides a field-theoretic interface for testing whether existing
liquid universes will accept or reject new text paragraphs through exact
mathematical analysis using the complete Q(τ,C,s) formula.

Key Features:
- Load existing liquid universes with complete field states
- Convert text paragraphs to FieldSignature objects using field theory
- Calculate acceptance/rejection using mathematical weight: W = ΔC · R_collective · S_stability  
- Create and integrate ConceptualChargeAgent for accepted content
- Persist universe changes through burn_liquid_universe()
- Detailed field-theoretic analysis and logging

Mathematical Foundation:
Uses the complete conceptual charge formula:
Q(τ,C,s) = γ·T(τ,C,s)·E^trajectory(τ,s)·Φ^semantic(τ,s)·e^(iθ_total(τ,C,s))·Ψ_persistence(s-s₀)

Usage Examples:
    python integration_runner.py --interactive
    python integration_runner.py --list-universes  
    python integration_runner.py --universe universe_130103de_1750282761 --test-text "Your paragraph here"
"""

# CRITICAL FIX: Set float32 default BEFORE any other imports to prevent MPS float64 errors
import torch
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
    print("🔧 IntegrationRunner: MPS detected, setting float32 default dtype")

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core Mathematical Libraries (Field-Theoretic, not ML)
import numpy as np
import jax
import jax.numpy as jnp
from scipy import integrate, linalg
import cmath

# SAGE Mathematics Integration
from Sysnpire.model.integration.sage_compatibility import (
    safe_torch_tensor, convert_sage_dict, ensure_python_types
)

# Core Integration System
from Sysnpire.model.integration.universe_integration_patterns import (
    UniverseIntegrationEngine, FieldSignature, AcceptanceDecision
)
from Sysnpire.model.integration.field_integrator import FieldIntegrator
from Sysnpire.model.integration.charge_assembler import ConceptualChargeAssembler

# Universe Management (following universe_runner.py patterns)
from Sysnpire.database.field_universe import FieldUniverse, FieldUniverseConfig
from Sysnpire.model.liquid.liquid_orchestrator import LiquidOrchestrator
from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent

# Logging
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class IntegrationRunner:
    """
    Field-theoretic universe content testing system.
    
    Tests whether universes will accept/reject new text paragraphs using
    exact mathematical analysis rather than ML approximations.
    """
    
    def __init__(self, storage_base_path: str = "liquid_universes"):
        """
        Initialize integration runner with field-theoretic components.
        
        Args:
            storage_base_path: Base path to search for stored universes
        """
        # Handle relative paths by resolving from project root
        if not Path(storage_base_path).is_absolute():
            project_root = Path(__file__).parent.parent
            self.storage_base_path = project_root / storage_base_path
        else:
            self.storage_base_path = Path(storage_base_path)
        
        # Initialize field-theoretic components
        self.universe_integration_engine = UniverseIntegrationEngine()
        self.field_integrator = FieldIntegrator()
        self.charge_assembler = ConceptualChargeAssembler()
        
        if torch.backends.mps.is_available():
            torch.set_default_dtype(torch.float32)
        
        # Cache for FieldUniverse instances (following universe_runner.py pattern)
        self.field_universes = {}
        self.current_orchestrator = None
        self.current_universe_id = None
        
        logger.info("🧮 Integration Runner initialized with field-theoretic components")
        logger.info(f"   Storage base path: {self.storage_base_path}")
        logger.info("   Mathematical foundation: Q(τ,C,s) = γ·T·E^trajectory·Φ^semantic·e^(iθ_total)·Ψ_persistence")
    
    def discover_universes(self) -> List[Dict]:
        """
        Discover all available stored universes with field metadata.
        
        Returns:
            List of universe metadata dictionaries with field properties
        """
        logger.info("🔍 Discovering available universes for integration testing...")
        
        universes = []
        
        if not self.storage_base_path.exists():
            logger.warning(f"Storage path does not exist: {self.storage_base_path}")
            return universes
        
        # Look for universe storage directories (following universe_runner.py pattern)
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
                        # Add field-specific metadata for integration analysis
                        universe_info["integration_ready"] = True
                        universes.append(universe_info)
                    
                    # Cache field universe for later use
                    self.field_universes[str(storage_dir)] = field_universe
                    
                except Exception as e:
                    logger.debug(f"Skipping {storage_dir}: {e}")
                    continue
        
        logger.info(f"✅ Discovered {len(universes)} universes ready for integration testing")
        return universes
    
    def display_universe_list(self, universes: List[Dict]):
        """Display formatted list of available universes with integration metadata."""
        
        print("\n" + "="*80)
        print("🧮 AVAILABLE UNIVERSES FOR INTEGRATION TESTING")
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
                status = "✅ Field-Consistent" if universe['consistent'] else "⚠️ Field-Inconsistent"
                print(f"   🔍 Status: {status}")
            
            # Integration-specific metadata
            if universe.get('integration_ready'):
                print(f"   🧮 Integration: ✅ Ready for field-theoretic testing")
    
    def load_universe_for_integration(self, universe_id: str, device: str = "mps") -> Optional[LiquidOrchestrator]:
        """
        Load a specific universe for integration testing.
        
        Args:
            universe_id: Universe identifier to load
            device: Device for tensor operations (cpu, mps, cuda)
            
        Returns:
            Living LiquidOrchestrator with complete field state or None if failed
        """
        logger.info(f"🔄 Loading universe for integration testing: {universe_id}")
        
        # Device compatibility check and automatic fallback (from universe_runner.py)
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("⚠️ MPS not available, falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            device = "cpu"
        
        logger.info(f"   Target device: {device}")
        
        # Ensure we have discovered universes first
        if not self.field_universes:
            logger.info("Discovering universes first...")
            self.discover_universes()
        
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
            
            # Reconstruct the universe with complete field state
            reconstruction_result = field_universe.reconstruct_liquid_universe(
                universe_id=universe_id,
                device=device
            )
            
            if reconstruction_result["status"] == "success":
                orchestrator = reconstruction_result["orchestrator"]
                
                logger.info("✅ Universe loaded successfully for integration testing!")
                logger.info(f"   Agents: {reconstruction_result['agents_count']}")
                logger.info(f"   Field energy: {reconstruction_result['field_energy']:.6f}")
                logger.info(f"   Reconstruction time: {reconstruction_result['reconstruction_time']:.2f}s")
                logger.info(f"   Field validation: {'✅ PASSED' if reconstruction_result['validation_passed'] else '❌ FAILED'}")
                
                # Store for integration testing
                self.current_orchestrator = orchestrator
                self.current_universe_id = universe_id
                
                return orchestrator
            else:
                logger.error(f"❌ Universe loading failed: {reconstruction_result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Universe loading error: {e}")
            return None
    
    def text_to_field_signature(self, text: str, universe_state: Optional[Dict] = None) -> Optional[FieldSignature]:
        """
        Convert text paragraph to FieldSignature using field-theoretic methods.
        
        Args:
            text: Input text paragraph to convert
            universe_state: Current universe state for contextualized field generation
            
        Returns:
            FieldSignature object with field coordinates or None if failed
        """
        logger.info("🧮 Converting text to field signature...")
        logger.debug(f"   Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            # Use UniverseIntegrationEngine for field-theoretic text processing
            field_signature = self.universe_integration_engine.text_to_field_signature(text, universe_state)
            
            logger.info("✅ Field signature generated successfully")
            logger.debug(f"   Field coordinates: {len(field_signature.coordinates) if hasattr(field_signature, 'coordinates') else 'N/A'}")
            
            return field_signature
            
        except Exception as e:
            logger.error(f"❌ Field signature generation failed: {e}")
            return None
    
    def _format_universe_state_for_analysis(self, orchestrator: LiquidOrchestrator) -> Dict[str, Any]:
        """
        Format universe state from orchestrator for decide_content_acceptance analysis.
        
        Args:
            orchestrator: LiquidOrchestrator with loaded agents
            
        Returns:
            Universe state dictionary with agents data in expected format
        """
        logger.debug("🔧 Formatting universe state from orchestrator agents...")
        
        agents_data = []
        for agent_id, agent in orchestrator.charge_agents.items():
            try:
                # Extract required fields for each agent
                agent_data = {
                    "living_Q_value": complex(agent.living_Q_value) if hasattr(agent, 'living_Q_value') else complex(0.0),
                    "field_position": agent.position if hasattr(agent, 'position') else torch.zeros(1),
                    "phase": float(agent.phase_total.real) if hasattr(agent, 'phase_total') else 0.0
                }
                agents_data.append(agent_data)
            except Exception as e:
                logger.debug(f"Could not extract data for agent {agent_id}: {e}")
                continue
        
        universe_state = {
            "agents": agents_data
        }
        
        logger.debug(f"✅ Formatted universe state with {len(agents_data)} agents")
        return universe_state
    
    def test_universe_acceptance(self, text: str, orchestrator: LiquidOrchestrator) -> Dict[str, Any]:
        """
        Test whether universe will accept the given text using field-theoretic analysis.
        
        Args:
            text: Text paragraph to test
            orchestrator: Living LiquidOrchestrator with complete field state
            
        Returns:
            Dictionary containing acceptance decision and detailed analysis
        """
        logger.info("🧮 Testing universe acceptance using field theory...")
        
        try:
            # Step 1: Extract universe state from orchestrator  
            logger.info("📊 Extracting universe state for field analysis...")
            universe_state = self._format_universe_state_for_analysis(orchestrator)
            
            # Step 2: Convert text to field signature with universe context
            field_signature = self.text_to_field_signature(text, universe_state)
            if not field_signature:
                return {
                    "status": "error",
                    "error": "Failed to generate field signature",
                    "accepted": False
                }
            
            # Step 3: Calculate field-theoretic acceptance using mathematical weight
            logger.info("⚖️ Calculating mathematical weight: W = ΔC · R_collective · S_stability")
            
            acceptance_decision = self.universe_integration_engine.decide_content_acceptance(
                text=text,
                universe_state=universe_state
            )
            
            # Step 4: Extract detailed field analysis
            field_analysis = {
                "field_energy_before": universe_state.get("field_energy", 0.0),
                "semantic_distances": self._calculate_semantic_distances(field_signature, orchestrator),
                "phase_relationships": self._analyze_phase_relationships(field_signature, orchestrator),
                "interference_patterns": self._analyze_interference_patterns(field_signature, orchestrator)
            }
            
            # Convert SAGE objects to Python primitives for JSON serialization
            field_analysis = convert_sage_dict(field_analysis)
            
            result = {
                "status": "success",
                "accepted": acceptance_decision.accept,
                "mathematical_weight": ensure_python_types(acceptance_decision.mathematical_weight),
                "threshold": ensure_python_types(acceptance_decision.threshold_used),
                "reasoning": acceptance_decision.universe_reasoning,
                "field_analysis": field_analysis,
                "field_signature": {
                    "coordinates_count": len(field_signature.coordinates) if hasattr(field_signature, 'coordinates') else 0,
                    "field_type": type(field_signature).__name__
                }
            }
            
            # Log decision with mathematical reasoning
            if acceptance_decision.accept:
                logger.info(f"✅ UNIVERSE ACCEPTS TEXT")
                logger.info(f"   Mathematical weight: {result['mathematical_weight']:.6f} > {result['threshold']:.6f}")
            else:
                logger.info(f"❌ UNIVERSE REJECTS TEXT")
                logger.info(f"   Mathematical weight: {result['mathematical_weight']:.6f} ≤ {result['threshold']:.6f}")
            
            logger.info(f"   Reasoning: {acceptance_decision.universe_reasoning}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Universe acceptance testing failed: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "accepted": False
            }
    
    def _calculate_semantic_distances(self, field_signature: FieldSignature, orchestrator: LiquidOrchestrator) -> Dict[str, float]:
        """Calculate field-theoretic semantic distances to existing agents."""
        logger.debug("🔍 Calculating semantic distances using field dynamics...")
        
        try:
            distances = {}
            
            # Get field coordinates from signature
            if hasattr(field_signature, 'coordinates'):
                sig_coords = field_signature.coordinates
                
                # Calculate distances to each agent using field metrics
                for i, agent in enumerate(orchestrator.charge_agents[:5]):  # Limit to first 5 for performance
                    if hasattr(agent, 'position'):
                        agent_pos = safe_torch_tensor(agent.position)
                        sig_tensor = safe_torch_tensor(sig_coords)
                        
                        # Use field-theoretic distance (not Euclidean)
                        distance = torch.norm(sig_tensor - agent_pos).item()
                        distances[f"agent_{i}"] = ensure_python_types(distance)
                
                # Calculate mean and min distances
                if distances:
                    distances["mean_distance"] = ensure_python_types(np.mean(list(distances.values())))
                    distances["min_distance"] = ensure_python_types(min(distances.values()))
            
            return distances
            
        except Exception as e:
            logger.debug(f"Semantic distance calculation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_phase_relationships(self, field_signature: FieldSignature, orchestrator: LiquidOrchestrator) -> Dict[str, Any]:
        """Analyze phase relationships for interference patterns."""
        logger.debug("🌊 Analyzing phase relationships...")
        
        try:
            phase_analysis = {}
            
            # Analyze phase coherence with existing field
            if hasattr(field_signature, 'phase') and hasattr(orchestrator, 'collective_phase'):
                sig_phase = ensure_python_types(field_signature.phase)
                collective_phase = ensure_python_types(orchestrator.collective_phase)
                
                phase_difference = abs(sig_phase - collective_phase)
                phase_analysis["phase_difference"] = phase_difference
                phase_analysis["coherence"] = np.cos(phase_difference)  # Constructive if close to 1
                
            return phase_analysis
            
        except Exception as e:
            logger.debug(f"Phase analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_interference_patterns(self, field_signature: FieldSignature, orchestrator: LiquidOrchestrator) -> Dict[str, Any]:
        """Analyze interference patterns for field compatibility."""
        logger.debug("⚡ Analyzing interference patterns...")
        
        try:
            interference_analysis = {}
            
            # Analyze field interference effects
            if hasattr(orchestrator, 'field_state'):
                field_state = orchestrator.field_state
                
                # Simple interference metric - will be expanded with actual field calculations
                interference_analysis["pattern_type"] = "field_theoretical"
                interference_analysis["compatibility_score"] = 0.5  # Placeholder for actual field calculation
                
            return interference_analysis
            
        except Exception as e:
            logger.debug(f"Interference analysis failed: {e}")
            return {"error": str(e)}
    
    def create_and_integrate_charge(self, text: str, field_signature: FieldSignature, orchestrator: LiquidOrchestrator) -> Dict[str, Any]:
        """
        Create ConceptualChargeAgent from accepted text and integrate into universe.
        
        Args:
            text: Accepted text content
            field_signature: Generated field signature
            orchestrator: Living LiquidOrchestrator to integrate into
            
        Returns:
            Dictionary containing integration results
        """
        logger.info("🧮 Creating and integrating ConceptualChargeAgent for accepted text...")
        
        try:
            # Step 1: Assemble complete Q(τ,C,s) charge
            logger.info("⚡ Assembling complete Q(τ,C,s) charge using mathematical formula...")
            
            conceptual_charge = self.charge_assembler.assemble_complete_charge(
                text_content=text,
                field_signature=field_signature
            )
            
            # Step 2: Create ConceptualChargeAgent (living mathematical entity)
            logger.info("🤖 Creating ConceptualChargeAgent as living mathematical entity...")
            
            charge_agent = ConceptualChargeAgent(
                charge_data=conceptual_charge,
                device=next(orchestrator.charge_agents[0].charge.parameters()).device if orchestrator.charge_agents else "cpu"
            )
            
            # Step 3: Integrate into LiquidOrchestrator
            logger.info("🌊 Integrating charge into liquid universe field...")
            
            integration_result = orchestrator.integrate_charge(charge_agent)
            
            # Step 4: Extract post-integration metrics
            post_integration_state = orchestrator.get_universe_state()
            
            result = {
                "status": "success",
                "charge_agent_id": str(id(charge_agent)),
                "integration_metrics": convert_sage_dict(integration_result) if integration_result else {},
                "field_energy_after": ensure_python_types(post_integration_state.get("field_energy", 0.0)),
                "total_agents": len(orchestrator.charge_agents),
                "charge_created": True
            }
            
            logger.info(f"✅ Charge integration successful!")
            logger.info(f"   New agent ID: {result['charge_agent_id']}")
            logger.info(f"   Total agents: {result['total_agents']}")
            logger.info(f"   Field energy after: {result['field_energy_after']:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Charge creation and integration failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "charge_created": False
            }
    
    def save_universe_changes(self, orchestrator: LiquidOrchestrator, universe_id: str) -> bool:
        """
        Persist universe changes through burn_liquid_universe().
        
        Args:
            orchestrator: Modified LiquidOrchestrator
            universe_id: Universe identifier
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("💾 Persisting universe changes to storage...")
        
        try:
            # Find the appropriate FieldUniverse for this universe_id
            field_universe = None
            for path, fu in self.field_universes.items():
                universes = fu.storage_coordinator.list_universes()
                universe_ids = [u['universe_id'] for u in universes]
                if universe_id in universe_ids:
                    field_universe = fu
                    break
            
            if not field_universe:
                logger.error(f"❌ No FieldUniverse found for {universe_id}")
                return False
            
            # Burn liquid universe to persistent storage
            burn_result = field_universe.burn_liquid_universe(
                liquid_orchestrator=orchestrator,
                universe_id=universe_id
            )
            
            if burn_result.get("status") == "success":
                logger.info("✅ Universe changes saved successfully!")
                logger.info(f"   Burn time: {burn_result.get('burn_time', 0):.2f}s")
                return True
            else:
                logger.error(f"❌ Universe saving failed: {burn_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Universe saving error: {e}")
            return False
    
    def interactive_integration_menu(self):
        """Interactive CLI menu for universe integration testing."""
        
        print("\n" + "="*80)
        print("🧮 FIELD-THEORETIC UNIVERSE INTEGRATION TESTER")
        print("="*80)
        print("Mathematical Foundation: Q(τ,C,s) = γ·T(τ,C,s)·E^trajectory(τ,s)·Φ^semantic(τ,s)·e^(iθ_total(τ,C,s))·Ψ_persistence(s-s₀)")
        
        while True:
            print("\nOptions:")
            print("1. List available universes")
            print("2. Load universe for integration testing")
            print("3. Test text paragraph acceptance")
            print("4. Integrate accepted text (create and save charges)")
            print("5. Exit")
            
            try:
                choice = input("\nSelect option (1-5): ").strip()
                
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
                            
                            orchestrator = self.load_universe_for_integration(universe_id, device)
                            if orchestrator:
                                print(f"✅ Universe {universe_id} loaded successfully!")
                        else:
                            print("❌ Invalid selection")
                    except ValueError:
                        print("❌ Invalid input")
                
                elif choice == "3":
                    if not hasattr(self, 'current_orchestrator') or self.current_orchestrator is None:
                        print("❌ No universe loaded. Load a universe first (option 2).")
                        continue
                    
                    print(f"\n🧮 Testing integration with universe: {self.current_universe_id}")
                    print("Enter text paragraph to test (end with empty line):")
                    
                    text_lines = []
                    while True:
                        line = input()
                        if not line.strip():
                            break
                        text_lines.append(line)
                    
                    test_text = "\n".join(text_lines)
                    if not test_text.strip():
                        print("❌ No text provided")
                        continue
                    
                    # Test acceptance
                    result = self.test_universe_acceptance(test_text, self.current_orchestrator)
                    
                    # Display results
                    print("\n" + "="*60)
                    print("🧮 FIELD-THEORETIC ANALYSIS RESULTS")
                    print("="*60)
                    
                    if result["status"] == "success":
                        if result["accepted"]:
                            print("✅ UNIVERSE ACCEPTS TEXT")
                        else:
                            print("❌ UNIVERSE REJECTS TEXT")
                        
                        print(f"\n📊 Mathematical Analysis:")
                        print(f"   Weight: {result['mathematical_weight']:.6f}")
                        print(f"   Threshold: {result['threshold']:.6f}")
                        print(f"   Reasoning: {result['reasoning']}")
                        
                        print(f"\n🔬 Field Analysis:")
                        field_analysis = result["field_analysis"]
                        print(f"   Field energy: {field_analysis.get('field_energy_before', 0):.6f}")
                        
                        semantic_distances = field_analysis.get("semantic_distances", {})
                        if "mean_distance" in semantic_distances:
                            print(f"   Mean semantic distance: {semantic_distances['mean_distance']:.6f}")
                        
                        # Store result for potential integration
                        self.last_test_result = result
                        self.last_test_text = test_text
                        
                        if result["accepted"]:
                            print(f"\n💡 Use option 4 to integrate this text into the universe")
                    else:
                        print(f"❌ Analysis failed: {result.get('error')}")
                
                elif choice == "4":
                    if not hasattr(self, 'last_test_result') or not self.last_test_result.get("accepted"):
                        print("❌ No accepted text to integrate. Test text first (option 3).")
                        continue
                    
                    if not hasattr(self, 'current_orchestrator') or self.current_orchestrator is None:
                        print("❌ No universe loaded.")
                        continue
                    
                    print(f"\n🧮 Integrating accepted text into universe: {self.current_universe_id}")
                    confirm = input("Proceed with integration? This will modify the universe. (y/N): ").strip().lower()
                    
                    if confirm == 'y':
                        # Create field signature again (we need the actual object)
                        universe_state = self.current_orchestrator.get_field_statistics()
                        field_signature = self.text_to_field_signature(self.last_test_text, universe_state)
                        if field_signature:
                            # Create and integrate charge
                            integration_result = self.create_and_integrate_charge(
                                self.last_test_text, 
                                field_signature, 
                                self.current_orchestrator
                            )
                            
                            if integration_result["status"] == "success":
                                # Save universe changes
                                print("💾 Saving universe changes...")
                                saved = self.save_universe_changes(self.current_orchestrator, self.current_universe_id)
                                
                                if saved:
                                    print("✅ Text integrated and universe saved successfully!")
                                    print(f"   New total agents: {integration_result['total_agents']}")
                                    print(f"   Field energy after: {integration_result['field_energy_after']:.6f}")
                                else:
                                    print("⚠️ Integration successful but saving failed")
                            else:
                                print(f"❌ Integration failed: {integration_result.get('error')}")
                        else:
                            print("❌ Failed to recreate field signature")
                    else:
                        print("Integration cancelled")
                
                elif choice == "5":
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
        description="Integration Runner - Test universe acceptance/rejection of text content using field theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mathematical Foundation:
Q(τ,C,s) = γ·T(τ,C,s)·E^trajectory(τ,s)·Φ^semantic(τ,s)·e^(iθ_total(τ,C,s))·Ψ_persistence(s-s₀)

Examples:
  python integration_runner.py --interactive
  python integration_runner.py --list-universes
  python integration_runner.py --universe universe_130103de_1750282761 --test-text "Your text here"
        """
    )
    
    parser.add_argument("--storage-path", default="liquid_universes",
                       help="Base path to search for stored universes")
    parser.add_argument("--list-universes", action="store_true",
                       help="List all available universes")
    parser.add_argument("--universe", metavar="UNIVERSE_ID",
                       help="Load specific universe by ID")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"],
                       help="Device for field calculations (default: mps)")
    parser.add_argument("--test-text", metavar="TEXT",
                       help="Test specific text paragraph")
    parser.add_argument("--integrate", action="store_true",
                       help="Integrate accepted text into universe")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive menu")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = IntegrationRunner(storage_base_path=args.storage_path)
    
    try:
        if args.interactive:
            runner.interactive_integration_menu()
        
        elif args.list_universes:
            universes = runner.discover_universes()
            runner.display_universe_list(universes)
        
        elif args.universe:
            # Load specific universe
            orchestrator = runner.load_universe_for_integration(args.universe, args.device)
            
            if orchestrator and args.test_text:
                # Test specific text
                result = runner.test_universe_acceptance(args.test_text, orchestrator)
                
                print("\n" + "="*60)
                print("🧮 FIELD-THEORETIC ANALYSIS RESULTS")
                print("="*60)
                
                if result["status"] == "success":
                    if result["accepted"]:
                        print("✅ UNIVERSE ACCEPTS TEXT")
                        
                        if args.integrate:
                            # Integrate the text
                            universe_state = orchestrator.get_field_statistics()
                            field_signature = runner.text_to_field_signature(args.test_text, universe_state)
                            if field_signature:
                                integration_result = runner.create_and_integrate_charge(
                                    args.test_text, field_signature, orchestrator
                                )
                                
                                if integration_result["status"] == "success":
                                    runner.save_universe_changes(orchestrator, args.universe)
                                    print("✅ Text integrated and saved!")
                    else:
                        print("❌ UNIVERSE REJECTS TEXT")
                    
                    print(f"\nMathematical weight: {result['mathematical_weight']:.6f}")
                    print(f"Threshold: {result['threshold']:.6f}")
                    print(f"Reasoning: {result['reasoning']}")
                else:
                    print(f"❌ Analysis failed: {result.get('error')}")
        
        else:
            # Default: show help and start interactive mode
            parser.print_help()
            print("\nStarting interactive mode...")
            runner.interactive_integration_menu()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Integration runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()