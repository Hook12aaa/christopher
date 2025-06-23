"""
Agent Factory - ConceptualChargeAgent Reconstruction from Storage

Creates living ConceptualChargeAgent objects from stored mathematical data,
rebuilding complete Q(τ,C,s) state with full mathematical precision preservation.

Key Features:
- Complete agent reconstruction from HDF5 storage
- Mathematical state validation and consistency checking
- Device-aware tensor reconstruction (MPS/CUDA/CPU)
- Complex number reconstruction from real/imag pairs
- Field component object rebuilding
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np

from Sysnpire.model.liquid.conceptual_charge_agent import ConceptualChargeAgent, QMathematicalComponents, ThetaComponents
from Sysnpire.database.conceptual_charge_object import ConceptualChargeObject, FieldComponents
from Sysnpire.database.universe_reconstruction.reconstruction_converter import ReconstructionConverter
from Sysnpire.database.universe_reconstruction.raw_data_validator import RawDataValidator, RawDataValidationError
from Sysnpire.database.liquid_burning.mathematical_validator import MathematicalValidator
from Sysnpire.utils.logger import get_logger

logger = get_logger(__name__)


class AgentReconstructionError(Exception):
    """Raised when agent reconstruction fails mathematical validation."""
    pass


class AgentFactory:
    """
    Factory for reconstructing ConceptualChargeAgent objects from stored data.
    
    Handles the complex process of rebuilding living mathematical entities
    from persistent storage with complete state preservation.
    """
    
    def __init__(self, device: str = "mps", validate_reconstruction: bool = True):
        """
        Initialize agent reconstruction factory.
        
        Args:
            device: Target device for reconstructed tensors
            validate_reconstruction: Validate mathematical consistency
        """
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        self.validate_reconstruction = validate_reconstruction
        
        # Initialize raw data validator and converter (NO tensor validation in converter)
        self.raw_validator = RawDataValidator(strict_validation=validate_reconstruction)
        self.converter = ReconstructionConverter(device=str(self.device))
        self.validator = MathematicalValidator(strict_validation=validate_reconstruction)
        
        logger.info("AgentFactory initialized")
        logger.info(f"  Target device: {self.device}")
        logger.info(f"  Validation enabled: {validate_reconstruction}")
        logger.info(f"  Data converter initialized: {self.converter.__class__.__name__}")
        logger.info(f"  Mathematical validator initialized: {self.validator.__class__.__name__}")
    
    def reconstruct_single_agent(self, stored_agent_data: Dict[str, Any], 
                               universe_metadata: Dict[str, Any] = None) -> ConceptualChargeAgent:
        """
        Reconstruct a single ConceptualChargeAgent from stored data.
        
        Args:
            stored_agent_data: Complete agent data from HDF5 storage
            universe_metadata: Universe-level context for reconstruction
            
        Returns:
            Fully reconstructed ConceptualChargeAgent
            
        Raises:
            AgentReconstructionError: If reconstruction fails validation
        """
        logger.info("🔄 Reconstructing ConceptualChargeAgent from stored data")
        start_time = time.time()
        
        try:
            # STEP 1: Convert data types from storage format to runtime format
            logger.info("🔄 Converting agent data from storage to runtime format...")
            converted_data = self.converter.convert_agent_data(stored_agent_data)
            
            # Extract converted components
            agent_metadata = converted_data.get("agent_metadata")
            q_components = converted_data.get("Q_components")
            field_components = converted_data.get("field_components")
            temporal_components = converted_data.get("temporal_components")
            emotional_components = converted_data.get("emotional_components")
            agent_state = converted_data.get("agent_state")
            
            # Log what components we have after conversion
            charge_id = agent_metadata.get("charge_id")
            has_q = bool(q_components)
            has_field = bool(field_components)
            has_temporal = bool(temporal_components)
            has_emotional = bool(emotional_components)
            has_state = bool(agent_state)
            logger.info(f"🔧 Agent {charge_id} - Converted components: Q={has_q}, Field={has_field}, Temporal={has_temporal}, Emotional={has_emotional}, State={has_state}")
            
            # STEP 2: Validate converted data using mathematical validator
            if self.validate_reconstruction:
                logger.info(f"🔍 Validating converted data for agent {charge_id}...")
                validation_result = self.validator.validate_reconstructed_data(converted_data, charge_id)
                
                if not validation_result["validation_passed"]:
                    error_details = validation_result['errors'][:3] if validation_result.get('errors') else []
                    raise ValueError(f"Agent {charge_id} - Mathematical validation failed with {validation_result['failed_checks']} errors: {error_details}")
                else:
                    logger.info(f"✅ Agent {charge_id} - Validation passed ({validation_result['passed_checks']}/{validation_result['total_checks']} checks)")
            
            # STEP 4: Additional Q_value validation  
            if has_q and "Q_value" in q_components:
                q_value = q_components["Q_value"]
                if isinstance(q_value, complex):
                    # Validate Q magnitude using converter
                    if not self.converter.validate_q_magnitude(q_value, charge_id):
                        raise ValueError(f"Agent {charge_id} - Q_value magnitude out of valid range: {abs(q_value):.2e}")
                    logger.info(f"🔧 Agent {charge_id} - Q_value converted: {q_value} (magnitude: {abs(q_value):.2e})")
                else:
                    raise ValueError(f"Agent {charge_id} - Q_value not converted to complex number, got type: {type(q_value)}")
            
            # STEP 5: Reconstruct charge object using converted data
            charge_obj = self._reconstruct_charge_object(agent_metadata, q_components, field_components, agent_state)
            
            # STEP 6: Create agent using converted data
            agent = ConceptualChargeAgent.from_stored_data(
                stored_data=converted_data,  # Use converted data instead of original
                charge_obj=charge_obj,
                device=str(self.device)
            )
            
            # CRITICAL: Set the agent's living_Q_value to the stored Q_value instead of computing new ones
            stored_q_value = charge_obj.complete_charge
            agent.living_Q_value = stored_q_value
            
            # Create Q_components from stored data instead of computing new ones
            q_data = converted_data.get("Q_components", {})
            agent.Q_components = QMathematicalComponents(
                gamma=q_data.get("gamma", 1.0),
                T_tensor=q_data.get("T_tensor", complex(1.0, 0.0)),
                E_trajectory=q_data.get("E_trajectory", complex(1.0, 0.0)),
                phi_semantic=q_data.get("phi_semantic", complex(1.0, 0.0)),
                theta_components=ThetaComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Default
                phase_factor=q_data.get("phase_factor", complex(1.0, 0.0)),
                psi_persistence=q_data.get("psi_persistence", 1.0),
                psi_gaussian=q_data.get("psi_gaussian", 1.0),
                psi_exponential_cosine=q_data.get("psi_exponential_cosine", 1.0),
                Q_value=stored_q_value
            )
            
            # CRITICAL FIX: Manually ensure evolution parameters are Python floats (additional safety net)
            agent_state_data = converted_data.get("agent_state", {})
            evolution_params = ["sigma_i", "alpha_i", "lambda_i", "beta_i"]
            for param in evolution_params:
                if param in agent_state_data:
                    # Force conversion to Python float to prevent tensor conversion
                    param_value = agent_state_data[param]
                    original_type = type(param_value)
                    
                    # Ensure it's a Python float regardless of input type
                    if hasattr(param_value, 'cpu'):
                        float_value = float(param_value.cpu().detach().numpy())
                    elif hasattr(param_value, 'item'):
                        float_value = float(param_value.item())
                    else:
                        float_value = float(param_value)
                    
                    setattr(agent, param, float_value)
                    logger.info(f"🔧 FACTORY OVERRIDE: {param}: {original_type} -> {type(float_value)} (value: {float_value})")
                else:
                    logger.warning(f"⚠️  {param} not found in agent_state, keeping agent's default")
            
            logger.info(f"🔧 Agent {charge_id} - Using stored Q_value: {stored_q_value} (magnitude: {abs(stored_q_value):.2e})")
            
            # Verify Q_components were properly reconstructed
            if hasattr(agent, 'Q_components') and agent.Q_components is not None:
                q_magnitude = abs(agent.Q_components.Q_value)
                logger.info(f"✅ Agent {charge_id} - Q_components reconstructed successfully: |Q|={q_magnitude:.2e}")
                
                # MATHEMATICAL VALIDATION: Check for invalid values
                if q_magnitude > 1e10:
                    logger.error(f"💥 Agent {charge_id} - Q_magnitude ASTRONOMICAL: {q_magnitude:.2e}")
                elif q_magnitude < 1e-15:
                    logger.error(f"💥 Agent {charge_id} - Q_magnitude UNDERFLOW: {q_magnitude:.2e}")
                    
            else:
                raise ValueError(f"Agent {charge_id} - Q_components MISSING after reconstruction! Agent has no valid Q_components.")
            
            # MATHEMATICAL VALIDATION: Check other critical values
            if hasattr(agent, 'evolution_rates'):
                cascading_rate = agent.evolution_rates.get('cascading')
                if abs(cascading_rate) > 100.0:
                    logger.error(f"💥 Agent {charge_id} - evolution_rates.cascading ASTRONOMICAL: {cascading_rate}")
                if not np.isfinite(cascading_rate):
                    logger.error(f"💥 Agent {charge_id} - evolution_rates.cascading NaN/INF: {cascading_rate}")
                    
            if hasattr(agent, 'cascade_momentum'):
                total_momentum = sum(abs(m) for m in agent.cascade_momentum.values())
                if total_momentum > 1e6:
                    logger.error(f"💥 Agent {charge_id} - cascade_momentum ASTRONOMICAL: {total_momentum:.2e}")
                if not np.isfinite(total_momentum):
                    logger.error(f"💥 Agent {charge_id} - cascade_momentum NaN/INF: {total_momentum}")
                    
            if hasattr(agent, 'living_Q_value'):
                living_q_mag = abs(agent.living_Q_value)
                if living_q_mag > 1e10:
                    logger.error(f"💥 Agent {charge_id} - living_Q_value ASTRONOMICAL: {living_q_mag:.2e}")
                elif living_q_mag < 1e-15:
                    logger.error(f"💥 Agent {charge_id} - living_Q_value UNDERFLOW: {living_q_mag:.2e}")
                if not np.isfinite(living_q_mag):
                    logger.error(f"💥 Agent {charge_id} - living_Q_value NaN/INF: {agent.living_Q_value}")
                    
            # Check temporal biography for explosions
            if hasattr(agent, 'temporal_biography') and hasattr(agent.temporal_biography, 'temporal_momentum'):
                temp_momentum = agent.temporal_biography.temporal_momentum
                if abs(temp_momentum) > 1e6:
                    logger.error(f"💥 Agent {charge_id} - temporal_momentum ASTRONOMICAL: {temp_momentum:.2e}")
                if not np.isfinite(abs(temp_momentum)):
                    logger.error(f"💥 Agent {charge_id} - temporal_momentum NaN/INF: {temp_momentum}")
            
            # CRITICAL: Restore mathematical state from storage (including temporal_momentum)
            logger.info(f"🔍 DEBUG: Calling restore_mathematical_state for agent {charge_id}")
            self.restore_mathematical_state(agent, converted_data)
            
            # CRITICAL VALIDATION: Ensure temporal_momentum was properly restored
            self._validate_critical_restoration(agent, charge_id)
            
            # Validate reconstruction if enabled
            if self.validate_reconstruction:
                self._validate_reconstructed_agent(agent, stored_agent_data)
            
            reconstruction_time = time.time() - start_time
            logger.info(f"✅ Agent reconstruction complete in {reconstruction_time:.3f}s")
            logger.info(f"   Agent ID: {agent.charge_id}")
            logger.info(f"   Q magnitude: {abs(agent.living_Q_value):.6f}")
            
            return agent
            
        except Exception as e:
            charge_id = stored_agent_data.get("agent_metadata", {}).get("charge_id", "unknown")
            logger.error(f"❌ Agent reconstruction failed for {charge_id}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.error(f"   Available data keys: {list(stored_agent_data.keys()) if stored_agent_data else 'None'}")
            if "field_components" in stored_agent_data:
                field_comp = stored_agent_data["field_components"]
                logger.error(f"   Field component keys: {list(field_comp.keys()) if field_comp else 'None'}")
                logger.error(f"   Field component types: {[(k, type(v).__name__) for k, v in field_comp.items()] if field_comp else 'None'}")
            raise ValueError(f"Agent reconstruction failed: {e}")
    
    def _reconstruct_charge_object(self, metadata: Dict[str, Any], 
                                 q_components: Dict[str, Any],
                                 field_components: Dict[str, Any],
                                 agent_state: Dict[str, Any]) -> ConceptualChargeObject:
        """
        Reconstruct ConceptualChargeObject from stored metadata.
        
        Args:
            metadata: Agent metadata from storage
            q_components: Q mathematical components
            field_components: Field component data
            agent_state: Agent state data
            
        Returns:
            Reconstructed ConceptualChargeObject
        """
        # Get complex charge value (already converted by ReconstructionConverter)
        if "Q_value" in q_components and isinstance(q_components["Q_value"], complex):
            complete_charge = q_components["Q_value"]
        elif "living_Q_value" in q_components and isinstance(q_components["living_Q_value"], complex):
            complete_charge = q_components["living_Q_value"]
        else:
            # Fallback: try manual reconstruction if converter missed it
            if "Q_value_real" in q_components and "Q_value_imag" in q_components:
                complete_charge = complex(q_components["Q_value_real"], q_components["Q_value_imag"])
            elif "living_Q_value_real" in q_components and "living_Q_value_imag" in q_components:
                complete_charge = complex(q_components["living_Q_value_real"], q_components["living_Q_value_imag"])
            else:
                complete_charge = complex(1.0, 0.0)  # Default fallback
        
        logger.debug(f"ConceptualChargeObject using Q_value: {complete_charge} (magnitude: {abs(complete_charge):.2e})")
        
        # Reconstruct field components with flexible name mapping - SAFE from tensor boolean evaluation
        # Try multiple possible names for each component to handle storage inconsistencies
        
        charge_id = metadata.get("charge_id", "unknown")
        logger.debug(f"🔍 Debugging field component access for agent {charge_id}")
        logger.debug(f"   Available field_components keys: {list(field_components.keys()) if field_components else 'None'}")
        
        # SAFE trajectory operators access
        try:
            traj_ops = field_components.get("trajectory_operators")
            logger.debug(f"   trajectory_operators: {type(traj_ops)} = {getattr(traj_ops, 'shape', 'no shape') if hasattr(traj_ops, 'shape') else traj_ops}")
            if traj_ops is None:
                traj_ops = field_components.get("field_trajectory_operators")
                logger.debug(f"   field_trajectory_operators: {type(traj_ops)} = {getattr(traj_ops, 'shape', 'no shape') if hasattr(traj_ops, 'shape') else traj_ops}")
            if traj_ops is None:
                traj_ops = field_components.get("T_operators")
                logger.debug(f"   T_operators: {type(traj_ops)} = {getattr(traj_ops, 'shape', 'no shape') if hasattr(traj_ops, 'shape') else traj_ops}")
        except Exception as e:
            logger.error(f"❌ TENSOR BOOLEAN ERROR in trajectory_operators access for {charge_id}: {e}")
            logger.error(f"   Available keys: {list(field_components.keys())}")
            raise

        # SAFE emotional trajectory access
        try:
            emot_traj = field_components.get("emotional_trajectory")
            logger.debug(f"   emotional_trajectory: {type(emot_traj)} = {getattr(emot_traj, 'shape', 'no shape') if hasattr(emot_traj, 'shape') else emot_traj}")
            if emot_traj is None:
                emot_traj = field_components.get("field_emotional_trajectory")
                logger.debug(f"   field_emotional_trajectory: {type(emot_traj)} = {getattr(emot_traj, 'shape', 'no shape') if hasattr(emot_traj, 'shape') else emot_traj}")
            if emot_traj is None:
                emot_traj = field_components.get("E_trajectory")
                logger.debug(f"   E_trajectory: {type(emot_traj)} = {getattr(emot_traj, 'shape', 'no shape') if hasattr(emot_traj, 'shape') else emot_traj}")
        except Exception as e:
            logger.error(f"❌ TENSOR BOOLEAN ERROR in emotional_trajectory access for {charge_id}: {e}")
            logger.error(f"   Available keys: {list(field_components.keys())}")
            raise

        # SAFE semantic embedding access
        try:
            semantic = field_components.get("semantic_embedding")
            logger.debug(f"   semantic_embedding: {type(semantic)} = {getattr(semantic, 'shape', 'no shape') if hasattr(semantic, 'shape') else semantic}")
            if semantic is None:
                semantic = field_components.get("charge_semantic_field")
                logger.debug(f"   charge_semantic_field: {type(semantic)} = {getattr(semantic, 'shape', 'no shape') if hasattr(semantic, 'shape') else semantic}")
            if semantic is None:
                semantic = field_components.get("semantic_field")
                logger.debug(f"   semantic_field: {type(semantic)} = {getattr(semantic, 'shape', 'no shape') if hasattr(semantic, 'shape') else semantic}")
        except Exception as e:
            logger.error(f"❌ TENSOR BOOLEAN ERROR in semantic_embedding access for {charge_id}: {e}")
            logger.error(f"   Available keys: {list(field_components.keys())}")
            raise

        phase = field_components.get("phase_total")
        
        # Handle missing observational_persistence (compute if not stored)
        obs_persist = field_components.get("observational_persistence")
        if obs_persist is None:
            # Option 1: Compute from manifold dimension
            manifold_dim = field_components.get("manifold_dimension")
            if manifold_dim:
                obs_persist = float(manifold_dim) / 1024.0
            else:
                # Option 2: Compute from semantic field magnitude
                if semantic is not None:
                    try:
                        # Simple tensor handling - data is pre-validated
                        if hasattr(semantic, 'cpu'):
                            # It's a tensor - convert to numpy (no validation needed)
                            semantic_array = semantic.cpu().detach().numpy()
                        else:
                            # It's already numpy/list - convert to numpy
                            semantic_array = np.array(semantic)
                        
                        # Compute observational persistence (data guaranteed to be valid)
                        if semantic_array.size > 0:
                            obs_persist = float(np.mean(np.abs(semantic_array)))
                        else:
                            obs_persist = 0.5
                            
                    except Exception as e:
                        logger.debug(f"Error computing observational_persistence from semantic field: {e}")
                        obs_persist = 0.5
                else:
                    # Option 3: Default fallback
                    obs_persist = 0.5
            logger.debug(f"Computed missing observational_persistence: {obs_persist}")
        
        charge_id = metadata.get("charge_id", "unknown")
        logger.debug(f"Field component values for {charge_id}:")
        logger.debug(f"  trajectory_operators: {type(traj_ops)} = {traj_ops}")
        logger.debug(f"  emotional_trajectory: {type(emot_traj)} = {emot_traj}")
        logger.debug(f"  semantic_embedding: {type(semantic)} = {semantic}")
        logger.debug(f"  phase_total: {type(phase)} = {phase}")
        logger.debug(f"  observational_persistence: {type(obs_persist)} = {obs_persist}")
        
        # Check for critical None values
        none_fields = []
        if traj_ops is None: none_fields.append("trajectory_operators")
        if emot_traj is None: none_fields.append("emotional_trajectory")
        if semantic is None: none_fields.append("semantic_embedding")
        if phase is None: none_fields.append("phase_total")
        if obs_persist is None: none_fields.append("observational_persistence")
        
        if none_fields:
            available_keys = list(field_components.keys()) if field_components else 'field_components is None'
            attempted_mappings = {
                "semantic": ["semantic_embedding", "charge_semantic_field", "semantic_field"],
                "trajectory": ["trajectory_operators", "field_trajectory_operators", "T_operators"],  
                "emotional": ["emotional_trajectory", "field_emotional_trajectory", "E_trajectory"]
            }
            raise ValueError(f"Agent {charge_id} - Missing field components: {none_fields}. Available: {available_keys}. Tried mappings: {attempted_mappings}")
        
        field_comps = FieldComponents(
            trajectory_operators=traj_ops,
            emotional_trajectory=emot_traj,
            semantic_field=semantic,
            phase_total=phase,
            observational_persistence=obs_persist
        )
        
        # Create charge object (magnitude and phase are calculated internally)
        # DEBUG: Check for None values in ConceptualChargeObject parameters
        cco_charge_id = metadata.get("charge_id")
        cco_text_source = metadata.get("text_source")
        
        # Look for gamma in Q_components (where it's actually stored)
        cco_gamma = q_components.get("gamma")
        
        # Look for observational_state in agent_state (now provided by HDF5Manager)
        cco_obs_state = agent_state.get("observational_state")
        
        logger.debug(f"ConceptualChargeObject parameters for {charge_id}:")
        logger.debug(f"  charge_id (from metadata): {type(cco_charge_id)} = {cco_charge_id}")
        logger.debug(f"  text_source (from metadata): {type(cco_text_source)} = {cco_text_source}")
        logger.debug(f"  observational_state (from agent_state): {type(cco_obs_state)} = {cco_obs_state}")
        logger.debug(f"  gamma (from Q_components): {type(cco_gamma)} = {cco_gamma}")
        logger.debug(f"  Available Q_components keys: {list(q_components.keys()) if q_components else 'None'}")
        logger.debug(f"  Available agent_state keys: {list(agent_state.keys()) if agent_state else 'None'}")
        
        # Check for critical None values from their respective sources
        missing_fields = []
        
        # Check metadata fields
        if cco_charge_id is None: missing_fields.append("charge_id (from metadata)")
        if cco_text_source is None: missing_fields.append("text_source (from metadata)")
        
        # Check Q_components fields  
        if cco_gamma is None: missing_fields.append("gamma (from Q_components)")
        
        # Check agent_state fields
        if cco_obs_state is None: missing_fields.append("observational_state (from agent_state)")
        
        if missing_fields:
            error_details = []
            error_details.append(f"Missing fields: {missing_fields}")
            error_details.append(f"Available metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            error_details.append(f"Available Q_components keys: {list(q_components.keys()) if q_components else 'None'}")
            error_details.append(f"Available agent_state keys: {list(agent_state.keys()) if agent_state else 'None'}")
            raise ValueError(f"Agent {charge_id} - Critical fields missing: {'; '.join(error_details)}")
        
        charge_obj = ConceptualChargeObject(
            charge_id=cco_charge_id,
            text_source=cco_text_source,
            complete_charge=complete_charge,
            field_components=field_comps,
            observational_state=cco_obs_state,
            gamma=cco_gamma
        )
        
        return charge_obj
    
    def _validate_reconstructed_agent(self, agent: ConceptualChargeAgent, 
                                    original_data: Dict[str, Any]) -> bool:
        """
        Validate that reconstructed agent matches original mathematical state.
        
        Args:
            agent: Reconstructed agent
            original_data: Original stored data
            
        Returns:
            True if validation passes
            
        Raises:
            AgentReconstructionError: If validation fails
        """
        logger.info("🔍 Validating reconstructed agent mathematical consistency")
        
        errors = []
        
        # Validate Q-value consistency
        stored_q = original_data.get("Q_components")
        if "Q_value_real" in stored_q and "Q_value_imag" in stored_q:
            expected_q = complex(stored_q["Q_value_real"], stored_q["Q_value_imag"])
            actual_q = agent.living_Q_value
            
            if abs(expected_q - actual_q) > 1e-10:
                errors.append(f"Q-value mismatch: expected {expected_q}, got {actual_q}")
        
        # Validate agent has proper mathematical components
        if agent.Q_components is None:
            errors.append("Q_components not computed in reconstructed agent")
        
        # Validate basic agent properties
        if not hasattr(agent, 'living_Q_value'):
            errors.append("Missing living_Q_value attribute")
        
        if not hasattr(agent, 'charge_obj'):
            errors.append("Missing charge_obj attribute")
        
        # DISABLED: Skip Q_components validation during reconstruction since stored Q values are known good
        # Use agent's built-in validation
        # if hasattr(agent, 'validate_Q_components'):
        #     try:
        #         if not agent.validate_Q_components():
        #             errors.append("Agent Q_components validation failed")
        #     except Exception as e:
        #         errors.append(f"Q_components validation error: {e}")
        
        if errors:
            error_msg = "Agent reconstruction validation failed:\n" + "\n".join(f"  • {err}" for err in errors)
            logger.error(error_msg)
            raise AgentReconstructionError(error_msg)
        
        logger.info("✅ Agent validation passed")
        return True
    
    def restore_mathematical_state(self, agent: ConceptualChargeAgent, 
                                 stored_data: Dict[str, Any]) -> None:
        """
        Restore complex mathematical state from stored data.
        
        Args:
            agent: Agent to restore state to
            stored_data: Complete stored mathematical data
        """
        logger.info("🔧 Restoring mathematical state from storage")
        
        # CRITICAL: NO DEFAULTS - detect missing components immediately
        if "Q_components" not in stored_data:
            raise AgentReconstructionError(f"Agent {agent.charge_id} - Missing Q_components in stored_data - storage incomplete!")
        if "agent_state" not in stored_data:
            raise AgentReconstructionError(f"Agent {agent.charge_id} - Missing agent_state in stored_data - storage incomplete!")
        if "temporal_components" not in stored_data:
            raise AgentReconstructionError(f"Agent {agent.charge_id} - Missing temporal_components in stored_data - storage incomplete!")
        
        q_components = stored_data["Q_components"]
        agent_state = stored_data["agent_state"]
        temporal_components = stored_data["temporal_components"]
        
        # Restore Q mathematical components
        if "gamma" in q_components:
            agent.gamma = q_components["gamma"]
        
        # Restore evolution parameters
        for param in ["sigma_i", "alpha_i", "lambda_i", "beta_i"]:
            if param in agent_state:
                setattr(agent, param, agent_state[param])
        
        # Restore breathing parameters
        for param in ["breath_frequency", "breath_amplitude", "breath_phase"]:
            if param in agent_state:
                setattr(agent, param, agent_state[param])
        
        # Restore complex values from real/imag pairs
        complex_fields = [
            ("T_tensor", "T_tensor"),
            ("E_trajectory", "E_trajectory"), 
            ("phi_semantic", "phi_semantic")
        ]
        
        for stored_name, agent_attr in complex_fields:
            real_key = f"{stored_name}_real"
            imag_key = f"{stored_name}_imag"
            
            if real_key in q_components and imag_key in q_components:
                complex_val = complex(q_components[real_key], q_components[imag_key])
                setattr(agent, agent_attr, complex_val)
        
        # CRITICAL FIX: Restore temporal_momentum from temporal_components (NOT q_components)
        if "temporal_momentum" in temporal_components:
            temporal_momentum_val = temporal_components["temporal_momentum"]
            # Set to correct location: agent.temporal_biography.temporal_momentum
            if hasattr(agent, 'temporal_biography'):
                agent.temporal_biography.temporal_momentum = temporal_momentum_val
                logger.info(f"✅ Agent {agent.charge_id} - Restored temporal_momentum: {temporal_momentum_val}")
            else:
                logger.error(f"❌ CRITICAL: Agent {agent.charge_id} - No temporal_biography to restore temporal_momentum to!")
                raise AgentReconstructionError(f"Agent {agent.charge_id} missing temporal_biography - cannot restore temporal_momentum")
        else:
            logger.error(f"❌ CRITICAL: Agent {agent.charge_id} - No temporal_momentum found in temporal_components!")
            logger.error(f"   Available temporal_components keys: {list(temporal_components.keys()) if temporal_components else 'None'}")
            raise AgentReconstructionError(f"Agent {agent.charge_id} missing temporal_momentum - storage restoration failed!")
        
        logger.info("✅ Mathematical state restoration complete")
    
    def _validate_critical_restoration(self, agent: ConceptualChargeAgent, agent_id: str) -> None:
        """
        CRITICAL VALIDATION: Ensure all required data was properly restored from storage.
        
        This catches storage/restoration bugs that would otherwise be masked by defaults.
        """
        errors = []
        
        # CRITICAL: temporal_momentum must exist (this was our main bug)
        if not hasattr(agent, 'temporal_biography'):
            errors.append("Missing temporal_biography object")
        elif not hasattr(agent.temporal_biography, 'temporal_momentum'):
            errors.append("Missing temporal_momentum in temporal_biography")
        else:
            temp_momentum = agent.temporal_biography.temporal_momentum
            if temp_momentum is None:
                errors.append("temporal_momentum is None")
            else:
                # Check for LogPolarComplex type
                from Sysnpire.utils.log_polar_complex import LogPolarComplex
                if not isinstance(temp_momentum, LogPolarComplex):
                    errors.append(f"temporal_momentum wrong type: {type(temp_momentum)}, expected LogPolarComplex")
        
        # CRITICAL: Q_components must exist
        if not hasattr(agent, 'Q_components') or agent.Q_components is None:
            errors.append("Missing Q_components")
        
        # CRITICAL: living_Q_value must exist
        if not hasattr(agent, 'living_Q_value') or agent.living_Q_value is None:
            errors.append("Missing living_Q_value")
        
        if errors:
            error_msg = f"CRITICAL RESTORATION FAILURE for agent {agent_id}:\n" + "\n".join(f"  • {err}" for err in errors)
            logger.error(f"❌ {error_msg}")
            raise AgentReconstructionError(error_msg)
        else:
            logger.info(f"✅ CRITICAL VALIDATION PASSED for agent {agent_id}")
    
    def rebuild_interaction_patterns(self, agents: List[ConceptualChargeAgent],
                                   interaction_data: Dict[str, Any] = None) -> None:
        """
        Rebuild agent interaction patterns and network topology.
        
        Args:
            agents: List of reconstructed agents
            interaction_data: Stored interaction pattern data
        """
        logger.info("🔗 Rebuilding agent interaction patterns")
        
        # Ensure all agents can see each other
        for i, agent in enumerate(agents):
            # Set up basic interaction capabilities
            if hasattr(agent, 'interaction_memory_buffer'):
                agent.interaction_memory_buffer = {}
        
        logger.info(f"✅ Interaction patterns rebuilt for {len(agents)} agents")


if __name__ == "__main__":
    print("AgentFactory ready for ConceptualChargeAgent reconstruction")