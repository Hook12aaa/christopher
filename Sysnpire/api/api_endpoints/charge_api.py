"""
Charge API - Commercial REST API for conceptual charge creation

Fast, scalable API for creating conceptual charges from text inputs.
"""

from flask import Flask, request, jsonify
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model import ChargeFactory

class ChargeAPI:
    """Commercial API for conceptual charge creation."""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.factory = ChargeFactory()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'service': 'charge_api',
                'timestamp': time.time()
            })
        
        @self.app.route('/charge/create', methods=['POST'])
        def create_charge():
            """Create a single conceptual charge from text."""
            try:
                data = request.get_json()
                
                # Validate input
                if 'text' not in data:
                    return jsonify({'error': 'Missing required field: text'}), 400
                
                text = data['text']
                observational_state = data.get('observational_state', 1.0)
                gamma = data.get('gamma', 1.0)
                
                # Create charge
                result = self.factory.create_charge(text, observational_state, gamma)
                
                # Format response for commercial use
                response = {
                    'success': True,
                    'data': {
                        'text': result['text'],
                        'charge_id': f"charge_{int(result['timestamp'])}",
                        'field_values': {
                            'magnitude': float(result['field_values']['magnitude']),
                            'phase': float(result['field_values']['phase']),
                            'real_part': float(result['field_values']['complete_charge'].real),
                            'imaginary_part': float(result['field_values']['complete_charge'].imag)
                        },
                        'parameters': {
                            'observational_state': observational_state,
                            'gamma': gamma
                        },
                        'processing_time': result['processing_time'],
                        'timestamp': result['timestamp']
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }), 500
        
        @self.app.route('/charge/batch', methods=['POST'])
        def create_charges_batch():
            """Create multiple conceptual charges from text list."""
            try:
                data = request.get_json()
                
                # Validate input
                if 'texts' not in data:
                    return jsonify({'error': 'Missing required field: texts'}), 400
                
                texts = data['texts']
                observational_states = data.get('observational_states')
                gamma_values = data.get('gamma_values')
                
                # Create charges
                results = self.factory.create_charges_batch(
                    texts, observational_states, gamma_values
                )
                
                # Format response
                charges = []
                for i, result in enumerate(results):
                    charges.append({
                        'charge_id': f"charge_{int(result['timestamp'])}_{i}",
                        'text': result['text'],
                        'field_values': {
                            'magnitude': float(result['field_values']['magnitude']),
                            'phase': float(result['field_values']['phase']),
                            'real_part': float(result['field_values']['complete_charge'].real),
                            'imaginary_part': float(result['field_values']['complete_charge'].imag)
                        },
                        'processing_time': result['processing_time']
                    })
                
                # Analyze universe
                analysis = self.factory.analyze_charge_universe(results)
                
                response = {
                    'success': True,
                    'data': {
                        'charges': charges,
                        'universe_analysis': {
                            'total_charges': analysis['num_charges'],
                            'total_energy': analysis['total_energy'],
                            'average_magnitude': analysis['average_magnitude'],
                            'processing_rate': analysis['charges_per_second']
                        },
                        'timestamp': time.time()
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }), 500
        
        @self.app.route('/info', methods=['GET'])
        def api_info():
            """API information and capabilities."""
            return jsonify({
                'service': 'Field Theory Charge API',
                'version': '1.0.0',
                'capabilities': [
                    'Single charge creation from text',
                    'Batch charge processing',
                    'Field-theoretic analysis',
                    'Commercial-grade scaling'
                ],
                'endpoints': {
                    'POST /charge/create': 'Create single conceptual charge',
                    'POST /charge/batch': 'Create multiple charges',
                    'GET /health': 'Health check',
                    'GET /info': 'API information'
                },
                'timestamp': time.time()
            })
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the API server."""
        print(f"🚀 Starting Charge API server...")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Debug: {debug}")
        
        self.app.run(host=host, port=port, debug=debug)

def create_app():
    """Create Flask app for deployment."""
    api = ChargeAPI()
    return api.app

if __name__ == "__main__":
    api = ChargeAPI()
    api.run(debug=True)