"""
Main entry point for the Constructivist Field Theory API.
Uses Quart for asynchronous web server functionality with API key authentication.
"""
from quart import Quart, request, jsonify
from functools import wraps
import uuid
import os
import asyncio

# Create Quart app
app = Quart(__name__)
app.config["TITLE"] = "Constructivist Field Theory API"
app.config["DESCRIPTION"] = "API for the Constructivist Mathematics implementation"
app.config["VERSION"] = "0.1.0"

# API key authentication setup
API_KEYS = {
    "development": "dev_key_" + str(uuid.uuid4())[:8],  # Generate a development key
}

# Add an environment variable API key if provided
if os.environ.get("CFT_API_KEY"):
    API_KEYS["production"] = os.environ.get("CFT_API_KEY")

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key is None:
            return jsonify({"error": "API key is missing"}), 401
        if api_key not in API_KEYS.values():
            return jsonify({"error": "Invalid API key"}), 401
        return await f(*args, **kwargs)
    return decorated_function

@app.route("/")
async def root():
    """Root endpoint returning API status and information."""
    return {
        "status": "online",
        "api_name": "Constructivist Field Theory API",
        "version": app.config["VERSION"],
        "documentation": "/docs",
    }

@app.route("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.route("/api/auth/generate_key", methods=["POST"])
@require_api_key
async def generate_api_key():
    """Generate a new API key (requires an existing valid key)."""
    key_name = (await request.get_json()).get("name", f"key_{len(API_KEYS)}")
    new_key = f"cft_key_{str(uuid.uuid4())}"
    API_KEYS[key_name] = new_key
    return {"key_name": key_name, "api_key": new_key}

@app.route("/api/auth/list_keys", methods=["GET"])
@require_api_key
async def list_api_keys():
    """List all API keys (for admin purposes)."""
    # In production, you might want to hide the actual keys
    return {"keys": list(API_KEYS.keys())}

# Import and register blueprints
from api.routers import embedding, charges, fields, resonance, viz

# Register all blueprints with their respective URL prefixes
app.register_blueprint(embedding.blueprint, url_prefix="/api/embedding")
app.register_blueprint(charges.blueprint, url_prefix="/api/charges")
app.register_blueprint(fields.blueprint, url_prefix="/api/fields")
app.register_blueprint(resonance.blueprint, url_prefix="/api/resonance")
app.register_blueprint(viz.blueprint, url_prefix="/api/viz")

# Display the development key when starting the server
print(f"Development API Key: {API_KEYS['development']}")

if __name__ == "__main__":
    # Quart's run method
    app.run(host="0.0.0.0", port=8000, debug=True)