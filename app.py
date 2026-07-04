"""
Flask application providing routes and server startup.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from tool_manager import setup_tool, get_available_tools, get_api_information
from react_engine import react_loop

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-in-production")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serves the main web interface."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Returns application health status."""
    return jsonify({"status": "ok", "model": "gemma-4-E2B-it + LoRA"})


@app.route("/api/tools")
def list_tools():
    """Returns available tools that do not require authentication."""
    tools = get_available_tools()
    return jsonify({"tools": tools, "count": len(tools)})


@app.route("/api/tools/<name>")
def tool_info(name):
    """Returns information for a specified tool."""
    info = get_api_information(name)
    if not info:
        return jsonify({"error": f"Tool '{name}' not found"}), 404
    return jsonify({
        "name": info.get("Name", name),
        "description": info.get("Description", ""),
        "introduction": info.get("Introduction", ""),
    })


@app.route("/api/react", methods=["POST"])
def run_react():
    """Processes a query using the specified tool through the reasoning loop."""
    data = request.get_json()

    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    tool_name = data.get("tool_name", "").strip() if isinstance(data.get("tool_name"), str) else ""
    query = data.get("query", "").strip() if isinstance(data.get("query"), str) else ""
    max_turns = data.get("max_turns", 5)

    if not tool_name:
        return jsonify({"success": False, "error": "tool_name is required"}), 400
    if not query:
        return jsonify({"success": False, "error": "query is required"}), 400

    valid, tool_info = setup_tool(tool_name)
    if not valid:
        return jsonify({"success": False, "error": tool_info}), 400

    try:
        result = react_loop(query, tool_info, max_turns=max_turns)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        logger.exception("ReAct loop failed")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
