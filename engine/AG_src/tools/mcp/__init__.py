"""
AG_SRC/tools/mcp — Local Computation MCP Server Wrappers
=========================================================
Exposes local computational tools (PyRosetta, FoldMason, PyMOL) as MCP
(Model Context Protocol) server objects. Each server registers its tools
with JSON Schema-validated inputs and outputs.

Usage:
    from AG_SRC.tools.mcp import (
        MCPServer,
        MCPTool,
        PyRosettaMCPServer,
        FoldMasonMCPServer,
        PyMOLMCPServer,
    )

    # Dispatch a tool call
    rosetta = PyRosettaMCPServer()
    result = rosetta.dispatch("relax_structure", pdb_path="/tmp/complex.pdb")

    # List available tools on a server
    for tool in rosetta.list_tools():
        print(tool["name"], "—", tool["description"])
"""

from .base_server import MCPServer, MCPTool
from .foldmason_server import FoldMasonMCPServer
from .pymol_server import PyMOLMCPServer
from .pyrosetta_server import PyRosettaMCPServer

__all__ = [
    # Base classes
    "MCPServer",
    "MCPTool",
    # Concrete servers (in pipeline step order)
    "PyRosettaMCPServer",   # Step06 — physics-based scoring & design
    "FoldMasonMCPServer",   # Step07 — structural MSA & clustering
    "PyMOLMCPServer",       # Step07 — structural visualisation
]
