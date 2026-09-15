"""
AG_SRC/tools — Tool Wrappers for the SSTR2 Peptide Binder Agentic Flow
=======================================================================
Two categories of tools are provided:

    tools/api/  — NVIDIA NIM REST API wrappers (cloud computation)
    tools/mcp/  — MCP server wrappers for local computation

Quick imports:
    from AG_SRC.tools.api import (
        ToolResult, OpenFold3Tool, RFdiffusionTool, ProteinMPNNTool,
        ESMFoldTool, DiffDockTool, Boltz2Tool, MolMIMTool, ESM2Tool,
    )
    from AG_SRC.tools.mcp import (
        PyRosettaMCPServer, FoldMasonMCPServer, PyMOLMCPServer,
    )
"""

from .api import (
    BaseTool,
    Boltz2Tool,
    DiffDockTool,
    ESM2Tool,
    ESMFoldTool,
    MolMIMTool,
    OpenFold3Tool,
    ProteinMPNNTool,
    RFdiffusionTool,
    ToolResult,
)
from .mcp import (
    FoldMasonMCPServer,
    MCPServer,
    MCPTool,
    PyMOLMCPServer,
    PyRosettaMCPServer,
)

__all__ = [
    # API tool base types
    "BaseTool",
    "ToolResult",
    # API tools
    "OpenFold3Tool",
    "RFdiffusionTool",
    "ProteinMPNNTool",
    "ESMFoldTool",
    "DiffDockTool",
    "Boltz2Tool",
    "MolMIMTool",
    "ESM2Tool",
    # MCP base types
    "MCPServer",
    "MCPTool",
    # MCP servers
    "PyRosettaMCPServer",
    "FoldMasonMCPServer",
    "PyMOLMCPServer",
]
