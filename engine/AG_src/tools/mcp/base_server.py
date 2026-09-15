"""
Base MCP Server for Local Computation Tools
============================================
Defines the MCPTool descriptor and MCPServer base class shared by all
local-computation MCP server wrappers (PyRosetta, FoldMason, PyMOL).

Design:
    - MCPTool dataclass: name, description, JSON schemas, handler callable
    - MCPServer: register(), list_tools(), dispatch()

All concrete server classes (PyRosettaMCPServer, FoldMasonMCPServer,
PyMOLMCPServer) inherit from MCPServer and register their tools in __init__.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Descriptor for a single MCP-exposed tool.

    Attributes:
        name:          Unique tool identifier (snake_case).
        description:   One-line description shown to the agent.
        input_schema:  JSON Schema dict describing required/optional inputs.
        output_schema: JSON Schema dict describing the returned outputs.
        handler:       Callable that implements the tool logic.
                       Signature: handler(**kwargs) -> dict[str, Any]
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


class MCPServer:
    """Generic MCP server: registers tools and dispatches calls by name.

    Usage:
        class MyServer(MCPServer):
            def __init__(self):
                super().__init__("my_server")
                self.register(MCPTool(name="foo", ..., handler=self._foo))

            def _foo(self, **kwargs):
                return {"result": 42}

        server = MyServer()
        result = server.dispatch("foo")
    """

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a tool with this server."""
        self._tools[tool.name] = tool
        logger.debug(
            "MCPServer '%s': registered tool '%s'", self.server_name, tool.name
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool manifest (name + description + schemas).

        Returns:
            List of dicts, one per registered tool.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
            }
            for t in self._tools.values()
        ]

    def dispatch(self, tool_name: str, validate_input: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Call a registered tool by name.

        Args:
            tool_name:      Tool identifier (must match a registered name).
            validate_input: If True, validate kwargs against the tool's
                            input_schema before calling the handler (requires
                            the jsonschema package).
            **kwargs:       Tool-specific keyword arguments forwarded to the handler.

        Returns:
            Tool output dict as returned by the handler.

        Raises:
            KeyError:    If tool_name is not registered.
            ValueError:  If validate_input is True and kwargs fail schema validation.
            RuntimeError: If the handler raises an exception.
        """
        if tool_name not in self._tools:
            registered = list(self._tools.keys())
            raise KeyError(
                f"Tool '{tool_name}' not found in server '{self.server_name}'. "
                f"Registered tools: {registered}"
            )
        tool = self._tools[tool_name]
        if validate_input:
            try:
                from jsonschema import validate, ValidationError
                validate(instance=kwargs, schema=tool.input_schema)
            except ImportError:
                logger.warning("jsonschema not installed; skipping input validation")
            except ValidationError as exc:
                raise ValueError(
                    f"Input validation failed for tool '{tool_name}': {exc.message}"
                ) from exc
        try:
            return tool.handler(**kwargs)
        except Exception as exc:
            logger.error(
                "Tool '%s' in server '%s' failed: %s", tool_name, self.server_name, exc
            )
            raise RuntimeError(
                f"Tool '{tool_name}' execution failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        tool_names = list(self._tools.keys())
        return (
            f"{self.__class__.__name__}("
            f"server={self.server_name!r}, tools={tool_names})"
        )
