"""
test_tools.py
Tests for API tool wrappers and MCP server classes.
"""

import sys
import unittest

sys.path.insert(0, '/Users/kimsoyeon/ai4sci_kaeri')

from AG_src.tools.api.base_tool import BaseTool, ToolResult
from AG_src.tools.api.rfdiffusion_tool import RFdiffusionTool
from AG_src.tools.api.proteinmpnn_tool import ProteinMPNNTool
from AG_src.tools.api.esmfold_tool import ESMFoldTool
from AG_src.tools.api.diffdock_tool import DiffDockTool
from AG_src.tools.api.openfold3_tool import OpenFold3Tool
from AG_src.tools.api.boltz2_tool import Boltz2Tool
from AG_src.tools.api.molmim_tool import MolMIMTool
from AG_src.tools.api.esm2_tool import ESM2Tool

from AG_src.tools.mcp.base_server import MCPServer, MCPTool
from AG_src.tools.mcp.pyrosetta_server import PyRosettaMCPServer
from AG_src.tools.mcp.foldmason_server import FoldMasonMCPServer
from AG_src.tools.mcp.pymol_server import PyMOLMCPServer, _sanitize_pml_path

_DUMMY_KEY = "test-key-xxx"


class TestBaseToolAbstract(unittest.TestCase):
    def test_base_tool_is_abstract(self):
        """BaseTool cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseTool(api_key=_DUMMY_KEY)  # type: ignore[abstract]


class TestToolResultDataclass(unittest.TestCase):
    def test_tool_result_dataclass(self):
        """ToolResult has success, data, and error fields."""
        ok = ToolResult(success=True, data={"key": "val"}, error=None)
        self.assertTrue(ok.success)
        self.assertEqual(ok.data, {"key": "val"})
        self.assertIsNone(ok.error)

        fail = ToolResult(success=False, data={}, error="something went wrong")
        self.assertFalse(fail.success)
        self.assertEqual(fail.error, "something went wrong")

    def test_tool_result_ok_classmethod(self):
        """ToolResult.ok() convenience constructor works."""
        result = ToolResult.ok(data={"x": 1}, elapsed_ms=42)
        self.assertTrue(result.success)
        self.assertEqual(result.data["x"], 1)
        self.assertEqual(result.elapsed_ms, 42)

    def test_tool_result_fail_classmethod(self):
        """ToolResult.fail() convenience constructor works."""
        result = ToolResult.fail(error="oops")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "oops")
        self.assertEqual(result.data, {})


class TestAPIToolsInstantiate(unittest.TestCase):
    def _try_instantiate(self, cls):
        """Try instantiating with a dummy key; return instance or None on key error."""
        try:
            return cls(api_key=_DUMMY_KEY)
        except Exception as exc:
            # Only accept failure if it's truly key-related, not a class-structure issue
            self.assertNotIsInstance(exc, TypeError,
                f"{cls.__name__} raised TypeError: {exc}")
            return None

    def test_api_tools_instantiate(self):
        """All 8 API tools instantiate (with dummy key where needed)."""
        tool_classes = [
            OpenFold3Tool,
            RFdiffusionTool,
            ProteinMPNNTool,
            ESMFoldTool,
            DiffDockTool,
            Boltz2Tool,
            MolMIMTool,
            ESM2Tool,
        ]
        for cls in tool_classes:
            with self.subTest(tool=cls.__name__):
                instance = self._try_instantiate(cls)
                if instance is not None:
                    self.assertIsInstance(instance, BaseTool)


class TestRFdiffusionTool(unittest.TestCase):
    def setUp(self):
        self.tool = RFdiffusionTool(api_key=_DUMMY_KEY)

    def test_rfdiffusion_has_design_binder(self):
        """RFdiffusionTool has design_binder method."""
        self.assertTrue(hasattr(self.tool, 'design_binder'))
        self.assertTrue(callable(self.tool.design_binder))


class TestProteinMPNNTool(unittest.TestCase):
    def setUp(self):
        self.tool = ProteinMPNNTool(api_key=_DUMMY_KEY)

    def test_proteinmpnn_has_predict(self):
        """ProteinMPNNTool has predict_sequences method."""
        self.assertTrue(hasattr(self.tool, 'predict_sequences'))
        self.assertTrue(callable(self.tool.predict_sequences))


class TestESMFoldTool(unittest.TestCase):
    def setUp(self):
        self.tool = ESMFoldTool(api_key=_DUMMY_KEY)

    def test_esmfold_has_predict(self):
        """ESMFoldTool has predict_structure method."""
        self.assertTrue(hasattr(self.tool, 'predict_structure'))
        self.assertTrue(callable(self.tool.predict_structure))


class TestDiffDockTool(unittest.TestCase):
    def setUp(self):
        self.tool = DiffDockTool(api_key=_DUMMY_KEY)

    def test_diffdock_has_dock(self):
        """DiffDockTool has dock method."""
        self.assertTrue(hasattr(self.tool, 'dock'))
        self.assertTrue(callable(self.tool.dock))


class TestMCPServerBase(unittest.TestCase):
    def test_mcp_server_base(self):
        """MCPServer instantiates with server_name and has register/list_tools/dispatch."""
        server = MCPServer(server_name="test_server")
        self.assertEqual(server.server_name, "test_server")
        self.assertTrue(hasattr(server, 'register') and callable(server.register))
        self.assertTrue(hasattr(server, 'list_tools') and callable(server.list_tools))
        self.assertTrue(hasattr(server, 'dispatch') and callable(server.dispatch))
        # Initially no tools registered
        self.assertEqual(server.list_tools(), [])


class TestMCPToolDataclass(unittest.TestCase):
    def test_mcp_tool_dataclass(self):
        """MCPTool has name, description, input_schema, output_schema, handler."""
        def dummy_handler(**kwargs):
            return {"result": "ok"}

        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
        )
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "A test tool")
        self.assertIsInstance(tool.input_schema, dict)
        self.assertIsInstance(tool.output_schema, dict)
        self.assertTrue(callable(tool.handler))
        self.assertEqual(tool.handler(), {"result": "ok"})


class TestPyRosettaServer(unittest.TestCase):
    def setUp(self):
        # PyRosetta may not be installed; server should still instantiate
        self.server = PyRosettaMCPServer()

    def test_pyrosetta_server_registers_7_tools(self):
        """PyRosettaMCPServer registers exactly 7 tools."""
        tools = self.server.list_tools()
        self.assertEqual(len(tools), 7,
            f"Expected 7 tools, got {len(tools)}: {[t['name'] for t in tools]}")

    def test_pyrosetta_tool_names(self):
        """PyRosettaMCPServer registers the expected tool names."""
        expected_names = {
            "relax_structure",
            "compute_ddg",
            "compute_binding_energy",
            "flexpep_dock",
            "fast_design",
            "energy_decomposition",
            "interface_analysis",
        }
        actual_names = {t['name'] for t in self.server.list_tools()}
        self.assertEqual(actual_names, expected_names)


class TestFoldMasonServer(unittest.TestCase):
    def setUp(self):
        self.server = FoldMasonMCPServer()

    def test_foldmason_server_registers_tools(self):
        """FoldMasonMCPServer registers its tools."""
        tools = self.server.list_tools()
        self.assertGreater(len(tools), 0)
        tool_names = [t['name'] for t in tools]
        self.assertIn('easy_msa', tool_names)
        self.assertIn('compute_lddt', tool_names)
        self.assertIn('cluster_structures', tool_names)


class TestPyMOLServer(unittest.TestCase):
    def setUp(self):
        self.server = PyMOLMCPServer()

    def test_pymol_server_registers_tools(self):
        """PyMOLMCPServer registers its tools."""
        tools = self.server.list_tools()
        self.assertGreater(len(tools), 0)
        tool_names = [t['name'] for t in tools]
        self.assertIn('render_overview', tool_names)
        self.assertIn('render_closeup', tool_names)
        self.assertIn('render_electrostatics', tool_names)


class TestMCPDispatchUnknownRaises(unittest.TestCase):
    def test_mcp_dispatch_unknown_raises(self):
        """Dispatching an unknown tool name raises KeyError."""
        server = MCPServer(server_name="test_server")
        with self.assertRaises(KeyError):
            server.dispatch("nonexistent_tool")


class TestPmlPathSanitization(unittest.TestCase):
    def test_pml_path_sanitization(self):
        """_sanitize_pml_path raises ValueError when path contains newlines."""
        with self.assertRaises(ValueError):
            _sanitize_pml_path("/valid/path\nmalicious_command")

    def test_pml_path_sanitization_carriage_return(self):
        """_sanitize_pml_path raises ValueError when path contains carriage returns."""
        with self.assertRaises(ValueError):
            _sanitize_pml_path("/valid/path\rmalicious_command")

    def test_pml_path_sanitization_clean_path(self):
        """_sanitize_pml_path returns a string for a clean path."""
        result = _sanitize_pml_path("/tmp/test.pdb")
        self.assertIsInstance(result, str)
        self.assertNotIn('\n', result)
        self.assertNotIn('\r', result)


if __name__ == '__main__':
    unittest.main()
