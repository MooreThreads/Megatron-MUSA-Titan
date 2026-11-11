"""CPU-only tests for Titan's PyTorch device selector compatibility."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "musa_titan" / "__init__.py"
DCP_MODULE = "torch.distributed.checkpoint.filesystem"


def load_helpers(torch, modules):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    names = {"_get_available_device_type", "_patch_available_device_type"}
    body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in body} != names:
        raise AssertionError("Titan device helpers are missing")
    namespace = {"torch": torch, "sys": SimpleNamespace(modules=modules)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_get_available_device_type"], namespace["_patch_available_device_type"]


class DeviceSelectorTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            musa=SimpleNamespace(is_available=lambda: True),
            _utils=SimpleNamespace(),
            _C=SimpleNamespace(_get_privateuse1_backend_name=lambda: "musa"),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        self.original = lambda: "cuda"
        self.torch._utils._get_available_device_type = self.original
        self.modules = {}
        self.fixed, self.apply = load_helpers(self.torch, self.modules)

    def test_musa_privateuse1_is_selected_before_cuda_alias(self):
        self.assertEqual(self.fixed(), "musa")

    def test_cuda_and_none_fallbacks_are_preserved(self):
        self.torch.musa.is_available = lambda: False
        self.torch.cuda.is_available = lambda: True
        self.assertEqual(self.fixed(), "cuda")
        self.torch.cuda.is_available = lambda: False
        self.assertIsNone(self.fixed())

    def test_imported_dcp_alias_is_updated_only_when_original(self):
        filesystem = SimpleNamespace(_get_available_device_type=self.original)
        self.modules[DCP_MODULE] = filesystem
        self.apply()
        self.assertIs(self.torch._utils._get_available_device_type, self.fixed)
        self.assertIs(filesystem._get_available_device_type, self.fixed)
        self.assertEqual(filesystem._get_available_device_type(), "musa")

    def test_third_party_dcp_selector_is_preserved(self):
        third_party = lambda: "custom"
        self.modules[DCP_MODULE] = SimpleNamespace(_get_available_device_type=third_party)
        self.apply()
        self.assertIs(self.modules[DCP_MODULE]._get_available_device_type, third_party)

    def test_later_dcp_import_uses_fixed_torch_helper(self):
        self.apply()
        filesystem = SimpleNamespace(
            _get_available_device_type=self.torch._utils._get_available_device_type
        )
        self.assertEqual(filesystem._get_available_device_type(), "musa")

    def test_module_without_selector_is_unchanged(self):
        filesystem = SimpleNamespace()
        self.modules[DCP_MODULE] = filesystem
        self.apply()
        self.assertFalse(hasattr(filesystem, "_get_available_device_type"))


if __name__ == "__main__":
    unittest.main()
