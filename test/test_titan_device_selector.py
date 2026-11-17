"""CPU-only tests for Titan's PyTorch device selector compatibility."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "musa_titan" / "__init__.py"
DCP_MODULE = "torch.distributed.checkpoint.filesystem"


def load_helpers(torch, modules):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    names = {"_get_available_device_type", "_patch_tensor_type", "patch_after_import_torch"}
    body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in body} != names:
        raise AssertionError("Device helpers are missing")
    namespace = {"torch": torch, "sys": SimpleNamespace(modules=modules)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)

    def apply():
        def apply_torchada():
            torch.cuda.get_device_capability = torch.musa.get_device_capability
            torch.cuda.get_device_properties = torch.musa.get_device_properties
            torch.cuda._get_device_properties = torch.musa.get_device_properties

        torchada = SimpleNamespace(apply_patches=apply_torchada)
        with patch.dict("sys.modules", {"torchada": torchada}):
            namespace["patch_after_import_torch"]()

    return namespace["_get_available_device_type"], apply


class DeviceProperties:
    __slots__ = ()
    major = 3
    minor = 1
    name = "MTT S5000"
    total_memory = 81920
    multi_processor_count = 60


class DeviceSelectorTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            Tensor=SimpleNamespace(type=lambda: "torch.musa.FloatTensor"),
            musa=SimpleNamespace(
                is_available=lambda: True,
                get_device_capability=lambda device=None: (3, 1),
                get_device_properties=lambda device=None: DeviceProperties(),
            ),
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

    def test_capability_and_properties_keep_torchada_bindings(self):
        self.apply()
        self.apply()
        self.assertIs(self.torch.cuda.get_device_capability, self.torch.musa.get_device_capability)
        self.assertEqual(self.torch.cuda.get_device_capability(device=0), (3, 1))
        for getter in (self.torch.cuda.get_device_properties, self.torch.cuda._get_device_properties):
            self.assertIs(getter, self.torch.musa.get_device_properties)
            props = getter(device=1)
            self.assertEqual((props.major, props.minor), (3, 1))

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
