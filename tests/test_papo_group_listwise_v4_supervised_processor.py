from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLAMAFACTORY_SRC = PROJECT_ROOT / "LLaMA-Factory" / "src" / "llamafactory"


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__path__ = []
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_supervised_processor():
    if "transformers.utils" not in sys.modules:
        transformers_module = types.ModuleType("transformers")
        transformers_module.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None, is_package=True)
        transformers_module.__path__ = []
        transformers_utils = types.ModuleType("transformers.utils")
        transformers_utils.__spec__ = importlib.machinery.ModuleSpec(
            "transformers.utils", loader=None, is_package=True
        )
        transformers_utils.SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
        transformers_utils.SAFE_WEIGHTS_NAME = "model.safetensors"
        transformers_utils.WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
        transformers_utils.WEIGHTS_NAME = "pytorch_model.bin"
        transformers_module.utils = transformers_utils
        sys.modules["transformers"] = transformers_module
        sys.modules["transformers.utils"] = transformers_utils

    if "peft" not in sys.modules:
        peft_module = types.ModuleType("peft")
        peft_module.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None, is_package=True)
        peft_module.__path__ = []
        peft_utils = types.ModuleType("peft.utils")
        peft_utils.__spec__ = importlib.machinery.ModuleSpec("peft.utils", loader=None)
        peft_utils.SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
        peft_utils.WEIGHTS_NAME = "adapter_model.bin"
        peft_module.utils = peft_utils
        sys.modules["peft"] = peft_module
        sys.modules["peft.utils"] = peft_utils

    for package in [
        "llamafactory",
        "llamafactory.extras",
        "llamafactory.data",
        "llamafactory.data.processor",
    ]:
        _ensure_package(package)

    _load_module("llamafactory.extras.constants", LLAMAFACTORY_SRC / "extras" / "constants.py")
    _load_module("llamafactory.extras.logging", LLAMAFACTORY_SRC / "extras" / "logging.py")
    _load_module(
        "llamafactory.data.processor.processor_utils",
        LLAMAFACTORY_SRC / "data" / "processor" / "processor_utils.py",
    )
    supervised = _load_module(
        "llamafactory.data.processor.supervised",
        LLAMAFACTORY_SRC / "data" / "processor" / "supervised.py",
    )
    return supervised.SupervisedDatasetProcessor


SupervisedDatasetProcessor = _bootstrap_supervised_processor()


class _TokenizerStub:
    def decode(self, token_ids, skip_special_tokens=False):
        return "|".join(str(int(token_id)) for token_id in token_ids)


class GroupedSupervisedProcessorTest(unittest.TestCase):
    def test_print_data_example_handles_grouped_candidates(self) -> None:
        processor = object.__new__(SupervisedDatasetProcessor)
        processor.tokenizer = _TokenizerStub()
        example = {
            "input_ids": [[101, 102, 103], [201, 202]],
            "labels": [[-100, 102, 103], [-100, 202]],
            "papo_group_target": [0.8, 0.2],
            "papo_group_oracle_index": 0,
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            processor.print_data_example(example)

        output = stream.getvalue()
        self.assertIn("papo_group_size", output)
        self.assertIn("labels[0]", output)
        self.assertIn("papo_group_target", output)


if __name__ == "__main__":
    unittest.main()
