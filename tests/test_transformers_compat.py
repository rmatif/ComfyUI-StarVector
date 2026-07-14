import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parents[1] / "transformers_compat.py"


def load_module():
    spec = importlib.util.spec_from_file_location("starvector_transformers_compat", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransformersCompatibilityTests(unittest.TestCase):
    def test_disables_optional_sklearn_integration_after_abi_error(self):
        module = load_module()
        utils = types.ModuleType("transformers.utils")
        import_utils = types.ModuleType("transformers.utils.import_utils")
        utils.import_utils = import_utils

        real_import = __import__

        def incompatible_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ValueError("numpy.dtype size changed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=incompatible_sklearn), mock.patch.dict(
            sys.modules,
            {
                "transformers": types.ModuleType("transformers"),
                "transformers.utils": utils,
                "transformers.utils.import_utils": import_utils,
            },
        ):
            self.assertTrue(module.prepare_transformers_imports())

        self.assertFalse(utils.is_sklearn_available())
        self.assertFalse(import_utils.is_sklearn_available())

    def test_initializes_transformers_4_remote_model_loading_attributes(self):
        module = load_module()
        utils = types.ModuleType("transformers.utils")
        import_utils = types.ModuleType("transformers.utils.import_utils")
        utils.import_utils = import_utils

        class FakePreTrainedModel:
            def __init__(self):
                self._no_split_modules = ["DecoderLayer"]

        transformers = types.ModuleType("transformers")
        transformers.PreTrainedModel = FakePreTrainedModel
        transformers.utils = utils
        real_import = __import__

        def incompatible_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ValueError("numpy.dtype size changed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=incompatible_sklearn), mock.patch.dict(
            sys.modules,
            {
                "transformers": transformers,
                "transformers.utils": utils,
                "transformers.utils.import_utils": import_utils,
            },
        ):
            module.prepare_legacy_pretrained_models()
            model = FakePreTrainedModel()

        self.assertEqual({}, model.all_tied_weights_keys)
        self.assertEqual({"DecoderLayer"}, model._no_split_modules)
        self.assertEqual({}, model._tp_plan)

    def test_restores_nested_language_model_tied_weights(self):
        module = load_module()
        input_embeddings = types.SimpleNamespace(weight=object())
        output_embeddings = types.SimpleNamespace(weight=object())

        class FakeLanguageModel:
            def tie_weights(self):
                output_embeddings.weight = input_embeddings.weight

            def get_input_embeddings(self):
                return input_embeddings

            def get_output_embeddings(self):
                return output_embeddings

        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                svg_transformer=types.SimpleNamespace(transformer=FakeLanguageModel())
            )
        )

        module.restore_starvector_tied_weights(model)

        self.assertIs(input_embeddings.weight, output_embeddings.weight)


if __name__ == "__main__":
    unittest.main()
