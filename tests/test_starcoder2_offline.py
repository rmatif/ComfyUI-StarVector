import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starcoder2_offline import (
    _get_siglip_vision_transformer,
    _load_local_siglip_processor,
    install_local_siglip_patch,
    install_local_starvector_v2_patch,
    install_local_starcoder2_patch,
)


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None
    bos_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.tokens = []

    def add_special_tokens(self, tokens):
        if "pad_token" in tokens:
            self.pad_token_id = 49152
        return len(tokens)

    def add_tokens(self, tokens):
        self.tokens.extend(tokens)
        return len(tokens)

    def encode(self, token):
        return [49153 + self.tokens.index(token)]

    def __len__(self):
        return 49157


class FakeInnerModel:
    def __init__(self, config):
        self.config = config
        self.resized_to = None

    def resize_token_embeddings(self, size):
        self.resized_to = size


class Starcoder2OfflineTests(unittest.TestCase):
    def test_starvector_v2_reuses_processor_from_local_image_encoder(self):
        original_calls = []

        def original_init(self, config, **kwargs):
            original_calls.append((config, kwargs))

        fake_class = type("StarVectorStarCoder2", (), {"__init__": original_init})
        fake_module = types.SimpleNamespace(StarVectorStarCoder2=fake_class)
        processor = object()

        class FakeBase:
            def __init__(self, config, **kwargs):
                self.image_encoder = types.SimpleNamespace(processor=processor)

        models_package = types.ModuleType("starvector.model.models")
        models_package.starvector_v2 = fake_module
        base_module = types.SimpleNamespace(StarVectorBase=FakeBase)
        modules = {
            "starvector.model.models": models_package,
            "starvector.model.models.starvector_v2": fake_module,
            "starvector.model.models.starvector_base": base_module,
        }
        with patch.dict(sys.modules, modules):
            install_local_starvector_v2_patch()
            model = fake_class.__new__(fake_class)
            model.__init__(object(), parent_model_dir="/models/starvector-8b")

        self.assertEqual(original_calls, [])
        self.assertIs(model.processor, processor)

    def test_siglip_processor_uses_preprocessor_config(self):
        class FakeProcessor:
            @classmethod
            def from_dict(cls, config):
                return config

        with tempfile.TemporaryDirectory() as model_dir:
            config = {"size": {"height": 384, "width": 384}}
            Path(model_dir, "preprocessor_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            processor = _load_local_siglip_processor(FakeProcessor, model_dir)

        self.assertEqual(processor, config)

    def test_siglip_transformer_supports_transformers_4_and_5_layouts(self):
        direct_model = object()
        wrapped_model = types.SimpleNamespace(vision_model=direct_model)

        self.assertIs(_get_siglip_vision_transformer(direct_model), direct_model)
        self.assertIs(_get_siglip_vision_transformer(wrapped_model), direct_model)

    def test_local_parent_constructs_siglip_without_from_pretrained(self):
        original_calls = []
        fake_class = type(
            "ImageEncoder",
            (),
            {"__init__": lambda *args, **kwargs: original_calls.append((args, kwargs))},
        )
        fake_module = types.SimpleNamespace(ImageEncoder=fake_class)
        fake_config = types.SimpleNamespace(
            image_encoder_type="siglip_384",
            image_size=384,
        )

        class FakeSiglipVisionConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeSiglipVisionModel:
            def __init__(self, config):
                self.vision_model = types.SimpleNamespace(config=config)

        processor = object()

        class FakeProcessor:
            @classmethod
            def from_dict(cls, config):
                return processor

        transformers = types.SimpleNamespace(
            SiglipImageProcessor=FakeProcessor,
            SiglipVisionConfig=FakeSiglipVisionConfig,
            SiglipVisionModel=FakeSiglipVisionModel,
        )
        torch_nn = types.ModuleType("torch.nn")
        torch_nn.Module = type("Module", (), {"__init__": lambda self: None})
        torch = types.ModuleType("torch")
        torch.nn = torch_nn

        class FakeEmptyWeights:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        accelerate = types.SimpleNamespace(init_empty_weights=FakeEmptyWeights)
        image_encoder_package = types.ModuleType("starvector.model.image_encoder")
        image_encoder_package.image_encoder = fake_module

        modules = {
            "starvector.model.image_encoder": image_encoder_package,
            "starvector.model.image_encoder.image_encoder": fake_module,
            "torch": torch,
            "torch.nn": torch_nn,
            "accelerate": accelerate,
            "transformers": transformers,
        }
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "preprocessor_config.json").write_text(
                json.dumps({"size": {"height": 384, "width": 384}}),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules):
                install_local_siglip_patch()
                model = fake_class.__new__(fake_class)
                model.__init__(fake_config, parent_model_dir=model_dir)

        self.assertEqual(original_calls, [])
        self.assertEqual(model.visual_encoder.config.hidden_size, 1024)
        self.assertEqual(model.visual_encoder.config.num_hidden_layers, 24)
        self.assertEqual(model.visual_encoder.config.image_size, 384)
        self.assertIs(model.processor, processor)

    def test_local_parent_constructs_model_without_from_pretrained(self):
        fake_class = type("StarCoderModel", (), {"__init__": lambda *args, **kwargs: None})
        fake_module = types.SimpleNamespace(StarCoderModel=fake_class)
        fake_config = types.SimpleNamespace(
            vocab_size=49152,
            hidden_size=4608,
            max_length=8192,
            max_length_train=16000,
            num_hidden_layers=32,
            num_attention_heads=36,
            num_kv_heads=4,
            use_cache=True,
        )
        tokenizer = FakeTokenizer()

        class FakeStarcoder2Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer
            ),
            Starcoder2Config=FakeStarcoder2Config,
            Starcoder2ForCausalLM=FakeInnerModel,
        )
        train_util = types.SimpleNamespace(
            get_module_class_from_name=lambda model, name: name
        )
        accelerate = types.SimpleNamespace()

        class FakeEmptyWeights:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        accelerate.init_empty_weights = FakeEmptyWeights
        torch_nn = types.ModuleType("torch.nn")
        torch_nn.Module = type(
            "Module", (), {"__init__": lambda self: None}
        )
        torch = types.ModuleType("torch")
        torch.nn = torch_nn

        modules = {
            "starvector.model.llm.starcoder2": fake_module,
            "starvector.train.util": train_util,
            "accelerate": accelerate,
            "torch": torch,
            "torch.nn": torch_nn,
            "transformers": transformers,
        }
        with patch.dict(sys.modules, modules):
            starvector_llm = types.ModuleType("starvector.model.llm")
            starvector_llm.starcoder2 = fake_module
            with patch.dict(sys.modules, {"starvector.model.llm": starvector_llm}):
                install_local_starcoder2_patch()
                model = fake_class.__new__(fake_class)
                model.__init__(fake_config, parent_model_dir="/models/starvector-8b")

        self.assertEqual(model.transformer.config.hidden_size, 4608)
        self.assertEqual(model.transformer.config.vocab_size, 49157)
        self.assertEqual(model.transformer.config.intermediate_size, 18432)
        self.assertEqual(model.transformer.config.num_key_value_heads, 4)
        self.assertEqual(model.transformer.config._attn_implementation, "sdpa")
        self.assertIsNone(model.transformer.resized_to)
        self.assertEqual(model.tokenizer.padding_side, "left")


if __name__ == "__main__":
    unittest.main()
