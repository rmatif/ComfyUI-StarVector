"""Keep StarVector 8B model construction local and offline."""

from __future__ import annotations


def install_local_siglip_patch() -> None:
    """Build StarVector 8B's SigLIP encoder without downloading its base repo."""

    from starvector.model.image_encoder import image_encoder

    image_encoder_class = image_encoder.ImageEncoder
    if getattr(image_encoder_class, "_comfy_local_model_patch", False):
        return

    original_init = image_encoder_class.__init__

    def local_init(self, config, **kwargs):
        parent_model_dir = kwargs.get("parent_model_dir")
        if not parent_model_dir or "siglip" not in config.image_encoder_type:
            return original_init(self, config, **kwargs)

        import torch.nn as nn
        from accelerate import init_empty_weights
        from transformers import (
            SiglipImageProcessor,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        nn.Module.__init__(self)
        self.image_encoder_type = config.image_encoder_type
        vision_config = SiglipVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_channels=3,
            patch_size=16,
            image_size=config.image_size,
            attention_dropout=0.0,
            layer_norm_eps=1e-6,
            hidden_act="gelu_pytorch_tanh",
        )
        with init_empty_weights():
            self.visual_encoder = SiglipVisionModel(vision_config).vision_model
        self.processor = SiglipImageProcessor.from_pretrained(
            parent_model_dir,
            local_files_only=True,
        )
        print("[StarVector] Local SigLIP vision structure is ready")

    image_encoder_class.__init__ = local_init
    image_encoder_class._comfy_local_model_patch = True


def install_local_starcoder2_patch() -> None:
    """Construct StarCoder2 from config when StarVector supplies local weights.

    The StarVector package normally calls ``from_pretrained`` for
    ``bigcode/starcoder2-7b`` while constructing the 8B model. The outer
    StarVector checkpoint already contains those weights, so that call causes
    an unnecessary runtime download and is incompatible with offline workers.
    """

    from starvector.model.llm import starcoder2

    starcoder_model = starcoder2.StarCoderModel
    if getattr(starcoder_model, "_comfy_local_model_patch", False):
        return

    original_init = starcoder_model.__init__

    def local_init(self, config, **kwargs):
        parent_model_dir = kwargs.get("parent_model_dir")
        if not parent_model_dir:
            return original_init(self, config, **kwargs)

        import torch.nn as nn
        from accelerate import init_empty_weights
        from starvector.train.util import get_module_class_from_name
        from transformers import AutoTokenizer, Starcoder2Config, Starcoder2ForCausalLM

        nn.Module.__init__(self)
        self.tokenizer = AutoTokenizer.from_pretrained(
            parent_model_dir,
            use_fast=False,
            local_files_only=True,
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.svg_start_token = "<svg-start>"
        self.svg_end_token = "<svg-end>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        self.tokenizer.add_tokens(
            [
                self.svg_start_token,
                self.image_start_token,
                self.text_start_token,
                self.svg_end_token,
            ]
        )
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
        self.svg_end_token_id = self.tokenizer.encode(self.svg_end_token)[0]
        self.tokenizer.padding_side = "left"

        self.max_length = getattr(config, "max_length", config.max_length_train)
        model_config = Starcoder2Config(
            # The StarVector tokenizer adds padding and task markers. Build the
            # embedding at its final size so the pad token is valid immediately.
            vocab_size=len(self.tokenizer),
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_kv_heads,
            hidden_act="gelu_pytorch_tanh",
            max_position_embeddings=16384,
            initializer_range=0.018042,
            norm_epsilon=1e-5,
            rope_theta=1_000_000,
            sliding_window=4096,
            attention_dropout=0.1,
            residual_dropout=0.1,
            embedding_dropout=0.1,
            use_bias=True,
            use_cache=config.use_cache,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        model_config._attn_implementation = "sdpa"

        print("[StarVector] Creating StarCoder2 from local config")
        print("[StarVector] Weights will be loaded from the StarVector checkpoint")
        # Avoid allocating and initializing a second 7B copy before the outer
        # StarVector checkpoint loader assigns the real tensors.
        with init_empty_weights():
            model = Starcoder2ForCausalLM(model_config)
        self.transformer = model
        self.prompt = "<svg"
        print("[StarVector] Local StarCoder2 structure is ready")

        transformer_layer_cls = kwargs.get(
            "transformer_layer_cls", "Starcoder2DecoderLayer"
        )
        self.transformer_layer_cls = get_module_class_from_name(
            self, transformer_layer_cls
        )

    starcoder_model.__init__ = local_init
    starcoder_model._comfy_local_model_patch = True
