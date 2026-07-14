"""Compatibility helpers for the Transformers version bundled with ComfyUI."""

from __future__ import annotations


def prepare_transformers_imports() -> bool:
    """Disable Transformers' optional sklearn integration when sklearn is broken.

    Transformers 5 imports ``sklearn.metrics`` while its generation package is
    initialized.  Some ComfyUI images include a scikit-learn wheel built for an
    older NumPy ABI, which makes otherwise unrelated model imports fail.  SVG
    generation does not use the assisted-generation feature backed by sklearn,
    so it is safe to mark that optional integration unavailable in this case.

    Returns ``True`` when the optional integration had to be disabled.
    """

    try:
        from sklearn.metrics import roc_curve  # noqa: F401
    except (ImportError, ValueError):
        import transformers.utils as transformers_utils
        from transformers.utils import import_utils

        unavailable = lambda: False
        transformers_utils.is_sklearn_available = unavailable
        import_utils.is_sklearn_available = unavailable
        return True

    return False


def prepare_legacy_pretrained_models() -> None:
    """Restore base attributes expected by pre-Transformers-5 remote models.

    Transformers 5 moved several loading attributes from
    ``PreTrainedModel.__init__`` to ``post_init``. StarVector's remote model
    classes were authored against Transformers 4 and do not call ``post_init``
    themselves. Initializing the legacy defaults in the base constructor lets
    those classes load while current model classes remain free to replace the
    values in their normal ``post_init`` call.
    """

    prepare_transformers_imports()

    from transformers import PreTrainedModel

    if getattr(PreTrainedModel, "_starvector_legacy_init_patch", False):
        return

    original_init = PreTrainedModel.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.all_tied_weights_keys = {}
        self._tp_plan = {}
        self._ep_plan = {}
        self._pp_plan = {}
        for name in (
            "_keep_in_fp32_modules",
            "_keep_in_fp32_modules_strict",
            "_no_split_modules",
            "_skip_keys_device_placement",
            "_keys_to_ignore_on_load_unexpected",
            "_keys_to_ignore_on_load_missing",
            "_keys_to_ignore_on_save",
        ):
            setattr(self, name, set(getattr(self, name, None) or []))

    PreTrainedModel.__init__ = compatible_init
    PreTrainedModel._starvector_legacy_init_patch = True


def restore_starvector_tied_weights(model) -> None:
    """Re-tie the nested StarCoder output head after legacy wrapper loading."""

    language_model = model.model.svg_transformer.transformer
    language_model.tie_weights()
    input_embeddings = language_model.get_input_embeddings()
    output_embeddings = language_model.get_output_embeddings()
    if input_embeddings.weight is not output_embeddings.weight:
        raise RuntimeError("StarVector language-model embeddings were not tied")
