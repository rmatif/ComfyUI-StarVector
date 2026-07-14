"""Resolve local StarVector model resources for offline ComfyUI workers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import threading


class ModelResourceError(RuntimeError):
    """Raised when a local model resource cannot be used safely."""


_EXTRACTION_LOCK = threading.Lock()
_EXTRACTION_MARKER = ".starvector-extraction-complete"
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)


def resolve_model_resource(
    resource_path: str, cache_root: str | os.PathLike[str]
) -> str:
    """Return a model directory for a local directory, config, or TAR resource.

    CustomComfy replaces a declared AIR in the workflow with an absolute mounted
    path. Repository resources currently arrive as TAR archives, while workers
    that use the shared repository extraction cache provide a directory. Both
    forms are accepted so the node works with either deployment.
    """

    value = resource_path.strip()
    if not value:
        raise ModelResourceError("The model resource path is empty")

    path = Path(value).expanduser().resolve()
    if path.is_dir():
        return str(_find_model_directory(path))

    if not path.is_file():
        raise ModelResourceError(
            f"Model resource does not exist: {path}. "
            "For CustomComfy jobs, declare the AIR as a resource and use the same AIR in model_path."
        )

    if path.name == "config.json":
        return str(_find_model_directory(path.parent))

    if not tarfile.is_tarfile(path):
        raise ModelResourceError(
            f"Model resource must be a StarVector repository directory or TAR archive, got: {path}"
        )

    return str(_extract_archive(path, Path(cache_root)))


def _find_model_directory(root: Path) -> Path:
    candidates = []
    for config_path in root.rglob("config.json"):
        candidate = config_path.parent
        if _has_model_weights(candidate):
            candidates.append(candidate)

    if not candidates:
        raise ModelResourceError(
            f"No complete Hugging Face model was found under {root}; "
            "expected config.json and model weight files"
        )

    # A repository snapshot normally has its model at the archive root. If it
    # contains other nested configs, prefer the shallowest complete model.
    candidates.sort(
        key=lambda candidate: (len(candidate.relative_to(root).parts), str(candidate))
    )
    shallowest_depth = len(candidates[0].relative_to(root).parts)
    shallowest = [
        candidate
        for candidate in candidates
        if len(candidate.relative_to(root).parts) == shallowest_depth
    ]
    if len(shallowest) > 1:
        rendered = ", ".join(str(candidate) for candidate in shallowest)
        raise ModelResourceError(
            f"Multiple StarVector model directories were found: {rendered}"
        )

    return shallowest[0]


def _has_model_weights(directory: Path) -> bool:
    return any(
        next(directory.glob(pattern), None) is not None for pattern in _WEIGHT_PATTERNS
    )


def _archive_cache_key(archive_path: Path) -> str:
    stat = archive_path.stat()
    identity = f"{archive_path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return hashlib.sha256(identity).hexdigest()[:24]


def _extract_archive(archive_path: Path, cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _archive_cache_key(archive_path)

    with _EXTRACTION_LOCK:
        cached_model = _read_cached_model_path(cache_path)
        if cached_model is not None:
            return cached_model

        if cache_path.exists():
            shutil.rmtree(cache_path)

        required_space = int(archive_path.stat().st_size * 1.1)
        available_space = shutil.disk_usage(cache_root).free
        if available_space < required_space:
            raise ModelResourceError(
                f"Insufficient disk space to extract {archive_path}: "
                f"need approximately {required_space} bytes, have {available_space} bytes"
            )

        temporary_path = Path(
            tempfile.mkdtemp(prefix=f"{cache_path.name}-", dir=cache_root)
        )
        try:
            _extract_tar_safely(archive_path, temporary_path)
            model_path = _find_model_directory(temporary_path)
            relative_model_path = model_path.relative_to(temporary_path)
            (temporary_path / _EXTRACTION_MARKER).write_text(
                relative_model_path.as_posix() or ".",
                encoding="utf-8",
            )
            os.replace(temporary_path, cache_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

    cached_model = _read_cached_model_path(cache_path)
    if cached_model is None:
        raise ModelResourceError(f"Model extraction did not complete: {archive_path}")
    return cached_model


def _read_cached_model_path(cache_path: Path) -> Path | None:
    marker_path = cache_path / _EXTRACTION_MARKER
    if not marker_path.is_file():
        return None

    relative_path = marker_path.read_text(encoding="utf-8").strip() or "."
    model_path = (cache_path / relative_path).resolve()
    try:
        model_path.relative_to(cache_path.resolve())
    except ValueError:
        return None

    try:
        return _find_model_directory(model_path)
    except ModelResourceError:
        return None


def _extract_tar_safely(archive_path: Path, destination: Path) -> None:
    """Extract regular files and directories without following archive links."""

    destination_root = destination.resolve()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if "\\" in member.name:
                raise ModelResourceError(f"Unsafe path in model archive: {member.name}")
            relative_path = PurePosixPath(member.name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ModelResourceError(f"Unsafe path in model archive: {member.name}")

            clean_parts = [
                part for part in relative_path.parts if part not in ("", ".")
            ]
            if not clean_parts:
                continue

            target = destination.joinpath(*clean_parts)
            target_parent = target.parent.resolve()
            try:
                target_parent.relative_to(destination_root)
            except ValueError as error:
                raise ModelResourceError(
                    f"Unsafe path in model archive: {member.name}"
                ) from error

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isreg():
                raise ModelResourceError(
                    f"Unsupported entry in model archive: {member.name} ({member.type!r})"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ModelResourceError(
                    f"Could not read model archive entry: {member.name}"
                )
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
