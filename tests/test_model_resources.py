import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_resources import ModelResourceError, resolve_model_resource


class ModelResourceTests(unittest.TestCase):
    def test_resolves_complete_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            self._write_model(model)

            resolved = resolve_model_resource(str(model), root / "cache")

            self.assertEqual(Path(resolved), model.resolve())

    def test_extracts_repository_tar_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            self._write_model(source)
            archive = root / "model.tar"
            with tarfile.open(archive, "w") as output:
                for path in source.iterdir():
                    output.add(path, arcname=path.name)

            first = Path(resolve_model_resource(str(archive), root / "cache"))
            second = Path(resolve_model_resource(str(archive), root / "cache"))

            self.assertEqual(first, second)
            self.assertTrue((first / "config.json").is_file())
            self.assertTrue((first / "model.safetensors").is_file())

    def test_finds_model_below_archive_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source" / "snapshot"
            self._write_model(source)
            archive = root / "model.tar"
            with tarfile.open(archive, "w") as output:
                output.add(source, arcname="snapshot")

            resolved = Path(resolve_model_resource(str(archive), root / "cache"))

            self.assertEqual(resolved.name, "snapshot")
            self.assertTrue((resolved / "config.json").is_file())

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.tar"
            with tarfile.open(archive, "w") as output:
                entry = tarfile.TarInfo("../escaped")
                payload = b"unsafe"
                entry.size = len(payload)
                output.addfile(entry, io.BytesIO(payload))

            with self.assertRaisesRegex(ModelResourceError, "Unsafe path"):
                resolve_model_resource(str(archive), root / "cache")

            self.assertFalse((root / "escaped").exists())

    def test_rejects_archive_backslash_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.tar"
            with tarfile.open(archive, "w") as output:
                entry = tarfile.TarInfo("..\\escaped")
                payload = b"unsafe"
                entry.size = len(payload)
                output.addfile(entry, io.BytesIO(payload))

            with self.assertRaisesRegex(ModelResourceError, "Unsafe path"):
                resolve_model_resource(str(archive), root / "cache")

    def test_rejects_incomplete_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                ModelResourceError, "No complete Hugging Face model"
            ):
                resolve_model_resource(str(model), root / "cache")

    @staticmethod
    def _write_model(path: Path):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")


if __name__ == "__main__":
    unittest.main()
