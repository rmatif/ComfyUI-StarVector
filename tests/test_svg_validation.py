from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svg_validation import require_valid_svg


class SvgValidationTests(unittest.TestCase):
    def test_accepts_well_formed_svg(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
        self.assertEqual(require_valid_svg(svg), svg)

    def test_rejects_truncated_svg(self):
        with self.assertRaisesRegex(RuntimeError, "generated malformed SVG"):
            require_valid_svg('<svg xmlns="http://www.w3.org/2000/svg"><path')

    def test_rejects_non_svg_document(self):
        with self.assertRaisesRegex(RuntimeError, "root element is not SVG"):
            require_valid_svg("<html />")


if __name__ == "__main__":
    unittest.main()
