from xml.etree import ElementTree


def require_valid_svg(svg: str) -> str:
    """Require a complete, well-formed SVG document from model generation."""
    if not svg.strip():
        raise RuntimeError("StarVector generated an empty SVG")

    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as error:
        raise RuntimeError(
            f"StarVector generated malformed SVG: {error}"
        ) from error

    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise RuntimeError("StarVector output root element is not SVG")

    return svg
