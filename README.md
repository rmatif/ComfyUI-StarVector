# ComfyUI-StarVector

Custom nodes for generating SVG files from images using [StarVector](https://huggingface.co/starvector/starvector-1b-im2svg) models.

![StarVector](https://img.shields.io/badge/StarVector-SVG%20Generation-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-green)

## Features

- **Image to SVG conversion** using StarVector-1B or StarVector-8B models
- **Automatic model downloading** from HuggingFace to `models/vector/<model-name>/`
- **Offline model resources** from a local Hugging Face repository directory or TAR archive
- **SVG Preview** - Rasterize SVGs to preview them in ComfyUI
- **Save/Load SVG** - Full file I/O support
- **String conversion** - Convert between SVG data and strings

## Installation

### Option 1: ComfyUI Manager (Recommended)
Search for "ComfyUI-StarVector" in the ComfyUI Manager and install.

### Option 2: Manual Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-StarVector.git
```

2. Install dependencies:
```bash
cd ComfyUI-StarVector
pip install -r requirements.txt
```

3. Restart ComfyUI

### Dependencies

**Required:**
- `transformers>=4.36.0`
- `torch>=2.0.0`
- `Pillow>=9.0.0`
- `starvector`

**For SVG Rasterization (at least one):**
- `cairosvg` (recommended)
- OR `svglib` + `reportlab`
- OR the built-in starvector rasterization

**Installing cairosvg on different platforms:**

```bash
# Ubuntu/Debian
sudo apt-get install libcairo2-dev
pip install cairosvg

# macOS
brew install cairo
pip install cairosvg

# Windows
# Download and install GTK+ runtime first, then:
pip install cairosvg
```

## Nodes

### StarVector Model Loader
Loads a StarVector model for SVG generation.

**Inputs:**
- `model_name`: Choose between `starvector-1b-im2svg` (faster) or `starvector-8b-im2svg` (higher quality)
- `device`: `auto`, `cuda`, or `cpu`
- `dtype`: `float16`, `float32`, or `bfloat16`
- `model_path` (optional): Complete local model directory or repository TAR archive. CustomComfy resource AIRs can be used directly.

**Outputs:**
- `model`: The loaded StarVector model

### StarVector Image to SVG
Converts an image to SVG using the loaded model.

**Inputs:**
- `model`: StarVector model from the loader
- `image`: Input image (ComfyUI IMAGE type)
- `max_length`: Maximum SVG token length (100-16000, default 4000)

**Outputs:**
- `svg`: SVG data object
- `svg_string`: Raw SVG string

### SVG Preview
Rasterizes an SVG to preview it as an image.

**Inputs:**
- `svg`: SVG data object
- `width`: Output width (64-4096, default 512)
- `height`: Output height (64-4096, default 512)
- `background`: Background color (`white`, `black`, `transparent`, `gray`)

**Outputs:**
- `image`: Rasterized image

### SVG Preview (String)
Same as SVG Preview but accepts a raw SVG string input.

### Save SVG
Saves an SVG to a file in the output directory.

**Inputs:**
- `svg`: SVG data object
- `filename_prefix`: Filename prefix (default: "starvector")
- `optimize`: Whether to optimize the SVG (remove comments, whitespace)

**Outputs:**
- `filepath`: Path to the saved file

### Load SVG
Loads an SVG file from the input directory.

**Inputs:**
- `svg_file`: Select from available SVG files

**Outputs:**
- `svg`: SVG data object
- `svg_string`: Raw SVG string

### SVG to String / String to SVG
Utility nodes to convert between SVG data objects and raw strings.

## Example Workflow

```
[Load Image] → [StarVector Image to SVG] → [SVG Preview]
                        ↓
                   [Save SVG]
```

1. Load an image using the standard Load Image node
2. Connect to StarVector Model Loader (first run will download the model)
3. Connect to StarVector Image to SVG
4. Preview with SVG Preview node
5. Optionally save with Save SVG node

## Model Information

| Model | Parameters | Quality | Speed | VRAM |
|-------|------------|---------|-------|------|
| starvector-1b-im2svg | 1 Billion | Good | Fast | ~4GB |
| starvector-8b-im2svg | 8 Billion | Best | Slower | ~16GB |

Models are automatically downloaded to `ComfyUI/models/vector/<model-name>/` on first use (e.g., `starvector-1b-im2svg/`).

## Offline and CustomComfy Usage

Workers without Hugging Face access can provide the complete model repository through `model_path`.
The value may be either:

- An extracted Hugging Face repository directory containing `config.json` and all model/tokenizer files.
- A repository TAR archive. Archives are extracted once into `ComfyUI/models/vector/.resource_cache/` and reused by the running ComfyUI installation.

For a CustomComfy job, declare a pinned repository snapshot AIR as a resource and put the exact
same AIR in the loader's `model_path` input. The worker rewrites it to the mounted resource path:

```json
{
  "resources": [
    "urn:air:starvector:repository:huggingface:starvector/starvector-1b-im2svg@<commit-sha>.tar"
  ],
  "workflow": {
    "1": {
      "class_type": "StarVectorModelLoader",
      "inputs": {
        "model_name": "starvector-1b-im2svg",
        "device": "auto",
        "dtype": "float16",
        "model_path": "urn:air:starvector:repository:huggingface:starvector/starvector-1b-im2svg@<commit-sha>.tar"
      }
    }
  }
}
```

When `model_path` is supplied, loading is strictly local (`local_files_only=True`). An invalid,
missing, or incomplete resource fails immediately and never falls back to a network download.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` on production workers as defense in depth.

The patched StarVector dependency bundled by this node supports a fully local
`starvector-1b-im2svg` snapshot. The 8B model still resolves its StarCoder2 base model separately
and is not guaranteed to be fully offline without that base model in the Hugging Face cache.

## Tips

- **For best results**: Use clean, high-contrast images with simple shapes
- **Memory management**: The 8B model requires significant VRAM; use float16 for efficiency
- **SVG complexity**: Higher `max_length` allows more complex SVGs but takes longer to generate
- **Optimization**: The Save SVG node can optionally optimize the output by removing comments and whitespace

## Troubleshooting

### "No SVG rasterization library found"
Install at least one SVG rasterization library:
```bash
pip install cairosvg
# or
pip install svglib reportlab
```

### Model download fails
- Check your internet connection
- Ensure you have enough disk space in `models/vector/<model-name>/`
- Try running ComfyUI with `--verbose` for more details
- If you see a 403 error for starcoder config, the fallback loader will attempt an alternative method

### CUDA out of memory
- Use the 1B model instead of 8B
- Use `float16` dtype
- Reduce `max_length` parameter
- Close other GPU applications

### SVG output is truncated or incomplete
- Increase the `max_length` parameter
- Some complex images may require values up to 8000-16000

## License

MIT License - See LICENSE file for details.

## Credits

- [StarVector](https://huggingface.co/starvector) by the StarVector team
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.
