"""
StarVector Custom Nodes for ComfyUI
Generates SVG files from images using StarVector models
"""

import os
import io
import re
import torch
import numpy as np
import folder_paths
from PIL import Image

from .model_resources import ModelResourceError, resolve_model_resource

# Register the model folder
VECTOR_MODELS_DIR = os.path.join(folder_paths.models_dir, "vector")
os.makedirs(VECTOR_MODELS_DIR, exist_ok=True)

# Add to folder_paths so ComfyUI knows about it
if "vector" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["vector"] = ([VECTOR_MODELS_DIR], {".safetensors", ".bin", ".pt", ".pth"})


class StarVectorModelLoader:
    """
    Loads a StarVector model for SVG generation.
    Models are automatically downloaded from HuggingFace on first use.
    """
    
    MODELS = {
        "starvector-1b-im2svg": "starvector/starvector-1b-im2svg",
        "starvector-8b-im2svg": "starvector/starvector-8b-im2svg",
    }
    
    def __init__(self):
        self.loaded_model = None
        self.loaded_model_name = None
        self.processor = None
        self.tokenizer = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(cls.MODELS.keys()), {
                    "default": "starvector-1b-im2svg"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
                "dtype": (["float16", "float32", "bfloat16"], {
                    "default": "float16"
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local model directory or repository TAR. CustomComfy AIRs are supported."
                }),
            },
        }
    
    RETURN_TYPES = ("STARVECTOR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "StarVector"
    
    def load_model(self, model_name, device, dtype, model_path=""):
        from transformers import AutoModelForCausalLM, AutoConfig

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[dtype]

        # Get the HuggingFace model path used only by the legacy online fallback.
        hf_model_path = self.MODELS[model_name]

        # Set local directory to download model to a named subdirectory
        local_dir = os.path.join(VECTOR_MODELS_DIR, model_name)
        os.makedirs(local_dir, exist_ok=True)

        resource_path = model_path.strip()
        if resource_path:
            try:
                resolved_model_path = resolve_model_resource(
                    resource_path,
                    os.path.join(VECTOR_MODELS_DIR, ".resource_cache"),
                )
            except ModelResourceError as error:
                raise RuntimeError(f"Invalid StarVector model resource: {error}") from error
            load_path = resolved_model_path
            model_exists = True
            print(f"[StarVector] Loading declared local resource: {resource_path}")
        else:
            load_path = local_dir
            model_exists = os.path.exists(os.path.join(local_dir, "config.json"))

        print(f"[StarVector] Model: {hf_model_path}")
        print(f"[StarVector] Resolved model directory: {load_path}")
        print(f"[StarVector] Device: {device}, dtype: {dtype}")

        # Save current environment variables
        old_offline = os.environ.get('HF_HUB_OFFLINE')
        old_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')

        try:
            if model_exists:
                # Load from local directory
                print(f"[StarVector] Loading from local directory...")

                # Set offline mode to prevent any network access
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'

                # Load config
                config = AutoConfig.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )

                # Pass parent model directory to starvector via kwargs
                print(f"[StarVector] Loading model with parent_model_dir={load_path}")

                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    local_files_only=True,
                    parent_model_dir=load_path,  # Pass to starvector model (no underscore)
                )
            else:
                if _offline_mode_enabled(old_offline) or _offline_mode_enabled(old_transformers_offline):
                    raise RuntimeError(
                        f"StarVector model is not installed at {local_dir} and offline mode is enabled. "
                        "Provide model_path or install the complete repository snapshot locally."
                    )

                # Download to local directory
                print(f"[StarVector] Downloading model to local directory...")
                print(f"[StarVector] Note: This will download the model and tokenizer...")

                # Keep the legacy online behavior for desktop users. Production
                # workers should provide model_path and never enter this branch.
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    parent_model_dir=local_dir,  # Pass to starvector model (no underscore)
                )

        except Exception as e:
            print(f"[StarVector] Error loading model: {e}")
            print(f"[StarVector] Full error details:")
            import traceback
            traceback.print_exc()

            raise RuntimeError(
                f"Failed to load StarVector model. "
                f"Error: {e}\n\n"
                f"Please ensure the model files are available at: {load_path}\n"
                f"The model requires local tokenizer files to avoid accessing the gated bigcode repository."
            )
        finally:
            # Restore original environment variables
            if old_offline is not None:
                os.environ['HF_HUB_OFFLINE'] = old_offline
            elif 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']

            if old_transformers_offline is not None:
                os.environ['TRANSFORMERS_OFFLINE'] = old_transformers_offline
            elif 'TRANSFORMERS_OFFLINE' in os.environ:
                del os.environ['TRANSFORMERS_OFFLINE']
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        # Get processor and tokenizer from the model
        processor = model.model.processor
        tokenizer = model.model.svg_transformer.tokenizer
        
        print(f"[StarVector] Model loaded successfully!")
        
        # Return a dictionary containing the model and related components
        model_dict = {
            "model": model,
            "processor": processor,
            "tokenizer": tokenizer,
            "device": device,
            "model_name": model_name,
        }
        
        return (model_dict,)


def _offline_mode_enabled(value):
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


class StarVectorImage2SVG:
    """
    Converts an image to SVG using a loaded StarVector model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STARVECTOR_MODEL",),
                "image": ("IMAGE",),
                "max_length": ("INT", {
                    "default": 4000,
                    "min": 100,
                    "max": 16000,
                    "step": 100,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("SVG", "STRING",)
    RETURN_NAMES = ("svg", "svg_string",)
    FUNCTION = "generate_svg"
    CATEGORY = "StarVector"
    
    def generate_svg(self, model, image, max_length):
        starvector = model["model"]
        processor = model["processor"]
        device = model["device"]
        
        # Convert from ComfyUI image format [B,H,W,C] to PIL
        # Take the first image from the batch
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(img_np, mode='RGB')
        
        # Process the image
        processed = processor(image_pil, return_tensors="pt")
        pixel_values = processed['pixel_values'].to(device)

        # Ensure we have batch dimension: [batch, channels, height, width]
        if pixel_values.dim() == 3:
            # Missing batch dimension, add it
            pixel_values = pixel_values.unsqueeze(0)

        print(f"[StarVector] Processed image shape: {pixel_values.shape}")

        batch = {"image": pixel_values}
        
        # Generate SVG
        with torch.no_grad():
            raw_svg = starvector.generate_im2svg(batch, max_length=max_length)[0]
        
        # Create SVG data structure
        svg_data = {
            "svg_string": raw_svg,
            "source": "starvector",
        }
        
        return (svg_data, raw_svg,)


class SVGPreview:
    """
    Previews an SVG by rasterizing it to an image.
    Uses cairosvg for high-quality rasterization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg": ("SVG",),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "background": (["white", "black", "transparent", "gray"], {
                    "default": "white"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "preview_svg"
    CATEGORY = "StarVector"
    
    def preview_svg(self, svg, width, height, background):
        svg_string = svg["svg_string"]
        
        # Try to use cairosvg first, fall back to other methods
        try:
            import cairosvg
            
            # Set background color
            bg_colors = {
                "white": "#FFFFFF",
                "black": "#000000",
                "transparent": None,
                "gray": "#808080",
            }
            bg = bg_colors[background]
            
            # Rasterize SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                bytestring=svg_string.encode('utf-8'),
                output_width=width,
                output_height=height,
                background_color=bg,
            )
            
            # Convert to PIL Image
            image_pil = Image.open(io.BytesIO(png_bytes))
            
        except ImportError:
            # Fall back to trying starvector's built-in rasterization
            try:
                from starvector.data.util import process_and_rasterize_svg
                _, raster_image = process_and_rasterize_svg(svg_string, canvas_size=max(width, height))
                image_pil = raster_image.resize((width, height), Image.Resampling.LANCZOS)
            except ImportError:
                # Final fallback: try svglib
                try:
                    from svglib.svglib import svg2rlg
                    from reportlab.graphics import renderPM
                    
                    drawing = svg2rlg(io.StringIO(svg_string))
                    if drawing:
                        # Scale to desired size
                        scale_x = width / drawing.width if drawing.width else 1
                        scale_y = height / drawing.height if drawing.height else 1
                        scale = min(scale_x, scale_y)
                        drawing.scale(scale, scale)
                        
                        png_bytes = renderPM.drawToString(drawing, fmt="PNG")
                        image_pil = Image.open(io.BytesIO(png_bytes))
                    else:
                        raise ValueError("Failed to parse SVG")
                except ImportError:
                    raise ImportError(
                        "No SVG rasterization library found. Please install one of: "
                        "cairosvg (`pip install cairosvg`), "
                        "starvector (`pip install starvector`), or "
                        "svglib (`pip install svglib`)"
                    )
        
        # Convert to RGB if necessary
        if image_pil.mode == 'RGBA':
            # Create background
            bg_color = {
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "transparent": (255, 255, 255),
                "gray": (128, 128, 128),
            }[background]
            bg_image = Image.new('RGB', image_pil.size, bg_color)
            bg_image.paste(image_pil, mask=image_pil.split()[3])
            image_pil = bg_image
        elif image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Resize to exact dimensions
        image_pil = image_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert to ComfyUI format [B,H,W,C]
        img_np = np.array(image_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        return (img_tensor,)


class SVGPreviewFromString:
    """
    Previews an SVG from a raw string by rasterizing it to an image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "background": (["white", "black", "transparent", "gray"], {
                    "default": "white"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "preview_svg"
    CATEGORY = "StarVector"
    
    def preview_svg(self, svg_string, width, height, background):
        # Create SVG data structure and use the main preview node
        svg_data = {"svg_string": svg_string, "source": "string"}
        preview_node = SVGPreview()
        return preview_node.preview_svg(svg_data, width, height, background)


class SaveSVG:
    """
    Saves an SVG to a file.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg": ("SVG",),
                "filename_prefix": ("STRING", {
                    "default": "starvector",
                }),
            },
            "optional": {
                "optimize": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "save_svg"
    CATEGORY = "StarVector"
    
    def save_svg(self, svg, filename_prefix, optimize=True):
        svg_string = svg["svg_string"]
        
        # Optimize SVG if requested
        if optimize:
            svg_string = self._optimize_svg(svg_string)
        
        # Generate unique filename
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, 1, 1
        )
        
        filename = f"{filename}_{counter:05}.svg"
        filepath = os.path.join(full_output_folder, filename)
        
        # Save the SVG
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        
        print(f"[StarVector] SVG saved to: {filepath}")
        
        return (filepath,)
    
    def _optimize_svg(self, svg_string):
        """Basic SVG optimization - remove unnecessary whitespace and comments"""
        # Remove XML comments
        svg_string = re.sub(r'<!--.*?-->', '', svg_string, flags=re.DOTALL)
        # Remove excessive whitespace between tags
        svg_string = re.sub(r'>\s+<', '><', svg_string)
        # Remove leading/trailing whitespace
        svg_string = svg_string.strip()
        return svg_string


class LoadSVG:
    """
    Loads an SVG file from disk.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if f.lower().endswith('.svg'):
                files.append(f)
        
        return {
            "required": {
                "svg_file": (sorted(files) if files else ["none"], {
                    "default": files[0] if files else "none"
                }),
            },
        }
    
    RETURN_TYPES = ("SVG", "STRING",)
    RETURN_NAMES = ("svg", "svg_string",)
    FUNCTION = "load_svg"
    CATEGORY = "StarVector"
    
    def load_svg(self, svg_file):
        if svg_file == "none":
            raise ValueError("No SVG file selected")
        
        input_dir = folder_paths.get_input_directory()
        filepath = os.path.join(input_dir, svg_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            svg_string = f.read()
        
        svg_data = {
            "svg_string": svg_string,
            "source": "file",
            "filename": svg_file,
        }
        
        return (svg_data, svg_string,)


class SVGToString:
    """
    Extracts the SVG string from an SVG data object.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg": ("SVG",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("svg_string",)
    FUNCTION = "convert"
    CATEGORY = "StarVector"
    
    def convert(self, svg):
        return (svg["svg_string"],)


class StringToSVG:
    """
    Creates an SVG data object from a string.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
        }
    
    RETURN_TYPES = ("SVG",)
    RETURN_NAMES = ("svg",)
    FUNCTION = "convert"
    CATEGORY = "StarVector"
    
    def convert(self, svg_string):
        svg_data = {
            "svg_string": svg_string,
            "source": "string",
        }
        return (svg_data,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "StarVectorModelLoader": StarVectorModelLoader,
    "StarVectorImage2SVG": StarVectorImage2SVG,
    "SVGPreview": SVGPreview,
    "SVGPreviewFromString": SVGPreviewFromString,
    "SaveSVG": SaveSVG,
    "LoadSVG": LoadSVG,
    "SVGToString": SVGToString,
    "StringToSVG": StringToSVG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVectorModelLoader": "StarVector Model Loader",
    "StarVectorImage2SVG": "StarVector Image to SVG",
    "SVGPreview": "SVG Preview",
    "SVGPreviewFromString": "SVG Preview (String)",
    "SaveSVG": "Save SVG",
    "LoadSVG": "Load SVG",
    "SVGToString": "SVG to String",
    "StringToSVG": "String to SVG",
}
