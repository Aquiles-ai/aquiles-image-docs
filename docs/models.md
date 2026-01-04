# Supported Models

Aquiles-Image supports a wide range of state-of-the-art diffusion models for image generation, editing, and video creation. All models are optimized for production use with fast inference times.

## Native Support Models

These models have native implementations with optimized performance and full feature support.

### Text-to-Image Generation

Models that generate images from text prompts via the `/images/generations` endpoint:

| Model | HuggingFace Link |
|-------|------------------|
| `stabilityai/stable-diffusion-3-medium` | [🤗 Link](https://huggingface.co/stabilityai/stable-diffusion-3-medium) |
| `stabilityai/stable-diffusion-3.5-medium` | [🤗 Link](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| `stabilityai/stable-diffusion-3.5-large` | [🤗 Link](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) |
| `stabilityai/stable-diffusion-3.5-large-turbo` | [🤗 Link](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) |
| `black-forest-labs/FLUX.1-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `black-forest-labs/FLUX.1-schnell` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| `black-forest-labs/FLUX.1-Krea-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) |
| `black-forest-labs/FLUX.2-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| `diffusers/FLUX.2-dev-bnb-4bit` | [🤗 Link](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) |
| `Tongyi-MAI/Z-Image-Turbo` | [🤗 Link](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |

**Recommended for Modal deployment:**
- **H100 (80GB)**: All models above
- **A100 (40GB)**: SD3.5 models, Z-Image-Turbo
- **A100 (80GB)**: All models including large variants

### Image Editing

Models that edit existing images with text guidance via the `/images/edits` endpoint:

| Model | HuggingFace Link | Notes |
|-------|------------------|-------|
| `black-forest-labs/FLUX.1-Kontext-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | ⚠️ Requires `diffusers==0.36.0` |
| `diffusers/FLUX.2-dev-bnb-4bit` | [🤗 Link](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | Quantized, slower for editing |

> ⚠️ **Important**: `FLUX.1-Kontext-dev` requires exactly `diffusers==0.36.0` to avoid errors. See configuration guide for details.

> ⚠️ **Performance Note**: The `diffusers/FLUX.2-dev-bnb-4bit` model tends to have high inference times for image editing tasks, even when running entirely on CUDA. Consider using FLUX.1-Kontext-dev for better performance.

### Video Generation

Models that generate videos from text prompts via the `/videos` endpoint:

| Model | HuggingFace Link | GPU Requirement | Inference Time | End-to-End Time | Peak VRAM | Output Video | Notes |
|-------|------------------|-----------------|----------------|-----------------|-----------|--------------|-------|
| `wan2.1-turbo-fp8` | TBD | <80GB VRAM | 32 seconds | 38 seconds | 30.09 GB | TBD | Fast, FP8 quantized, 4 steps |
| `wan2.1-14B` | TBD | ~50GB VRAM | 21 min 16 sec | 21 min 24 sec | 49.95 GB | TBD | High quality, 40 inference steps |
| `wan2.1-turbo` | TBD | ~43GB VRAM | ~33 seconds | ~39 seconds | 42.74 GB | TBD | Fast turbo variant, 4 steps |
| `wan2.1-3B` | TBD | ~16GB VRAM | 1 min 3 sec | 1 min 8 sec | 15.78 GB | TBD | Lightweight model, 40 steps |
| `wan2.2` | [🤗 Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | ~80GB VRAM | ~30 minutes | ~30 min 35 sec | ~77 GB |  <iframe src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=video_modal_deploy_aftsi6&profile=cld-default" width="640" height="360" style="height: auto; width: 100%; aspect-ratio: 640 / 360;" allow="autoplay; fullscreen; encrypted-media; picture-in-picture" allowfullscreen frameborder="0"></iframe> | High quality, 40 inference steps |
| `wan2.2-turbo` | [🤗 Link](https://huggingface.co/Aquiles-ai/Wan2.2-Turbo) | ~80GB VRAM | ⚡ ~3 minutes | ~3 min 15 sec | ~77 GB | TBD | **9.5x faster** - Same quality in 4 steps! |
| `hunyuanVideo-1.5-480p` | TBD | ~57GB VRAM | 6 min 4 sec | 6 min 22 sec | 56.51 GB | TBD | 480p resolution, 50 steps |
| `hunyuanVideo-1.5-480p-fp8` | TBD | ~49GB VRAM | 5 min 57 sec | 6 min 14 sec | 48.93 GB | TBD | 480p FP8 quantized, 50 steps |
| `hunyuanVideo-1.5-480p-turbo` | TBD | ~49GB VRAM | ⚡ 8 seconds | 17 seconds | 48.82 GB | TBD | **Ultra fast** 480p turbo, 4 steps |
| `hunyuanVideo-1.5-480p-turbo-fp8` | TBD | ~41GB VRAM | ⚡ 8 seconds | 20 seconds | 40.76 GB | TBD | **Ultra fast** 480p turbo FP8, 4 steps |
| `hunyuanVideo-1.5-720p` | TBD | ~57GB VRAM | 25 min 53 sec | 26 min 48 sec | 56.56 GB | TBD | 720p resolution, 50 steps |
| `hunyuanVideo-1.5-720p-fp8` | TBD | ~49GB VRAM | 25 min 50 sec | 26 min 45 sec | 48.98 GB | TBD | 720p FP8 quantized, 50 steps |

> **Generated with prompt:** A direct continuation of the existing shot of a chameleon crawling slowly along a mossy branch. 
>    Begin with the chameleon already mid-step, camera tracking right at the same close, eye-level angle. 
>    After three seconds, its eyes swivel independently, one pausing to glance toward the lens before it 
>    resumes moving forward. Maintain the 100 mm anamorphic lens with shallow depth of field, dappled 
>    rainforest light, faint humidity haze, and subtle film grain. The moss texture and background greenery should 
>    remain consistent, with the chameleon's deliberate gait flowing naturally as if no cut occurred.

**Video generation requirements:**
- Minimum 80GB VRAM (NVIDIA H100 or A100-80GB)
- Generation time varies by model (see table above)
- Uses polling API for status updates
- Both models produce equivalent quality output

## AutoPipeline Support (Experimental)

Aquiles-Image supports additional models through the experimental **AutoPipeline** feature, which uses `AutoPipelineForText2Image` from Hugging Face's Diffusers library.

### Status

![Experimental](https://img.shields.io/badge/Status-Experimental-orange)
![Requires AutoPipeline](https://img.shields.io/badge/Requires-AutoPipelineForText2Image-blue)
![Slower Performance](https://img.shields.io/badge/Performance-Slower-yellow)

### What is AutoPipeline?

AutoPipeline automatically detects and loads compatible diffusion models without manual configuration. This provides greater model flexibility but with performance trade-offs.

### Compatible Models

Any model that works with `AutoPipelineForText2Image` out-of-the-box, including:

- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `runwayml/stable-diffusion-v1-5`
- And many more from HuggingFace Hub

### Usage on Modal

**Update your image configuration:**

```python
aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential")
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .uv_pip_install(
        "torch==2.8",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==4.57.3",
        "tokenizers==0.22.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")
    })
)

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
```

**Update the serve function:**

```python
@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A100:1",  # SDXL works well on A100
    scaledown_window=6 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=AQUILES_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "aquiles-image",
        "serve",
        "--host", "0.0.0.0",
        "--port", str(AQUILES_PORT),
        "--model", MODEL_NAME,
        "--set-steps", "30",
        "--auto-pipeline",  # Enable AutoPipeline
        "--api-key", "dummy-api-key",
    ]

    print(f"Starting Aquiles-Image with AutoPipeline model: {MODEL_NAME}")
    print(f"Command: {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)
```

### Client Usage Example

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://username--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"
)

result = client.images.generate(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    prompt="a beautiful sunset over mountains",
    size="1024x1024",
    response_format="b64_json"
)

# Save the image
image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Image generated successfully!")
```

### More Examples

**Stable Diffusion v1.5:**
```bash
# In your Modal serve function
cmd = [
    "aquiles-image", "serve",
    "--model", "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "--set-steps", "40",
    "--auto-pipeline",
    # ... other options
]
```

**SDXL Base with authentication:**
```bash
cmd = [
    "aquiles-image", "serve",
    "--model", "stabilityai/stable-diffusion-xl-base-1.0",
    "--set-steps", "30",
    "--auto-pipeline",
    "--api-key", "your-secret-key",
    # ... other options
]
```

### Important Limitations

| Limitation | Description |
|------------|-------------|
| 🐌 **Slower Inference** | AutoPipeline models have longer inference times compared to optimized native implementations |
| 🚫 **No LoRA Support** | LoRA adapters are not currently supported |
| 🚫 **No Adapters** | ControlNet, T2I-Adapter, and other adapter types are not supported |
| 🧪 **Experimental** | This feature is in active development and may have stability issues |
| 📦 **Limited Configs** | Only models that work out-of-the-box with default `AutoPipelineForText2Image` settings are supported |
| ❌ **No Image Editing** | AutoPipeline only supports text-to-image generation, not image editing |
| ❌ **No Video** | Video generation is not supported with AutoPipeline |

### Troubleshooting AutoPipeline

**1. Check model compatibility**

Test locally before deploying to Modal:
```python
from diffusers import AutoPipelineForText2Image

# Test if model loads correctly
pipe = AutoPipelineForText2Image.from_pretrained(
    "your-model-name",
    torch_dtype=torch.float16
)
```

**2. Verify VRAM requirements**

- Check the model card on HuggingFace for memory requirements
- Monitor GPU usage during inference
- Some models may need more VRAM than their native counterparts

**3. Use native implementations when possible**

For supported models (FLUX, SD3, etc.), always use native support for better performance:
```python
# ✅ Better: Native support
MODEL_NAME = "stabilityai/stable-diffusion-3.5-medium"
# No --auto-pipeline flag needed

# ❌ Slower: AutoPipeline
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
# Requires --auto-pipeline flag
```

**4. Check server logs**

Modal provides detailed logs for debugging:
```bash
# View real-time logs
modal app logs aquiles-image-server

# Look for errors marked with "X" emoji
# They indicate AutoPipeline loading issues
```

**5. Common issues**

| Issue | Solution |
|-------|----------|
| Model won't load | Verify model exists on HuggingFace and is public or you have access |
| Out of memory | Use a larger GPU or try a smaller model |
| Slow inference | Expected with AutoPipeline; consider requesting native support |
| Missing dependencies | Some models need additional packages; check model documentation |

### Requesting Native Support

If you frequently use a model with AutoPipeline, you can request native support for better performance:

1. Open an issue on [Aquiles-Image GitHub](https://github.com/Aquiles-ai/Aquiles-Image)
2. Include:
   - Model name and HuggingFace link
   - Your use case
   - Performance comparisons if available

Native implementations are prioritized based on community demand and model popularity.

## Model Updates

This list is actively maintained. New models are added regularly as they become available and tested. Check the [Aquiles-Image GitHub repository](https://github.com/Aquiles-ai/Aquiles-Image) for the latest updates.
