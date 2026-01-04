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
| `Qwen/Qwen-Image` | [🤗 Link](https://huggingface.co/Qwen/Qwen-Image) |
| `Qwen/Qwen-Image-2512` | [🤗 Link](https://huggingface.co/Qwen/Qwen-Image-2512) |

**Recommended for Modal deployment:**
- **H100 (80GB)**: All models above
- **A100 (40GB)**: SD3.5 models, Z-Image-Turbo
- **A100 (80GB)**: All models including large variants

### Image Editing

Models that edit existing images with text guidance via the `/images/edits` endpoint:

| Model | HuggingFace Link | Multi-Image Support | Notes |
|-------|------------------|---------------------|-------|
| `black-forest-labs/FLUX.1-Kontext-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | ❌ Single image | ⚠️ Requires `diffusers==0.36.0` |
| `diffusers/FLUX.2-dev-bnb-4bit` | [🤗 Link](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | ✅ Up to 10 images | Quantized, slower |
| `black-forest-labs/FLUX.2-dev` | [🤗 Link](https://huggingface.co/black-forest-labs/FLUX.2-dev) | ✅ Up to 10 images | ⚠️ Requires H200 64GB+ RAM, variable inference (17s-2min) |
| `Qwen/Qwen-Image-Edit` | [🤗 Link](https://huggingface.co/Qwen/Qwen-Image-Edit) | ❌ Single image | Qwen edit base |
| `Qwen/Qwen-Image-Edit-2509` | [🤗 Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | ✅ Up to 3 images | Sept 2025 variant |
| `Qwen/Qwen-Image-Edit-2511` | [🤗 Link](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | ✅ Up to 3 images | Nov 2025 variant (latest) |

> ⚠️ **Performance Note**: The `diffusers/FLUX.2-dev-bnb-4bit` model tends to have high inference times for image editing tasks, even when running entirely on CUDA. Consider using FLUX.1-Kontext-dev for single-image edits or Qwen models for multi-image editing.

#### Generated Examples

<div id="imageModal" class="image-modal" onclick="document.getElementById('imageModal').style.display='none'">
  <img class="modal-content" id="modalImage">
</div>


<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555903/image_a_brown_dog_playing_in_the_park_flux2_7_2_yunwiy.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555903/image_a_green_tree_in_a_beautiful_forest_flux2_2_1_ueomai.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555905/image_sc_1_eugmfx.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555904/image_hr_3_ejrkgo.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555902/image_7_yoap68.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555904/image_sd_6_0_kmm6dv.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
  <div style="text-align: center;">
    <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1767555905/image_xd_5_0_x6c4jj.png" alt="Generated image" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: pointer;" onclick="document.getElementById('imageModal').style.display='block'; document.getElementById('modalImage').src=this.src">
  </div>
</div>

<p style="text-align: center; color: #888; font-size: 14px; margin-top: 10px; font-style: italic;">
Click on any image to view in full size (click again to close)
</p>

### Video Generation

Models that generate videos from text prompts via the `/videos` endpoint:

#### Model Specifications

> The metrics shown were taken when running the models on an NVIDIA H100.

| Model | HuggingFace Link | GPU Requirement | Inference Time | End-to-End Time | Peak VRAM | Notes |
|-------|------------------|-----------------|----------------|-----------------|-----------|-------|
| `wan2.1-turbo-fp8` | [🤗 Link](https://huggingface.co/Aquiles-ai/Wan2.1-Turbo-fp8) | <80GB VRAM | 32 seconds | 38 seconds | 30.09 GB | Fast, FP8 quantized, 4 steps |
| `wan2.1` | [🤗 Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | ~50GB VRAM | 21 min 16 sec | 21 min 24 sec | 49.95 GB | High quality, 40 inference steps |
| `wan2.1-turbo` | [🤗 Link](https://huggingface.co/Aquiles-ai/Wan2.1-Turbo) | ~43GB VRAM | ~33 seconds | ~39 seconds | 42.74 GB | Fast turbo variant, 4 steps |
| `wan2.1-3B` | [🤗 Link](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | ~16GB VRAM | 1 min 3 sec | 1 min 8 sec | 15.78 GB | Lightweight model, 40 steps |
| `wan2.2` | [🤗 Link](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | ~80GB VRAM | ~30 minutes | ~30 min 35 sec | ~77 GB | High quality, 40 inference steps |
| `wan2.2-turbo` | [🤗 Link](https://huggingface.co/Aquiles-ai/Wan2.2-Turbo) | ~80GB VRAM | ⚡ ~3 minutes | ~3 min 15 sec | ~77 GB | **9.5x faster** - Same quality in 4 steps! |
| `hunyuanVideo-1.5-480p` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-480p) | ~57GB VRAM | 6 min 4 sec | 6 min 22 sec | 56.51 GB | 480p resolution, 50 steps |
| `hunyuanVideo-1.5-480p-fp8` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-480p-fp8) | ~49GB VRAM | 5 min 57 sec | 6 min 14 sec | 48.93 GB | 480p FP8 quantized, 50 steps |
| `hunyuanVideo-1.5-480p-turbo` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-480p-Turbo) | ~49GB VRAM | ⚡ 8 seconds | 17 seconds | 48.82 GB | **Ultra fast** 480p turbo, 4 steps |
| `hunyuanVideo-1.5-480p-turbo-fp8` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-480p-Turbo-fp8) | ~41GB VRAM | ⚡ 8 seconds | 20 seconds | 40.76 GB | **Ultra fast** 480p turbo FP8, 4 steps |
| `hunyuanVideo-1.5-720p` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-720p) | ~57GB VRAM | 25 min 53 sec | 26 min 48 sec | 56.56 GB | 720p resolution, 50 steps |
| `hunyuanVideo-1.5-720p-fp8` | [🤗 Link](https://huggingface.co/Aquiles-ai/HunyuanVideo-1.5-720p-fp8) | ~49GB VRAM | 25 min 50 sec | 26 min 45 sec | 48.98 GB | 720p FP8 quantized, 50 steps |

#### Output Examples

> **Generated with prompt:** A direct continuation of the existing shot of a chameleon crawling slowly along a mossy branch. 
>    Begin with the chameleon already mid-step, camera tracking right at the same close, eye-level angle. 
>    After three seconds, its eyes swivel independently, one pausing to glance toward the lens before it 
>    resumes moving forward. Maintain the 100 mm anamorphic lens with shallow depth of field, dappled 
>    rainforest light, faint humidity haze, and subtle film grain. The moss texture and background greenery should 
>    remain consistent, with the chameleon's deliberate gait flowing naturally as if no cut occurred.

<table style="width: 100%; border-collapse: collapse;">
  <thead>
    <tr style="background-color: #2d2d2d;">
      <th style="padding: 16px; text-align: center; font-size: 18px; font-weight: 600; border: 1px solid #404040;">Model</th>
      <th style="padding: 16px; text-align: center; font-size: 18px; font-weight: 600; border: 1px solid #404040;">Output Video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.2
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=video_modal_deploy_aftsi6&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.2-turbo
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=wan2_2_turbo_th0wcl&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.1-turbo-fp8
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=wan2_1_turbo_fp8_lj7woy&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.1
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=wan2_1_base_14b_kc2f9e&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.1-turbo
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=wan2_1_turbo_gw2u7j&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        wan2.1-3B
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=wan2_1_3B_qimtse&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-480p
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuan_480p_st_kj9abu&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-480p-fp8
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuan_480p_fp8_j440gb&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-480p-turbo
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuan_480p_turbo_fnkwwy&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-480p-turbo-fp8
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuan_480p_turbo_fp8_mthiiq&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-720p
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuanvideo_720_n3nznq&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
    <tr>
      <td style="padding: 20px; text-align: center; vertical-align: middle; font-size: 20px; font-weight: 500; font-family: 'Courier New', monospace; border: 1px solid #404040;">
        hunyuanVideo-1.5-720p-fp8
      </td>
      <td style="padding: 12px; border: 1px solid #404040;">
        <iframe 
          src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=hunyuanvideo-720p-fp8_vc3str&profile=cld-default" 
          width="100%" 
          height="400" 
          allow="autoplay; fullscreen; encrypted-media; picture-in-picture" 
          allowfullscreen 
          frameborder="0"
          style="border-radius: 8px;">
        </iframe>
      </td>
    </tr>
  </tbody>
</table>

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
