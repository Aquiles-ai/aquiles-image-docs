## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 24GB+ VRAM
- 10GB+ free disk space

### Step-by-step Installation

#### 1. Install Core Dependencies

Install PyTorch and essential libraries:
```bash
uv pip install torch==2.8 numpy packaging torchvision
```

> **Note:** We use PyTorch 2.8 because pre-compiled `flash_attn` wheels are available for this version, providing optimal performance. If you're using newer PyTorch versions or don't need `flash_attn`, you can install the latest version instead:
> ```bash
> uv pip install torch numpy packaging torchvision
> ```

#### 2. Install Flash Attention (Optional but Recommended)

Flash Attention significantly improves inference speed and memory efficiency. Install the pre-compiled wheel:
```bash
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
```

> **Note:** This wheel is for:
> - PyTorch 2.8
>
> If you have a different configuration, find the appropriate wheel at [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) or compile from source.

#### 3. Install Transformers and Additional Libraries
```bash
uv pip install transformers ftfy kernels
```

#### 4. Install Diffusers

Choose one of the following options:

**Option A: From PyPI (Stable)**
```bash
uv pip install diffusers
```

**Option B: From Source (Required for FLUX.2 and Z-Image-Turbo)**
```bash
uv pip install git+https://github.com/huggingface/diffusers.git
```

**Option C: To run FLUX.1-Kontext-dev correctly, you must install this version**
```bash
uv pip install diffusers==0.36.0
```

> **Important:** If you plan to use **FLUX.2** or **Z-Image-Turbo** models, you **must** install from source (Option B).

#### 5. Install Aquiles-Image

Choose one of the following options:

**From PyPI (Recommended)**
```bash
uv pip install aquiles-image
```

**From Source (Latest Development Version)**
```bash
uv pip install git+https://github.com/Aquiles-ai/Aquiles-Image.git
```

### Authentication for Gated Models

Some models on Hugging Face require authentication before downloading. To authenticate:
```bash
hf auth login
```

Enter your Hugging Face token when prompted. You can create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Launch Your Server

Once installation is complete, start your Aquiles-Image server:
```bash
aquiles-image serve --host "0.0.0.0" --port 5500 --model "stabilityai/stable-diffusion-3.5-medium"
```

Your server will be available at `http://0.0.0.0:5500` and ready to generate images!

### Verify Installation

Test that everything works correctly:
```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024",
    response_format="b64_json"
)

print(f"Downloading image\n")

image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print(f"Image downloaded successfully\n")
```

### Troubleshooting

- **CUDA errors**: Ensure your GPU drivers are up to date
- **Out of memory**: Most models require 24GB+ VRAM
- **Authentication errors**: Make sure you've logged in with `hf auth login`
- **Model not found**: Check that the model name is correct and you have access to it
- **Flash Attention errors**: Verify your Python version, PyTorch version, and CUDA version match the wheel requirements


## Installation for Video Generation

Video generation requires additional dependencies and has stricter hardware requirements compared to standard image generation.

### Prerequisites

- Python 3.8+
- **CUDA-compatible GPU with 80GB+ VRAM** (e.g., NVIDIA H100 or A100-80GB)
- 20GB+ free disk space

> **Warning:** Video generation models like Wan2.2 require significantly more VRAM than image generation models. A consumer GPU will not be sufficient.

### Step-by-step Installation

#### 1. Install Core Dependencies

Install PyTorch and essential libraries:
```bash
uv pip install torch==2.8 numpy packaging torchvision
```

#### 2. Install Flash Attention (Required)

Flash Attention is **mandatory** for video generation due to memory requirements:
```bash
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl
```

> **Note:** This wheel is for:
> - PyTorch 2.8
>
> If you have a different configuration, find the appropriate wheel at [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases) or compile from source.

#### 3. Install Transformers and Additional Libraries
```bash
uv pip install transformers ftfy kernels
```

#### 4. Install Diffusers from Source

Video generation requires the latest Diffusers features:
```bash
uv pip install git+https://github.com/huggingface/diffusers.git
```

#### 5. Install LightX2V

LightX2V is required for video generation and must be installed from source:
```bash
uv pip install git+https://github.com/ModelTC/LightX2V.git
```

> **Note:** LightX2V is not available on PyPI and must be installed from the GitHub repository.

#### 6. Install Aquiles-Image

Choose one of the following options:

**From PyPI (Recommended)**
```bash
uv pip install aquiles-image
```

**From Source (Latest Development Version)**
```bash
uv pip install git+https://github.com/Aquiles-ai/Aquiles-Image.git
```

### Authentication for Gated Models

Video generation models require Hugging Face authentication:
```bash
hf auth login
```

Enter your Hugging Face token when prompted. You can create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Launch Video Generation Server

Start your server with a video generation model:
```bash
aquiles-image serve --host "0.0.0.0" --port 5500 --model "wan2.2"
```

Your video generation server will be available at `http://0.0.0.0:5500`!

### Generate Your First Video
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

video = client.videos.generate(
    model="wan2.2",
    prompt="A cat playing with a ball of yarn",
)

print("Video generation started:", video)
```

### Important Limitations

- **Sequential Processing**: Only one video can be generated at a time
- **Inference Time**: Approximately 5+ minutes per video on an H100 GPU
- **Hardware Requirements**: Requires professional-grade GPUs (80GB+ VRAM)
- **Experimental Feature**: Video generation is still in active development

### Troubleshooting

- **Out of memory**: Verify your GPU has at least 80GB VRAM
- **Flash Attention errors**: Flash Attention is mandatory for video generation
- **LightX2V import errors**: Ensure it was installed from source correctly
- **Slow inference**: Video generation is computationally intensive; 5+ minutes is expected
