# Deploy Aquiles-Image on Modal

This guide shows you how to deploy Aquiles-Image on Modal's GPU infrastructure for production-ready image generation at scale.

## What You'll Deploy

A FastAPI server running Aquiles-Image with:
- **OpenAI-compatible API** - Drop-in replacement for OpenAI's image endpoints
- **GPU acceleration** - H100/A100 GPUs for fast inference
- **Auto-scaling** - Scale to zero when idle, scale up on demand
- **Persistent caching** - Model weights cached across restarts

## Prerequisites

- Modal account ([sign up here](https://modal.com))
- Modal CLI installed: `uv pip install modal`
- Modal authentication configured: `modal setup`
- HuggingFace account (for gated models)

## Quick Start

### 1. Install and Authenticate with Modal

```bash
# Install Modal
uv pip install modal

# Authenticate (this will open your browser to sign in with GitHub)
modal setup
```

This will create a Modal account or link your existing one. The authentication uses your GitHub account.

### 2. Create Your Deployment Script

Save this as `aquiles_modal.py`:

```python
import modal
import os

# Container image with all dependencies
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
        # HuggingFace token - use environment variable or Modal secrets
        "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")
    })
)

# Model configuration
MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"

# Persistent volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

# App configuration
app = modal.App("aquiles-image-server")

N_GPU = 1
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"H100:{N_GPU}",
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
        "--set-steps", "35",
        "--api-key", "dummy-api-key",
        "--device-map", "cuda",
    ]

    print(f"Starting Aquiles-Image with model: {MODEL_NAME}")
    print(f"Command: {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)
```

### 3. Configure HuggingFace Authentication

You have two options for providing your HuggingFace token:

**Option A: Using Modal Secrets (Recommended)**

```bash
# Create a Modal secret with your HF token
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

You can also create secrets through the [Modal Dashboard](https://modal.com/secrets), which provides templates for common services including HuggingFace.

Then your code uses:
```python
@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # ... other config
)
```

To verify your secrets:
```bash
# List all your secrets
modal secret list

# You should see "huggingface-secret" in the list
```

**Option B: Using Environment Variables**

```bash
# Set environment variable before deploying
export Hugging_face_token_for_deploy=hf_your_token_here
modal deploy aquiles_modal.py
```

The code will pick it up via:
```python
"HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")
```

> 💡 **Tip**: Option A (Modal Secrets) is more secure and recommended for production deployments. The token is encrypted and managed by Modal.

### 4. Deploy

```bash
modal deploy aquiles_modal.py
```

You'll get a URL like:
```
✓ Created web function serve => https://username--aquiles-image-server-serve.modal.run
```

## Configuration Guide

### Choosing a GPU

The default configuration uses **H100 GPUs** for maximum performance. Adjust based on your model:

| Model Type | Recommended GPU | N_GPU |
|------------|----------------|-------|
| FLUX.1-dev, FLUX.1-Krea-dev | `H100:1` | 1 |
| Stable Diffusion 3.5 (large) | `H100:1` | 1 |
| Stable Diffusion 3.5 (medium) | `A100:1` | 1 |
| FLUX.2-dev-bnb-4bit (quantized) | `H100:1` | 1 |

```python
# For smaller models, use A100 40GB
gpu=f"A100:{N_GPU}"

# For large models, use H100 80GB
gpu=f"H100:{N_GPU}"
```

### Supported Models

**Text-to-Image Models** (most common):

```python
# FLUX models (require H100 or large A100)
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"
# require H200
MODEL_NAME = "black-forest-labs/FLUX.2-dev"

# Stable Diffusion 3.5 (can run on A100)
MODEL_NAME = "stabilityai/stable-diffusion-3.5-medium"
# Stable Diffusion 3.5 (can run on H100)
MODEL_NAME = "stabilityai/stable-diffusion-3.5-large"
MODEL_NAME = "stabilityai/stable-diffusion-3.5-large-turbo"

# (can run on H100)
MODEL_NAME = "diffusers/FLUX.2-dev-bnb-4bit"

# Other models (can run on A100)
MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
```

**Image Editing Models**:

```python
MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_NAME = "diffusers/FLUX.2-dev-bnb-4bit"
```

### Special Configuration: FLUX.1-Kontext-dev

> ⚠️ **Important**: The `FLUX.1-Kontext-dev` model requires a specific version of diffusers to avoid errors.

For this model, modify your image setup:

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
        "diffusers==0.36.0",  # ⚠️ Fixed version for FLUX.1-Kontext-dev
        "transformers==4.57.3",
        "tokenizers==0.22.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")
    })
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
```

### Adjusting Server Parameters

#### Inference Steps

Control generation quality vs. speed:

```python
cmd = [
    "aquiles-image", "serve",
    "--set-steps", "35",  # Higher = better quality, slower
    # ... other options
]
```

#### Concurrency

Control how many requests one container handles simultaneously:

```python
@modal.concurrent(max_inputs=4)  # Optimal range: 3-5
```

**How it works:**
- **Lower concurrency (1-2)**: Faster individual generation times, but requires more containers for high traffic
- **Medium concurrency (3-5)**: ⭐ **Optimal balance** between speed and throughput
- **Higher concurrency (6+)**: Each generation becomes slower as GPU resources are shared

**Recommendation:** Keep `max_inputs` between **3-5** for the best balance of generation speed and cost efficiency. Going higher will make individual images generate more slowly as the GPU handles multiple requests simultaneously.

#### Scale-down Window

How long to wait before shutting down idle containers:

```python
scaledown_window=6 * MINUTES  # 6 minutes of idle time
```

Trade-offs:
- Shorter window = lower costs, more cold starts
- Longer window = faster response times, higher costs

#### API Key Security

Change the default API key for production:

```python
cmd = [
    "aquiles-image", "serve",
    "--api-key", "your-production-key-here",  # Change this!
    # ... other options
]
```

Or use Modal secrets:
```bash
modal secret create aquiles-api-key API_KEY=your-secret-key
```

```python
@app.function(
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aquiles-api-key")
    ],
    # ... other config
)
def serve():
    import subprocess
    import os
    
    cmd = [
        "aquiles-image", "serve",
        "--api-key", os.environ.get("API_KEY", "dummy-api-key"),
        # ... other options
    ]
```

## Using Your Deployment

### With OpenAI Client

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://username--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"  # or your production key
)

response = client.images.generate(
    model="black-forest-labs/FLUX.1-Krea-dev",
    prompt="A serene Japanese garden with cherry blossoms",
    size="1024x1024",
    response_format="b64_json"
)

# Save the image
image_bytes = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

### API Endpoints

| Endpoint | Purpose | Models |
|----------|---------|--------|
| `/images/generations` | Generate images from text | All text-to-image models |
| `/images/edits` | Edit existing images | FLUX.1-Kontext-dev, FLUX.2-dev-bnb-4bit |
| `/videos` | Generate videos from text | wan2.2 |

> **Note**: Depending on the model that is loaded, that will be the endpoint that is available.

### Interactive Documentation

Visit your deployment's `/docs` endpoint for Swagger UI:
```
https://username--aquiles-image-server-serve.modal.run/docs
```

## Testing Your Deployment

Add this to your `aquiles_modal.py`:

```python
@app.local_entrypoint()
async def test():
    from openai import OpenAI
    import base64
    
    url = serve.get_web_url()
    print(f"🚀 Testing server at: {url}\n")
    
    client = OpenAI(base_url=url, api_key="dummy-api-key")
    
    prompt = "A futuristic cityscape at sunset with flying cars"
    print(f"Generating image: {prompt}\n")
    
    result = client.images.generate(
        model=MODEL_NAME,
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json"
    )
    
    print("Saving image...\n")
    image_bytes = base64.b64decode(result.data[0].b64_json)
    with open("test_output.png", "wb") as f:
        f.write(image_bytes)
    
    print("Image saved as 'test_output.png'!")
```

Run the test:

```bash
modal run aquiles_modal.py
```

## Monitoring and Logs

### View Real-Time Logs

Stream logs for your deployed app:
```bash
modal app logs aquiles-image-server
```

Press `Ctrl+C` to stop streaming logs.

Add timestamps to log lines:
```bash
modal app logs aquiles-image-server --timestamps
```

### View App Dashboard

Every running app logs a dashboard link when it starts:
```
✓ Initialized. View app page at https://modal.com/apps/ap-XYZ1234
```

From the dashboard you can:
- View logs (both application and system-level)
- Monitor compute resources (CPU, RAM, GPU)
- See function call history and success/failure counts

### List All Apps

View all your deployed and running apps:
```bash
modal app list
```

## Cost Optimization Tips

1. **Choose the right GPU**: Use A100 for smaller models, H100 only for large FLUX models
2. **Adjust scale-down window**: Shorter windows reduce idle costs
3. **Optimize concurrency**: Keep between 3-5 for best speed/throughput balance
4. **Use quantized models**: 4-bit models like `FLUX.2-dev-bnb-4bit` are much cheaper
5. **Cache aggressively**: Modal Volumes prevent re-downloading models

**Performance vs Cost trade-off:**
- Lower concurrency (3) = Faster generations, more containers = higher cost
- Higher concurrency (5) = Slightly slower generations, fewer containers = lower cost

## Troubleshooting

### Model Download Fails

**Problem**: Model won't download from HuggingFace

**Solution**: Verify your HF token has access to the model
```bash
# Test your token
modal secret list
modal secret create huggingface-secret HF_TOKEN=hf_your_new_token
```

### Out of Memory Errors

**Problem**: GPU runs out of VRAM

**Solutions**:
- Use a larger GPU: `gpu=f"H100:{N_GPU}"`
- Reduce concurrency: `@modal.concurrent(max_inputs=2)`
- Use a quantized model: `diffusers/FLUX.2-dev-bnb-4bit`
- Reduce inference steps: `--set-steps "20"`

### Cold Start Times

**Problem**: First request takes too long

**Solutions**:
- Increase `scaledown_window` to keep containers warm longer
- Use `modal app keep-warm` to maintain minimum replicas
- Pre-download models by running a test generation

### FLUX.1-Kontext-dev Errors

**Problem**: Errors when using FLUX.1-Kontext-dev

**Solution**: Ensure you're using exactly `diffusers==0.36.0`:
```python
.uv_pip_install(
    "torch==2.8",
    "diffusers==0.36.0",  # Required for FLUX.1-Kontext-dev
    # ... rest of packages
)
```

## Next Steps

- **Production**: Add proper authentication and rate limiting
- **Monitoring**: Set up Modal's built-in metrics and alerts
- **Scaling**: Configure auto-scaling rules for your traffic patterns
- **Multi-Model**: Deploy multiple models with different endpoints

## Resources

- [Aquiles-Image GitHub](https://github.com/Aquiles-ai/Aquiles-Image)
- [Modal Documentation](https://modal.com/docs)
- [HuggingFace Models](https://huggingface.co/models)
- [FLUX Model Cards](https://huggingface.co/black-forest-labs)