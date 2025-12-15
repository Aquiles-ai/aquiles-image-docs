## Deploy Aquiles-Image on Modal

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

## Video Generation with wan2.2 and wan2.2-turbo

> ⚠️ **High Inference Times & Costs**: Video generation has significantly longer inference times compared to image generation. The **wan2.2** model takes ~30 minutes per video (40 steps), while **wan2.2-turbo** takes ~3 minutes (4 steps) with equivalent quality. Both models require **NVIDIA H100/A100-80GB GPUs minimum** and will incur substantial compute costs. Monitor your usage carefully and consider longer scale-down windows to avoid frequent cold starts.

### Overview

The wan2.2 family of models enables text-to-video generation through Aquiles-Image. Unlike image generation which completes in seconds, video generation is a compute-intensive process with two model options:

| Model | Inference Time | Steps | Quality | Best For |
|-------|----------------|-------|---------|----------|
| **wan2.2** | ~30 min/video | 40 | High | Maximum quality, less time-sensitive workflows |
| **wan2.2-turbo** ⚡ | ~3 min/video | 4 | High (equivalent) | **Production use, faster iteration, cost optimization** |

### Requirements

- **Minimum GPU**: NVIDIA H100 or A100-80GB (80GB VRAM)
- **Typical inference time**: 
  - wan2.2: 28-30 minutes per video
  - wan2.2-turbo: 3-5 minutes per video
- **Recommended concurrency**: 1 request at a time
- **Scale-down window**: 
  - wan2.2: 30+ minutes to minimize cold starts
  - wan2.2-turbo: 10-15 minutes (faster turnaround)

### Recommendation

**For most use cases, `wan2.2-turbo` is the recommended choice:**
- ✅ 9.5x faster inference (3 min vs 30 min)
- ✅ Same output quality in production testing
- ✅ 10x lower compute costs per video
- ✅ Better user experience (no timeouts)
- ✅ Higher throughput (~18 videos/hour vs ~2 videos/hour)

Use `wan2.2` only if you have specific quality requirements that cannot be met by the turbo variant.

### Deployment Configuration

Save this as `aquiles_modal_wan22.py`:

```python
import modal
import os

aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "wget", "libgl1", "libglib2.0-0")
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel",
    )
    .uv_pip_install(
        "torch==2.8",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==4.57.3",
        "tokenizers==0.22.1",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git",
        "git+https://github.com/ModelTC/LightX2V.git"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_TOKEN": os.getenv("Hugging_face_token_for_deploy", "")
    })  
)

MODEL_NAME = "wan2.2" # or wan2.2-turbo

# Persistent volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)
aquiles_video_vol = modal.Volume.from_name("aquiles-video-cache", create_if_missing=True)

app = modal.App("aquiles-image-server-wan2-2")

N_GPU = 1
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"H100:{N_GPU}",
    scaledown_window=30 * MINUTES,  # Keep warm longer due to high cold start cost
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
        "/root/.local/share": aquiles_video_vol,
    },
)
@modal.concurrent(max_inputs=1)  # Process one video at a time
@modal.web_server(port=AQUILES_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "aquiles-image",
        "serve",
        "--host", "0.0.0.0",
        "--port", str(AQUILES_PORT),
        "--model", MODEL_NAME,
        "--api-key", "dummy-api-key",
    ]

    print(f"Starting Aquiles-Image with model: {MODEL_NAME}")
    print(f"Command: {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)
```

### Key Configuration Differences

**Compared to image generation deployments**, wan2.2 requires:

| Setting | Image Generation | wan2.2 Video | Reason |
|---------|-----------------|--------------|---------|
| **GPU** | A100 or H100 | H100 minimum | Higher VRAM and compute requirements |
| **Concurrency** | 3-5 requests | 1 request | Video generation is memory-intensive |
| **Scale-down** | 6 minutes | 30+ minutes | Avoid expensive cold starts |
| **Timeout** | 10 minutes | 15 minutes | Longer generation times |
| **Additional packages** | Standard | flash-attention, LightX2V | Video model dependencies |

### Deploy to Modal

```bash
# Set your HuggingFace token (if not using Modal secrets)
export Hugging_face_token_for_deploy=hf_your_token_here

# Deploy the service
modal deploy aquiles_modal_wan22.py
```

You'll receive a URL like:
```
✓ Created web function serve => https://username--aquiles-image-server-wan2-2-serve.modal.run
```

### Using the Video API

The wan2.2 model uses an asynchronous workflow:
1. **Create** a video generation job
2. **Poll** for completion status with progress updates
3. **Download** the final video when ready

#### Complete Example with Progress Tracking

Save this as `generate_video.py`:

```python
from openai import OpenAI
import sys
import time

# Initialize client with your Modal deployment URL
openai = OpenAI(
    base_url="https://username--aquiles-image-server-wan2-2-serve.modal.run",
    api_key="dummy-api-key"
)

# Polling configuration
POLL_INTERVAL = 2.0  # Check status every 2 seconds
TIMEOUT_SECONDS = 120 * 30  # 30 minutes max wait
DOWNLOAD_RETRIES = 5
DOWNLOAD_BACKOFF_BASE = 2

# Status categories
IN_PROGRESS_STATES = {
    "queued", "processing", "in_progress", "running", "starting"
}
SUCCESS_STATES = {
    "succeeded", "completed", "ready", "finished", "success"
}
FAILED_STATES = {"failed", "error"}

PROMPT = """
A direct continuation of the existing shot of a chameleon crawling slowly along a mossy branch. 
Begin with the chameleon already mid-step, camera tracking right at the same close, eye-level angle. 
After three seconds, its eyes swivel independently, one pausing to glance toward the lens before it 
resumes moving forward. Maintain the 100 mm anamorphic lens with shallow depth of field, dappled 
rainforest light, faint humidity haze, and subtle film grain. The moss texture and background greenery should 
remain consistent, with the chameleon's deliberate gait flowing naturally as if no cut occurred.
"""

def pretty_progress_bar(progress, length=30):
    """Convert progress percentage to a visual progress bar"""
    try:
        p = float(progress or 0.0)
    except Exception:
        p = 0.0
    filled = int((p / 100.0) * length)
    return "=" * filled + "-" * (length - filled), p

def poll_until_done(video_id):
    """Poll the API until video generation completes or fails"""
    start = time.time()
    bar_length = 30

    video = openai.videos.retrieve(video_id)

    while True:
        status = (getattr(video, "status", "") or "").lower()
        progress = getattr(video, "progress", None)
        bar, p = pretty_progress_bar(progress, bar_length)
        status_text = status.capitalize() if status else "Unknown"
        
        # Display progress
        sys.stdout.write(f"\r{status_text}: [{bar}] {p:.1f}%")
        sys.stdout.flush()

        # Check for completion
        if status in SUCCESS_STATES:
            sys.stdout.write("\n")
            print("Video generation completed!")
            return video
            
        if status in FAILED_STATES:
            sys.stdout.write("\n")
            msg = getattr(getattr(video, "error", None), "message", "Video generation failed")
            raise RuntimeError(f"Video generation failed: {msg}")

        # Check for timeout
        elapsed = time.time() - start
        if TIMEOUT_SECONDS and elapsed > TIMEOUT_SECONDS:
            sys.stdout.write("\n")
            raise TimeoutError(
                f"Timed out after {TIMEOUT_SECONDS} seconds "
                f"(last status: {status})"
            )

        time.sleep(POLL_INTERVAL)
        video = openai.videos.retrieve(video_id)

def download_with_retries(video_id, out_path="video.mp4"):
    """Download video with exponential backoff retry logic"""
    attempt = 0
    while attempt < DOWNLOAD_RETRIES:
        attempt += 1
        try:
            print(f"Downloading video (attempt {attempt}/{DOWNLOAD_RETRIES})...")
            content = openai.videos.download_content(video_id, variant="video")
            content.write_to_file(out_path)
            print(f"✓ Video saved to {out_path}")
            return out_path

        except Exception as e:
            err_text = str(e)
            print(f"Download error: {err_text}")
            
            if attempt >= DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"Failed to download after {DOWNLOAD_RETRIES} attempts: {err_text}"
                )
            
            # Exponential backoff
            backoff = DOWNLOAD_BACKOFF_BASE ** attempt
            backoff = min(backoff, 60)  # Cap at 60 seconds
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)

def main():
    print("Creating video generation job...")
    
    try:
        # Start video generation
        created = openai.videos.create(
            model="wan2.2",
            prompt=PROMPT,
        )
    except Exception as e:
        print(f"Error creating video: {e}")
        sys.exit(1)

    video_id = getattr(created, "id", None)
    if not video_id:
        print("No video ID returned from create call")
        sys.exit(1)

    print(f"Video generation started: {video_id}\n")

    try:
        # Wait for completion
        finished_video = poll_until_done(video_id)
        
        # Download the result
        download_with_retries(video_id, out_path="video.mp4")
        
    except TimeoutError as te:
        print(f"Timeout: {te}")
        sys.exit(1)
    except RuntimeError as re:
        print(re)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python generate_video.py
```

You'll see real-time progress:
```
Video generation started: vid_abc123

Processing: [==============----------------] 45.2%
```

### Example Output

Here's a sample video generated with the prompt above:

#### Video Result: **Chameleon on Branch**


<iframe
  src="https://player.cloudinary.com/embed/?cloud_name=dmtomxyvm&public_id=video_modal_deploy_aftsi6&profile=cld-default"
  width="640"
  height="360" 
  style="height: auto; width: 100%; aspect-ratio: 640 / 360;"
  allow="autoplay; fullscreen; encrypted-media; picture-in-picture"
  allowfullscreen
  frameborder="0"
></iframe>

The wan2.2 model produces high-quality videos with:
- Smooth motion and natural animation
- Consistent lighting and camera work
- Detailed textures and depth of field

### API Endpoints

When deploying wan2.2, these video-specific endpoints are available:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/videos` | POST | Create a new video generation job |
| `/videos/{video_id}` | GET | Check status and get progress updates |
| `/videos/{video_id}/content` | GET | Download the completed video |

### Cost Management Tips

Video generation is significantly more expensive than image generation. Here's how to optimize costs:

#### 1. **Use Longer Scale-Down Windows**

```python
scaledown_window=30 * MINUTES  # Keep containers warm longer
```

**Why?** Cold starts for video models are expensive (2-3 minutes to load). If you're generating multiple videos, keeping the container warm saves money.

#### 2. **Process Videos in Batches**

If you have multiple videos to generate, queue them and process sequentially rather than starting/stopping the container:

```python
video_ids = []
for prompt in prompts:
    created = openai.videos.create(model="wan2.2", prompt=prompt)
    video_ids.append(created.id)

# Now wait for all to complete
for video_id in video_ids:
    poll_until_done(video_id)
    download_with_retries(video_id, f"video_{video_id}.mp4")
```

#### 3. **Monitor GPU Hours**

Check your Modal dashboard regularly to track H100 usage. A single video generation typically uses:
- **Generation time**: 5-25 minutes
- **Cold start overhead**: 4-6 minutes (first generation only)

#### 4. **Set Realistic Timeouts**

```python
timeout=15 * MINUTES  # Prevents runaway costs from stuck jobs
```

### Troubleshooting

#### Timeout Errors

**Problem**: Video generation times out

**Solutions**:
- Increase timeout: `timeout=20 * MINUTES`
- Simplify your prompt (fewer details = faster generation)
- Check Modal logs: `modal app logs aquiles-image-server-wan2-2`

#### Out of Memory

**Problem**: Container crashes with OOM error

**Solutions**:
- Verify you're using H100 (not A100): `gpu=f"H100:{N_GPU}"`
- Ensure `max_inputs=1` (don't process multiple videos simultaneously)
- Check that all required packages are installed, especially flash-attention

#### Download Failures

**Problem**: Video generates successfully but download fails

**Solutions**:
- The example code includes automatic retries with exponential backoff
- Check network connectivity and Modal service status
- Verify the video ID is correct: `openai.videos.retrieve(video_id)`

#### Long Cold Starts

**Problem**: First video takes 6+ minutes before generation even starts

**Solutions**:
- This is expected behavior - model loading takes 2-3 minutes
- Increase `scaledown_window` to keep the container warm between requests
- Consider using `modal app keep-warm` for production workloads

### Production Considerations

For production deployments of wan2.2:

1. **Authentication**: Replace `dummy-api-key` with proper authentication
2. **Rate Limiting**: Implement request queuing to prevent overload
3. **Monitoring**: Set up alerts for long-running jobs and failures
4. **Cost Alerts**: Configure Modal billing alerts for H100 usage
5. **Queue System**: Consider adding a job queue (Redis, PostgreSQL) for multiple users
6. **Storage**: Videos are large - plan for persistent storage of outputs

## Next Steps

- **Production**: Add proper authentication and rate limiting
- **Monitoring**: Set up Modal's built-in metrics and alerts
- **Scaling**: Configure auto-scaling rules for your traffic patterns
- **Multi-Model**: Deploy multiple models with different endpoints
- **Experiment with prompts**: Video prompts benefit from cinematic details (camera angles, lighting, movement)
- **Implement webhooks**: For long-running jobs, consider webhook callbacks instead of polling
- **Add preprocessing**: Validate prompts before expensive generation
- **Monitor costs**: Set up budget alerts in Modal dashboard

## Resources

- [Aquiles-Image GitHub](https://github.com/Aquiles-ai/Aquiles-Image)
- [Modal Documentation](https://modal.com/docs)
- [HuggingFace Models](https://huggingface.co/models)
- [FLUX Model Cards](https://huggingface.co/black-forest-labs)