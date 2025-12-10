## Python Client

Aquiles-Image provides full compatibility with OpenAI's Python client library. This means any OpenAI SDK feature for image and video generation will work seamlessly with Aquiles-Image, as long as you're using the equivalent endpoints shown below.

> 💡 **SDK Compatibility**: All OpenAI SDKs (Python, Node.js, Go, etc.) are compatible with Aquiles-Image for the endpoints demonstrated here.

#### Install the OpenAI Client

```bash
pip install openai
```

#### Image Generation

Generate images from text prompts using any supported model:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://username--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"
)

prompt = (
    "A vast futuristic city curving upward into the sky, its buildings bending "
    "and connecting overhead in a continuous loop. Gravity shifts seamlessly along "
    "the curve, with sunlight streaming across inverted skyscrapers. The scene feels "
    "serene and awe-inspiring—earthlike fields and rivers running along the inner "
    "surface of a colossal rotating structure."
)

result = client.images.generate(
    model="black-forest-labs/FLUX.1-Krea-dev",
    prompt=prompt,
    size="1024x1024",
    response_format="b64_json"
)

print("Downloading image\n")

image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Image downloaded successfully\n")
```

**Available models for generation:**
- All FLUX models (FLUX.1-dev, FLUX.1-schnell, FLUX.1-Krea-dev, FLUX.2-dev)
- All Stable Diffusion 3.5 models
- Quantized models (FLUX.2-dev-bnb-4bit)
- Z-Image-Turbo

#### Image Editing

Edit existing images with text guidance using specialized editing models:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://username--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"
)

print("Editing an image...")

result = client.images.edit(
    model="black-forest-labs/FLUX.1-Kontext-dev",
    image=open("vercel.jpeg", "rb"),
    prompt="Hey, remove the triangle next to the word 'Vercel' and change the word 'Vercel' to 'Aquiles-ai'",
    response_format="b64_json"
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

print("Saving the image to output.png...")
with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Image edited successfully!")
```

**Available models for editing:**
- `black-forest-labs/FLUX.1-Kontext-dev` (recommended)
- `diffusers/FLUX.2-dev-bnb-4bit`

> ⚠️ **Note**: Remember to use `diffusers==0.36.0` for FLUX.1-Kontext-dev to avoid errors.

#### Video Generation (Experimental)

Generate videos from text prompts using the Wan2.2 model:

```python
from openai import OpenAI
import sys
import time

client = OpenAI(
    base_url="https://username--aquiles-image-server-serve.modal.run",
    api_key="dummy-api-key"
)

# Configuration
POLL_INTERVAL = 2.0
TIMEOUT_SECONDS = 60 * 30  # 30 minutes
DOWNLOAD_RETRIES = 5
DOWNLOAD_BACKOFF_BASE = 2

IN_PROGRESS_STATES = {
    "queued", "processing", "in_progress", "running", "starting"
}
SUCCESS_STATES = {
    "succeeded", "completed", "ready", "finished", "success"
}
FAILED_STATES = {"failed", "error"}

def pretty_progress_bar(progress, length=30):
    """Display a progress bar based on percentage."""
    try:
        p = float(progress or 0.0)
    except Exception:
        p = 0.0
    filled = int((p / 100.0) * length)
    return "=" * filled + "-" * (length - filled), p

def poll_until_done(video_id):
    """Poll the video generation status until completion."""
    start = time.time()
    bar_length = 30

    video = client.videos.retrieve(video_id)

    while True:
        status = (getattr(video, "status", "") or "").lower()

        progress = getattr(video, "progress", None)
        bar, p = pretty_progress_bar(progress, bar_length)
        status_text = status.capitalize() if status else "Unknown"
        sys.stdout.write(f"\r{status_text}: [{bar}] {p:.1f}%")
        sys.stdout.flush()

        if status in SUCCESS_STATES:
            sys.stdout.write("\n")
            print("Final status:", status)
            return video
        if status in FAILED_STATES:
            sys.stdout.write("\n")
            msg = getattr(getattr(video, "error", None), "message", "Video generation failed")
            raise RuntimeError(f"Video generation failed: {msg}")

        elapsed = time.time() - start
        if TIMEOUT_SECONDS and elapsed > TIMEOUT_SECONDS:
            sys.stdout.write("\n")
            raise TimeoutError(
                f"Timed out after {TIMEOUT_SECONDS} seconds while waiting for video generation "
                f"(last status: {status})"
            )

        time.sleep(POLL_INTERVAL)
        video = client.videos.retrieve(video_id)

def download_with_retries(video_id, out_path="video.mp4"):
    """Download video with exponential backoff retry logic."""
    attempt = 0
    while attempt < DOWNLOAD_RETRIES:
        attempt += 1
        try:
            print(f"Attempting download (try {attempt}/{DOWNLOAD_RETRIES})...")
            content = client.videos.download_content(video_id, variant="video")
            content.write_to_file(out_path)
            print(f"Video saved to {out_path}")
            return out_path

        except Exception as e:
            err_text = str(e)
            print(f"Download error: {err_text}")
            if attempt >= DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"Failed to download after {DOWNLOAD_RETRIES} attempts: {err_text}"
                )
            backoff = DOWNLOAD_BACKOFF_BASE ** attempt
            backoff = min(backoff, 60)
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)

def main():
    # Create video generation request
    try:
        created = client.videos.create(
            model="wan2.2",
            prompt="A video of a cool cat on a motorcycle in the night",
        )
    except Exception as e:
        print("Error creating video:", e)
        sys.exit(1)

    video_id = getattr(created, "id", None)
    if not video_id:
        print("No video id returned from create call.")
        sys.exit(1)

    print("Video generation started:", video_id)

    # Poll until video is ready
    try:
        finished_video = poll_until_done(video_id)
    except TimeoutError as te:
        print("Timeout:", te)
        sys.exit(1)
    except RuntimeError as re:
        print(re)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("Unexpected error while waiting for generation:", e)
        sys.exit(1)

    # Download the generated video
    try:
        download_with_retries(video_id, out_path="video.mp4")
    except Exception as e:
        print("Error downloading or writing video content:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Video generation notes:**
- Video generation can take several minutes to complete
- The script polls the server for progress updates
- Includes automatic retry logic for downloads
- Model: `wan2.2` (Wan-AI/Wan2.2-T2V-A14B)

### Using Other SDKs

Since Aquiles-Image is fully OpenAI-compatible, you can use any OpenAI SDK in any language:

**Node.js/TypeScript:**
```bash
npm install openai
```

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://username--aquiles-image-server-serve.modal.run',
  apiKey: 'dummy-api-key',
});

const result = await client.images.generate({
  model: 'black-forest-labs/FLUX.1-Krea-dev',
  prompt: 'A serene Japanese garden with cherry blossoms',
  size: '1024x1024',
});
```

**Go:**
```bash
go get github.com/openai/openai-go
```

```go
client := openai.NewClient(
    option.WithBaseURL("https://username--aquiles-image-server-serve.modal.run"),
    option.WithAPIKey("dummy-api-key"),
)

image, err := client.Images.Generate(ctx, openai.ImageGenerateParams{
    Model:  openai.F("black-forest-labs/FLUX.1-Krea-dev"),
    Prompt: openai.F("A serene Japanese garden with cherry blossoms"),
    Size:   openai.F(openai.ImageGenerateParamsSize1024x1024),
})
```

For other languages, check the [OpenAI SDK documentation](https://platform.openai.com/docs/libraries) and simply point the `base_url` to your Aquiles-Image deployment.


### API Endpoints Reference

| Endpoint | Method | Purpose | Compatible Models |
|----------|--------|---------|-------------------|
| `/images/generations` | POST | Generate images from text | All FLUX, SD3.5, Z-Image-Turbo |
| `/images/edits` | POST | Edit existing images | FLUX.1-Kontext-dev, FLUX.2-dev-bnb-4bit |
| `/videos` | POST | Generate videos from text | wan2.2 |
| `/videos/{video_id}` | GET | Check video generation status | wan2.2 |
| `/videos/{video_id}/content` | GET | Download generated video | wan2.2 |

### Interactive API Documentation

Visit your deployment's `/docs` endpoint for interactive Swagger UI documentation:
```
https://username--aquiles-image-server-serve.modal.run/docs
```

This provides:
- Full API schema
- Try-it-out functionality
- Request/response examples
- Parameter descriptions