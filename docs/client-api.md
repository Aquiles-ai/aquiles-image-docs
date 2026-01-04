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

#### Video Generation

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


## 📊 Monitoring & Stats

Aquiles-Image provides a custom `/stats` endpoint for real-time monitoring of your server's performance and resource utilization.

### Python Example

```python
import requests

# Basic stats request
def get_server_stats(base_url, api_key=None):
    """Retrieve server statistics from Aquiles-Image."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.get(f"{base_url}/stats", headers=headers)
    response.raise_for_status()
    return response.json()

# Usage
base_url = "http://localhost:5500"
api_key = "YOUR_API_KEY"  # Optional, only if server requires authentication

try:
    stats = get_server_stats(base_url, api_key)
    
    # Display basic metrics
    print(f"Total requests: {stats.get('total_requests', stats.get('total_tasks', 0))}")
    print(f"Queued: {stats['queued']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Server available: {stats['available']}")
    
    # For image models, show additional metrics
    if 'total_images' in stats:
        print(f"Total images generated: {stats['total_images']}")
        print(f"Total batches: {stats['total_batches']}")
    
    # For distributed mode, show device-specific stats
    if stats.get('mode') == 'distributed':
        print("\nDevice Statistics:")
        for device_id, device_stats in stats['devices'].items():
            print(f"\n  {device_id}:")
            print(f"    Available: {device_stats['available']}")
            print(f"    Processing: {device_stats['processing']}")
            print(f"    Images completed: {device_stats['images_completed']}")
            print(f"    Avg batch time: {device_stats['avg_batch_time']:.2f}s")
            print(f"    Estimated load: {device_stats['estimated_load']:.2%}")
            
except requests.exceptions.RequestException as e:
    print(f"X Error fetching stats: {e}")
```

### JavaScript/TypeScript Example

```typescript
// TypeScript/JavaScript example
interface ServerStats {
  mode?: 'single-device' | 'distributed';
  total_requests?: number;
  total_tasks?: number;
  total_batches?: number;
  total_images?: number;
  queued: number;
  completed: number;
  failed: number;
  processing?: boolean;
  available: boolean;
  devices?: Record<string, DeviceStats>;
  global?: GlobalStats;
}

interface DeviceStats {
  id: string;
  available: boolean;
  processing: boolean;
  can_accept_batch: boolean;
  batch_size: number;
  max_batch_size: number;
  images_processing: number;
  images_completed: number;
  total_batches_processed: number;
  avg_batch_time: number;
  estimated_load: number;
  error_count: number;
  last_error: string | null;
}

interface GlobalStats {
  total_requests: number;
  total_batches: number;
  total_images: number;
  queued: number;
  active_batches: number;
  completed: number;
  failed: number;
  processing: boolean;
}

async function getServerStats(
  baseUrl: string, 
  apiKey?: string
): Promise<ServerStats> {
  const headers: Record<string, string> = {};
  
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }
  
  const response = await fetch(`${baseUrl}/stats`, { headers });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
const baseUrl = 'http://localhost:5500';
const apiKey = 'YOUR_API_KEY'; // Optional

try {
  const stats = await getServerStats(baseUrl, apiKey);
  
  console.log(`Total requests: ${stats.total_requests || stats.total_tasks || 0}`);
  console.log(`Queued: ${stats.queued}`);
  console.log(`Completed: ${stats.completed}`);
  console.log(`Failed: ${stats.failed}`);
  console.log(`Server available: ${stats.available}`);
  
  // For image models
  if (stats.total_images !== undefined) {
    console.log(`Total images generated: ${stats.total_images}`);
    console.log(`Total batches: ${stats.total_batches}`);
  }
  
  // For distributed mode
  if (stats.mode === 'distributed' && stats.devices) {
    console.log('\nDevice Statistics:');
    
    for (const [deviceId, deviceStats] of Object.entries(stats.devices)) {
      console.log(`\n  ${deviceId}:`);
      console.log(`    Available: ${deviceStats.available}`);
      console.log(`    Processing: ${deviceStats.processing}`);
      console.log(`    Images completed: ${deviceStats.images_completed}`);
      console.log(`    Avg batch time: ${deviceStats.avg_batch_time.toFixed(2)}s`);
      console.log(`    Estimated load: ${(deviceStats.estimated_load * 100).toFixed(1)}%`);
    }
  }
  
} catch (error) {
  console.error('X Error fetching stats:', error);
}
```

### Response Formats

The response varies depending on the model type and configuration:

#### Image Models - Single-Device Mode

```json
{
  "mode": "single-device",
  "total_requests": 150,
  "total_batches": 42,
  "total_images": 180,
  "queued": 3,
  "completed": 147,
  "failed": 0,
  "processing": true,
  "available": false
}
```

#### Image Models - Distributed Mode (Multi-GPU)

```json
{
  "mode": "distributed",
  "devices": {
    "cuda:0": {
      "id": "cuda:0",
      "available": true,
      "processing": false,
      "can_accept_batch": true,
      "batch_size": 4,
      "max_batch_size": 8,
      "images_processing": 0,
      "images_completed": 45,
      "total_batches_processed": 12,
      "avg_batch_time": 2.5,
      "estimated_load": 0.3,
      "error_count": 0,
      "last_error": null
    },
    "cuda:1": {
      "id": "cuda:1",
      "available": true,
      "processing": true,
      "can_accept_batch": false,
      "batch_size": 2,
      "max_batch_size": 8,
      "images_processing": 2,
      "images_completed": 38,
      "total_batches_processed": 10,
      "avg_batch_time": 2.8,
      "estimated_load": 0.7,
      "error_count": 0,
      "last_error": null
    }
  },
  "global": {
    "total_requests": 150,
    "total_batches": 42,
    "total_images": 180,
    "queued": 3,
    "active_batches": 1,
    "completed": 147,
    "failed": 0,
    "processing": true
  }
}
```

#### Video Models

```json
{
  "total_tasks": 25,
  "queued": 2,
  "processing": 1,
  "completed": 20,
  "failed": 2,
  "available": false,
  "max_concurrent": 1
}
```

### Key Metrics Explained

- **`total_requests/tasks`** - Total number of generation requests received
- **`total_images`** - Total images generated (image models only)
- **`total_batches`** - Total batches processed (image models only)
- **`queued`** - Requests waiting to be processed
- **`processing`** - Currently processing requests
- **`completed`** - Successfully completed requests
- **`failed`** - Failed requests
- **`available`** - Whether server can accept new requests
- **`mode`** - Operation mode for image models: `single-device` or `distributed`

#### Distributed Mode Specific Metrics

- **`can_accept_batch`** - Whether device can accept new batch
- **`batch_size`** - Current number of images in batch
- **`max_batch_size`** - Maximum configured batch size
- **`images_processing`** - Images currently being processed
- **`images_completed`** - Total images completed on this device
- **`avg_batch_time`** - Average time to process a batch (seconds)
- **`estimated_load`** - Estimated load on device (0.0 to 1.0)
- **`error_count`** - Number of errors encountered
- **`last_error`** - Last error message if any

### Monitoring Dashboard Example

Here's a complete example of a monitoring function that periodically checks server stats:

```python
import requests
import time
from datetime import datetime

def monitor_server(base_url, api_key=None, interval=5):
    """
    Monitor server statistics in real-time.
    
    Args:
        base_url: Base URL of Aquiles-Image server
        api_key: Optional API key for authentication
        interval: Polling interval in seconds
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    print("Starting server monitoring... (Press Ctrl+C to stop)\n")
    
    try:
        while True:
            try:
                response = requests.get(f"{base_url}/stats", headers=headers)
                response.raise_for_status()
                stats = response.json()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}]")
                print("-" * 50)
                
                if stats.get('mode') == 'distributed':
                    # Distributed mode monitoring
                    global_stats = stats.get('global', {})
                    print(f"Mode: Distributed")
                    print(f"Total: {global_stats.get('total_requests', 0)} | "
                          f"Queued: {global_stats.get('queued', 0)} | "
                          f"Completed: {global_stats.get('completed', 0)} | "
                          f"Failed: {global_stats.get('failed', 0)}")
                    
                    print("\nDevices:")
                    for device_id, device in stats.get('devices', {}).items():
                        status = "🟢" if device['available'] else "🔴"
                        processing_indicator = "⚙️" if device['processing'] else "✓"
                        print(f"  {status} {device_id} {processing_indicator}")
                        print(f"    Load: {device['estimated_load']:.1%} | "
                              f"Completed: {device['images_completed']} | "
                              f"Avg Time: {device['avg_batch_time']:.2f}s")
                else:
                    # Single-device or video model monitoring
                    total = stats.get('total_requests', stats.get('total_tasks', 0))
                    print(f"Total: {total} | "
                          f"Queued: {stats['queued']} | "
                          f"Completed: {stats['completed']} | "
                          f"Failed: {stats['failed']}")
                    
                    if 'total_images' in stats:
                        print(f"Images: {stats['total_images']} | "
                              f"Batches: {stats['total_batches']}")
                    
                    status = "🟢 Available" if stats['available'] else "🔴 Busy"
                    print(f"Status: {status}")
                
            except requests.exceptions.RequestException as e:
                print(f"X Error fetching stats: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

# Usage
if __name__ == "__main__":
    monitor_server(
        base_url="http://localhost:5500",
        api_key="YOUR_API_KEY",  # Optional
        interval=5
    )
```

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
| `/images/generations` | POST | Generate images from text | All FLUX, SD3.5, Z-Image-Turbo, etc. |
| `/images/edits` | POST | Edit existing images | FLUX.1-Kontext-dev, FLUX.2-dev-bnb-4bit, etc. |
| `/videos` | POST | Generate videos from text | wan2.2, wan2.1, hunyuanvideo, etc. |
| `/videos/{video_id}` | GET | Check video generation status | wan2.2, wan2.1, hunyuanvideo, etc. |
| `/videos/{video_id}/content` | GET | Download generated video | wan2.2, wan2.1, hunyuanvideo, etc. |
| `/stats` | GET | Get server statistics and monitoring data | All models |

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