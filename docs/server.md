## Launch Server

Once you've completed the installation, you can start your Aquiles-Image server using the CLI. The server provides various configuration options to customize your deployment.

### Basic Usage

Start a server with default settings:
```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium"
```

This will start the server at `http://0.0.0.0:5500` and load the specified model.

### CLI Options

#### Server Configuration

**`--host`**
- **Default:** `0.0.0.0`
- **Description:** Host address where the server will run
- **Example:** `--host "127.0.0.1"` (localhost only)

**`--port`**
- **Default:** `5500`
- **Type:** Integer
- **Description:** Port where the server will listen
- **Example:** `--port 8000`

#### Model Configuration

**`--model`**
- **Required:** Yes (unless configured previously)
- **Type:** String
- **Description:** The model to use for generation
- **Example:** `--model "black-forest-labs/FLUX.1-dev"`

**`--set-steps`**
- **Default:** Model-specific default
- **Type:** Integer
- **Description:** Number of inference steps for generation
- **Example:** `--set-steps 30`

**`--auto-pipeline`**
- **Default:** `False`
- **Type:** Flag
- **Description:** Enable AutoPipeline for models not natively supported
- **Example:** `--auto-pipeline`

**`--auto-pipeline-type`**
- **Default:** `None`
- **Type:** Choice (`t2i` | `i2i`)
- **Description:** Specifies the AutoPipeline mode. Required when using `--auto-pipeline`. Use `t2i` for Text-to-Image or `i2i` for Image-to-Image.
- **Example:** `--auto-pipeline-type t2i`

**`--device-map`**
- **Default:** `None`
- **Type:** String
- **Description:** Device mapping for model loading (only compatible with `diffusers/FLUX.2-dev-bnb-4bit`)
- **Example:** `--device-map "cuda"`

**`--guidance-scale`**
- **Default:** `None`
- **Type:** Float
- **Description:** Guidance scale value for image generation. Higher values make the output more closely follow the prompt, at the cost of image diversity.
- **Example:** `--guidance-scale 7.5`

**`--seed`**
- **Default:** `None`
- **Type:** Integer
- **Description:** Fixed seed for reproducible image generation. When set, the same prompt will always produce the same output.
- **Example:** `--seed 42`

**`--load-lora / --no-load-lora`**
- **Default:** `None`
- **Type:** Flag
- **Description:** Enable LoRA adapter loading at startup. Must be paired with `--lora-config` pointing to a JSON file describing the LoRA weights.
- **Use case:** Apply fine-tuned styles or concepts to any supported base model.
- **Example:** `--load-lora`

**`--lora-config`**
- **Default:** `None`
- **Type:** String (file path)
- **Description:** Path to a JSON file containing LoRA configuration (repo_id, weight_name, adapter_name, scale). Required when `--load-lora` is set.
- **Example:** `--lora-config "./lora_config.json"`

**`--mode`**
- **Default:** `eager`
- **Type:** Choice (`eager` | `piecewise`)
- **Description:** Compilation strategy for the model. `eager` runs the standard PyTorch pipeline. `piecewise` enables per-shape warmup compilation (HyperKernels) on FLUX models via `torch.compile`, improving throughput after an initial warmup phase.
- **Use case:** Use `piecewise` for FLUX models in production to get optimized inference after the first few requests.
- **Example:** `--mode piecewise`

**`--api-key`**
- **Default:** No authentication
- **Type:** String
- **Description:** Require API key for requests
- **Example:** `--api-key "your-secret-key"`
```python
# Client usage with API key
client = OpenAI(
    base_url="http://127.0.0.1:5500",
    api_key="your-secret-key"
)
```

**`--username`**
- **Default:** `None`
- **Type:** String
- **Description:** Username for the playground. Enables the playground interface when set together with `--password`.
- **Example:** `--username "admin"`

**`--password`**
- **Default:** `None`
- **Type:** String
- **Description:** Password for the playground. Enables the playground interface when set together with `--username`.
- **Example:** `--password "my-secure-password"`

> **Note:** The playground is only enabled when **both** `--username` and `--password` are provided.

#### Performance & Concurrency

**`--max-concurrent-infer`**
- **Default:** 50
- **Type:** Integer
- **Description:** Maximum number of concurrent inference requests
- **Example:** `--max-concurrent-infer 3`

**`--block-request / --no-block-request`**
- **Default:** `False`
- **Type:** Flag
- **Description:** Block new requests when max concurrent inferences is reached instead of queueing them
- **Example:** `--block-request`

**`--dist-inference / --no-dist-inference`**
- **Default:** `None`
- **Type:** Flag
- **Description:** Enable distributed inference mode for load balancing across multiple workers/devices
- **Note:** Batch inference is always active in both single-device and distributed modes
- **Example:** `--dist-inference`

**`--max-batch-size`**
- **Default:** `None`
- **Type:** Integer
- **Description:** Maximum number of requests to group in a single batch for inference (applies to both single-device and distributed modes)
- **Use case:** Optimize GPU utilization by processing multiple requests together
- **Example:** `--max-batch-size 8`

**`--batch-timeout`**
- **Default:** `None`
- **Type:** Float
- **Description:** Maximum time (in seconds) to wait before processing a batch even if not full (applies to both single-device and distributed modes)
- **Use case:** Balance between batch efficiency and response latency
- **Example:** `--batch-timeout 0.5`

**`--worker-sleep`**
- **Default:** `None`
- **Type:** Float
- **Description:** Time (in seconds) the worker sleeps between checking for new batch requests (applies to both single-device and distributed modes)
- **Use case:** Fine-tune CPU usage and batch processing frequency
- **Example:** `--worker-sleep 0.1`

#### GGUF Registry Commands

**`gguf-download`**
- **Description:** Download a GGUF model file from the Aquiles registry to your local HuggingFace cache. Must be run before serving a `gguf:` model for the first time.
- **Usage:** `aquiles-image gguf-download --model-id <id>`
- **Example:** `aquiles-image gguf-download --model-id flux1-dev-q4k`

**`gguf-update`**
- **Description:** Refresh the local Aquiles GGUF registry from HuggingFace. Run this to pick up newly added GGUF models.
- **Usage:** `aquiles-image gguf-update`

> See the [GGUF documentation](gguf.md) for a full list of available model IDs, the registry format, and Modal deployment examples.

#### Development Options

**`--no-load-model`**
- **Type:** Flag
- **Description:** Start server without loading any model (dev mode)
- **Use case:** Faster development, testing endpoints without GPU
- **Example:** `--no-load-model`
```bash
# Dev mode - instant startup, returns test images
aquiles-image serve --no-load-model
```

**`--force`**
- **Type:** Flag
- **Description:** Force overwrite existing configuration
- **Example:** `--force`

### Common Usage Examples

#### Production Server with Security
```bash
aquiles-image serve \
  --host "0.0.0.0" \
  --port 5500 \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --api-key "prod-secret-key-2024" \
  --max-concurrent-infer 5 \
  --block-request
```

#### Production Server with Playground
```bash
aquiles-image serve \
  --host "0.0.0.0" \
  --port 5500 \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --api-key "prod-secret-key-2024" \
  --username "admin" \
  --password "my-secure-password"
```

#### Single-Device Server with Optimized Batching
```bash
aquiles-image serve \
  --host "0.0.0.0" \
  --port 5500 \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --max-batch-size 8 \
  --batch-timeout 0.5 \
  --worker-sleep 0.1 \
  --max-concurrent-infer 10
```

#### Distributed Inference Server
```bash
aquiles-image serve \
  --host "0.0.0.0" \
  --port 5500 \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --dist-inference \
  --max-batch-size 8 \
  --batch-timeout 0.5 \
  --worker-sleep 0.1
```

#### Development Server
```bash
aquiles-image serve \
  --host "127.0.0.1" \
  --port 8000 \
  --model "black-forest-labs/FLUX.1-schnell" \
  --set-steps 4
```

#### Testing Without GPU
```bash
aquiles-image serve \
  --host "127.0.0.1" \
  --port 5500 \
  --no-load-model
```

#### GGUF Quantized Model

```bash
# Download first (only needed once)
aquiles-image gguf-download --model-id flux1-dev-q4k

# Serve
aquiles-image serve \
  --model "gguf:flux1-dev-q4k" \
  --set-steps 30 \
  --api-key "your-key"
```

#### Video Generation Server
```bash
aquiles-image serve \
  --host "0.0.0.0" \
  --port 5500 \
  --model "wan2.2"
```

#### AutoPipeline with Custom Model
```bash
aquiles-image serve \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --set-steps 30 \
  --auto-pipeline \
  --auto-pipeline-type t2i
```

#### AutoPipeline Image-to-Image
```bash
aquiles-image serve \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --auto-pipeline \
  --auto-pipeline-type i2i
```

#### LoRA Adapter Loading
```bash
# Create a LoRA config file first, then serve with it
aquiles-image serve \
  --model "black-forest-labs/FLUX.1-dev" \
  --load-lora \
  --lora-config "./lora_config.json"
```

#### HyperKernels (Piecewise Compilation)
```bash
# Enables per-shape torch.compile warmup for FLUX models
aquiles-image serve \
  --model "black-forest-labs/FLUX.1-dev" \
  --mode piecewise \
  --set-steps 30
```

#### Quantized Model with Device Mapping
```bash
aquiles-image serve \
  --model "diffusers/FLUX.2-dev-bnb-4bit" \
  --device-map "cuda"
```

#### Reproducible Generation with Fixed Seed
```bash
aquiles-image serve \
  --model "stabilityai/stable-diffusion-3.5-medium" \
  --seed 42 \
  --guidance-scale 7.5
```

#### Distributed Inference with Custom Batch Settings
```bash
aquiles-image serve \
  --model "black-forest-labs/FLUX.1-dev" \
  --dist-inference \
  --max-batch-size 16 \
  --batch-timeout 1.0 \
  --worker-sleep 0.05
```

> **Note**: Depending on the model that is loaded, that will be the endpoint that is available.
> 
> **Batch Processing**: Batch inference is always enabled for optimal performance. The `--max-batch-size`, `--batch-timeout`, and `--worker-sleep` options allow you to fine-tune batch processing behavior in both single-device and distributed modes.

### Configuration Persistence

Aquiles-Image saves your configuration automatically. Once you've configured your server, you can start it without repeating all options:
```bash
# First time - full configuration
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium" --api-key "my-key"

# Next time - uses saved configuration
aquiles-image serve
```

To update specific settings without changing others:
```bash
# Change only the model
aquiles-image serve --model "black-forest-labs/FLUX.1-dev"

# Change only the port
aquiles-image serve --port 8000

# Enable distributed inference mode
aquiles-image serve --dist-inference

# Adjust batch processing parameters (works in both single-device and distributed modes)
aquiles-image serve --max-batch-size 12 --batch-timeout 0.8

# Set a fixed seed and guidance scale
aquiles-image serve --seed 42 --guidance-scale 7.5

# Enable playground access
aquiles-image serve --username "admin" --password "my-password"
```

### Verifying Server Status

Once your server is running, you should see output similar to:
```
Starting Aquiles-Image server:
   Host: 0.0.0.0
   Port: 5500
   Model: stabilityai/stable-diffusion-3.5-medium
   Config: 8 settings loaded

Server will be available at: http://0.0.0.0:5500
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5500 (Press CTRL+C to quit)
```

You can verify the server is working by visiting `http://localhost:5500/docs` in your browser to see the interactive API documentation.

### Server Management

**Stopping the Server**

Press `CTRL+C` in the terminal where the server is running:
```
^C
Server stopped by user.
```

**Resetting Configuration**

If you encounter configuration issues, reset to defaults:
```bash
aquiles-image configs --reset
```

### Troubleshooting

**Port Already in Use**
```bash
# Error: Address already in use
# Solution: Use a different port
aquiles-image serve --port 5501
```

**Model Loading Errors**
```bash
# If model fails to load, try dev mode first
aquiles-image serve --no-load-model

# Then test with the actual model
aquiles-image serve --model "your-model-name"
```

**Authentication Issues**
```bash
# If you get 401 errors, make sure API key matches
aquiles-image serve --api-key "correct-key"
```

**Out of Memory**
```bash
# Try a smaller model or quantized version
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium"
aquiles-image serve --model "diffusers/FLUX.2-dev-bnb-4bit" --device-map "cuda"
```

**AutoPipeline Type Not Set**
```bash
# If using --auto-pipeline, always specify the pipeline type
aquiles-image serve --auto-pipeline --auto-pipeline-type t2i
aquiles-image serve --auto-pipeline --auto-pipeline-type i2i
```

**Batch Processing Performance**
```bash
# Batch inference is always active. If experiencing high latency, reduce batch timeout
aquiles-image serve --batch-timeout 0.3

# If GPU utilization is low, increase batch size
aquiles-image serve --max-batch-size 16

# Fine-tune worker sleep for optimal balance
aquiles-image serve --worker-sleep 0.05

# For distributed workloads across multiple devices/workers
aquiles-image serve --dist-inference --max-batch-size 16
```