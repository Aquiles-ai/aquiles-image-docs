## GGUF Model Support

Aquiles-Image supports running **GGUF-quantized diffusion models** natively through a built-in registry system. GGUF quantization dramatically reduces VRAM requirements while preserving generation quality, making it possible to run models like FLUX.1-dev on GPUs with less memory than the full-precision version requires.

> **Note:** GGUF support requires `diffusers` installed from source (GitHub), as GGUF quantization in diffusers is not yet stable in a PyPI release. See the installation section below.


### How GGUF Support Works

When you pass a model ID prefixed with `gguf:` to Aquiles-Image, the server:

1. Looks up the model in the local **Aquiles GGUF Registry** (a JSON file stored in your user data directory)
2. Downloads the `.gguf` file from HuggingFace Hub (cached locally after first download)
3. Loads the GGUF-quantized transformer using `GGUFQuantizationConfig` from diffusers
4. Assembles the full pipeline from the base (non-quantized) repo on HuggingFace
5. Starts serving requests at the standard `/images/generations` endpoint

The registry entry defines which GGUF file to use, which HuggingFace repo contains the base model, and which transformer and pipeline classes to instantiate. This makes it easy to add new GGUF models without changing the server code.


### Installation

GGUF support requires installing `diffusers` from source and the `gguf` package:

```bash
uv pip install git+https://github.com/huggingface/diffusers.git
uv pip install gguf
```

The `gguf` package is also included automatically when you install `aquiles-image` from PyPI or source, but `diffusers` from source must be installed explicitly.


### Quick Start

#### 1. Download a GGUF Model from the Registry

```bash
aquiles-image gguf-download --model-id flux1-dev-q4k
```

This command downloads the `.gguf` file from HuggingFace to your local cache (typically `~/.cache/huggingface/hub`). The registry is fetched automatically on first use.

#### 2. Start the Server

```bash
aquiles-image serve --model "gguf:flux1-dev-q4k" --set-steps 30 --api-key "your-key"
```

Note the `gguf:` prefix — this is how Aquiles-Image distinguishes GGUF models from standard HuggingFace model IDs.

#### 3. Generate Images

GGUF models use the standard `/images/generations` endpoint with the OpenAI client:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:5500",
    api_key="your-key"
)

result = client.images.generate(
    model="gguf:flux1-dev-q4k",
    prompt="A cat sitting on a futuristic city rooftop at sunset",
    size="1024x1024",
    response_format="b64_json"
)

image_bytes = base64.b64decode(result.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```


### Available GGUF Models

The Aquiles GGUF Registry currently includes the following models:

| Model ID | GGUF File | Base Model | Quantization |
|----------|-----------|------------|--------------|
| `flux1-dev-q4k` | `flux1-dev-Q4_0.gguf` | `black-forest-labs/FLUX.1-dev` | Q4_0 |
| `flux1-dev-q4-1` | `flux1-dev-Q4_1.gguf` | `black-forest-labs/FLUX.1-dev` | Q4_1 |

Use the model ID (first column) as the argument to `gguf-download` and as the `gguf:` suffix when serving. The registry is hosted at [`Aquiles-ai/aquiles-gguf-registry`](https://huggingface.co/datasets/Aquiles-ai/aquiles-gguf-registry) on HuggingFace and is updated as new quantized models become available.

> **Access:** FLUX.1-dev is a gated model. You must have a HuggingFace account with access approved and run `hf auth login` before the server can download the base model weights.


### CLI Commands

#### `gguf-download`

Downloads a GGUF model file from the registry to your local HuggingFace cache:

```bash
aquiles-image gguf-download --model-id <model-id>
```

**Options:**

| Option | Description |
|--------|-------------|
| `--model-id` | Model ID from the Aquiles GGUF registry (required) |

**Example:**

```bash
# Download the Q4_0 quantized FLUX.1-dev transformer
aquiles-image gguf-download --model-id flux1-dev-q4k

# Download the Q4_1 variant
aquiles-image gguf-download --model-id flux1-dev-q4-1
```

If the model ID is not found in the registry, the command will list the available IDs.

#### `gguf-update`

Refreshes the local GGUF registry from HuggingFace, picking up newly added models:

```bash
aquiles-image gguf-update
```

Run this periodically to get access to newly added GGUF models without reinstalling Aquiles-Image.


### Supported Capabilities

GGUF models loaded through this system support:

- **Text-to-image generation** via `/images/generations`
- **LoRA loading** (pass `--load-lora` and `--lora-config` at serve time)
- **CUDA and MPS** devices (Apple Silicon supported)
- **Low-VRAM mode** via `enable_model_cpu_offload` when `low_vram` is active

**Not supported with GGUF models:**
- Image editing (`/images/edits`) — the assembled pipeline is text-to-image only
- Distributed inference (`--dist-inference`)
- `enable_sequential_cpu_offload` — incompatible with GGUF quantization; `enable_model_cpu_offload` is used instead

### Registry File Structure

The registry is a JSON file stored at `<user_data_dir>/aquiles/Aquiles-Image/registry.json`. Each entry has the following fields:

```json
{
    "flux1-dev-q4k": {
        "gguf_repo": "city96/FLUX.1-dev-gguf",
        "gguf_file": "flux1-dev-Q4_0.gguf",
        "base_repo": "black-forest-labs/FLUX.1-dev",
        "transformer_cls": "diffusers.FluxTransformer2DModel",
        "pipeline_cls": "diffusers.FluxPipeline",
        "added_by": "FredyRivera-dev",
        "date_added": "2026-06-20"
    }
}
```

| Field | Description |
|-------|-------------|
| `gguf_repo` | HuggingFace repo containing the `.gguf` file |
| `gguf_file` | Filename of the GGUF transformer weights |
| `base_repo` | HuggingFace repo for the full base model (VAE, text encoders, scheduler) |
| `transformer_cls` | Fully-qualified Python class for the transformer (e.g. `diffusers.FluxTransformer2DModel`) |
| `pipeline_cls` | Fully-qualified Python class for the pipeline (e.g. `diffusers.FluxPipeline`) |
| `added_by` | Author who submitted the entry |
| `date_added` | ISO date when the entry was added |


### Deploy on Modal (GGUF)

GGUF models run well on smaller GPUs. Here is a complete Modal deployment example:

```python
import modal
import os

aquiles_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential")
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .uv_pip_install(
        "torch==2.9",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==5.12.1",
        "git+https://github.com/Aquiles-ai/Aquiles-Image.git@feature/Add-GGUF-Support",
        "bitsandbytes",
        "gguf"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "gguf:flux1-dev-q4k"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
aquiles_config_vol = modal.Volume.from_name("aquiles-cache", create_if_missing=True)

app = modal.App("aquiles-image-server")

N_GPU = 1
MINUTES = 60
AQUILES_PORT = 5500

@app.function(
    image=aquiles_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"H100:{N_GPU}",
    scaledown_window=6 * MINUTES,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": aquiles_config_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=AQUILES_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "aquiles-image", "serve",
        "--host", "0.0.0.0",
        "--port", str(AQUILES_PORT),
        "--model", MODEL_NAME,
        "--set-steps", "30",
        "--api-key", "dummy-api-key",
        "--username", "root",
        "--password", "root",
    ]

    print(f"Starting Aquiles-Image with GGUF model: {MODEL_NAME}")
    subprocess.Popen(" ".join(cmd), shell=True)
```

Deploy with:

```bash
modal deploy aquiles_modal_gguf.py
```

> **Note:** The GGUF model files are downloaded at container startup. The HuggingFace volume (`huggingface-cache`) ensures they are cached across restarts, so only the first cold start pays the full download cost.


### Troubleshooting

**`Model 'flux1-dev-q4k' not found in registry`**

The registry has not been downloaded yet, or it is outdated. Run:

```bash
aquiles-image gguf-update
```

**`Incomplete registry entry, missing fields`**

The registry JSON entry is malformed. Run `gguf-update` to re-fetch the registry from HuggingFace.

**`No CUDA or MPS device available`**

GGUF pipelines require either a CUDA GPU or Apple Silicon (MPS). CPU-only inference is not supported.

**`GGUFQuantizationConfig` import error**

`diffusers` from PyPI does not yet expose `GGUFQuantizationConfig`. Install from source:

```bash
uv pip install git+https://github.com/huggingface/diffusers.git
```

**Authentication error downloading the base model**

FLUX.1-dev is gated. Log in first:

```bash
hf auth login
```

### Requesting New GGUF Models

To request a new model be added to the Aquiles GGUF registry, open an issue on [Aquiles-Image GitHub](https://github.com/Aquiles-ai/Aquiles-Image) with:

- The HuggingFace repo and file path for the `.gguf` weights
- The base model repo (for VAE and text encoders)
- The transformer and pipeline classes from diffusers