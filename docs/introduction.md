<div align="center">

# Aquiles-Image

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1763763684/aquiles_image_m6ej7u.png" alt="Aquiles-Image Logo" class="w-full rounded-xl shadow-lg my-6"/>

### **A high-performance, memory-efficient inference server for diffusion models, compatible with the OpenAI client**

*🚀 FastAPI • Diffusers • Compatible with the OpenAI client*

</div>

## What is Aquiles-Image?

Aquiles-Image is a production-ready API server that brings state-of-the-art image generation models to your applications. Built on FastAPI and Diffusers, it provides an **OpenAI-compatible interface** for generating and editing images using models like FLUX, Stable Diffusion 3.5, and more.

### Key Features

- **🔌 OpenAI Compatible** - Use the official OpenAI client with zero code changes
- **⚡ Intelligent Batching** - Automatic request grouping by shared parameters for maximum throughput on single or multi-GPU setups
- **🎨 49 Optimized Models** - 35 image (FLUX, SD3.5, Qwen, Krea-2, Ideogram-4, ERNIE, Nucleus, GLM, Z-Image) + 14 video models (Wan2.x, HunyuanVideo, LTX-2, LTX-2.3) + unlimited via AutoPipeline (Only T2I)
- **🚀 Multi-GPU Support** - Distributed inference with dynamic load balancing across GPUs (image models) for horizontal scaling
- **🛠️ Superior DevX** - Simple CLI, dev mode for testing, built-in monitoring
- **🎬 Advanced Video** - Text-to-video with Wan2.x and HunyuanVideo series (+ Turbo variants)