## Dev Mode

Dev Mode allows you to start the Aquiles-Image server without loading any AI models. This is ideal for development, testing, and integration work when you don't need actual model inference.

### What is Dev Mode?

When you start the server with `--no-load-model`, Aquiles-Image runs in Dev Mode. Instead of loading models and performing real inference, the server returns mock responses that simulate the actual API behavior.

### Enabling Dev Mode

```bash
aquiles-image serve --no-load-model
```

Or with additional configuration:

```bash
aquiles-image serve --host "127.0.0.1" --port 5500 --no-load-model
```

### How Dev Mode Works

In Dev Mode, all endpoints return realistic mock responses that match the exact format of real API responses:

#### Image Generation (`/images/generations`)

Returns mock images with proper response structure:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

# Dev mode will return a mock image
result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024",
    n=2
)

# Response structure is identical to production
print(result.data[0].url)  # Valid image URL
print(result.data[0].b64_json)  # Valid base64 image (if requested)
```

#### Image Editing (`/images/edits`)

Simulates image editing operations:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

# Dev mode accepts the request and returns a mock edited image
result = client.images.edit(
    model="black-forest-labs/FLUX.1-Kontext-dev",
    image=open("input.png", "rb"),
    prompt="add sunglasses to the person",
    size="1024x1024"
)

print(result.data[0].url)  # Mock edited image URL
```

#### Video Generation (`/videos`)

Returns mock video generation responses:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

# Dev mode simulates video generation
result = client.videos.generate(
    model="wan2.2",
    prompt="A cat playing with a ball of yarn"
)

# Returns processing status with mock data
print(result.status)  # "processing"
print(result.progress)  # 50
```

### Mock Response Behavior

Dev Mode responses include:

- **Correct response format** - Matches production API exactly
- **Valid image data** - Returns actual test images (not broken links)
- **Proper metadata** - Includes size, format, quality fields
- **Multiple images** - Respects `n` parameter for batch generation
- **Format options** - Supports both `url` and `b64_json` response formats
- **All parameters** - Accepts and validates all API parameters

### Use Cases

Dev Mode is perfect for:

| Use Case | Description |
|----------|-------------|
| 🧪 **API Testing** | Validate integration without GPU |
| 🔧 **Development** | Build features faster without model loading delays |
| 📝 **Documentation** | Generate examples and test code snippets |
| 🔄 **CI/CD** | Run automated tests in pipelines |
| 💻 **Local Development** | Work on machines without powerful GPUs |
| 🎓 **Learning** | Explore the API without hardware requirements |


### Response Formats

Dev Mode supports all response formats:

**URL Format (default):**
```python
result = client.images.generate(
    prompt="test",
    response_format="url"
)
print(result.data[0].url)  # Returns image URL
```

**Base64 Format:**
```python
result = client.images.generate(
    prompt="test",
    response_format="b64_json"
)
print(result.data[0].b64_json)  # Returns base64 encoded image
```

### Limitations

Dev Mode has the following limitations:

- **Mock images only** - Does not generate real AI-created images
- **No prompt interpretation** - Mock responses ignore prompt content
- **Fixed test images** - Returns the same test images for all requests
- **No model differences** - All models return identical mock responses
- **Development only** - Not suitable for production use

### Best Practices

1. **Use for development** - Dev Mode is perfect for building features
2. **Test with production** - Always validate with real models before deploying
3. **Document clearly** - Make it obvious when using Dev Mode in code comments
4. **Switch easily** - Design your code to work with both modes
5. **Monitor logs** - Check for `[DEV MODE]` indicators to verify mode

