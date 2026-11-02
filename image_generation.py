"""Image backends for RuralHealth Vision.

Colab is optional and only available while its notebook runtime and tunnel are running.
"""
import io
import os
import re
import random
from pathlib import Path

import requests

_LOCAL_PIPELINE = None
_LOCAL_PIPELINE_MODEL = None

class ImageBackendUnavailable(RuntimeError):
    pass

def _build_prompt(disease, topic):
    disease = re.sub(r"[^a-zA-Z0-9 -]", "", str(disease))[:80]
    topic = re.sub(r"[^a-zA-Z0-9 -]", "", str(topic))[:40]
    art_direction = random.choice([
        "vibrant hand-painted gouache illustration",
        "bold editorial screen-print illustration",
        "layered paper-cut illustration with soft shadows",
        "warm contemporary storybook illustration",
    ])
    return (
        f"A {art_direction} for a respectful rural Indian community health campaign. "
        f"Create one coherent vertical scene about {disease}, topic: {topic}, "
        "with expressive people, a rich environmental background, and a clear focal point. "
        "Portrait composition, no collage, no grid, no panels, no frame. Artwork only: "
        "no words, letters, labels, numbers, logos, or medical claims."
    )

def generate_illustration(disease, topic, facts, output_path):
    """Generate only the illustration; sourced poster text is composed separately."""
    endpoint = os.getenv("COLAB_IMAGE_API_URL", "").strip().rstrip("/")
    # Prefer a usable local CUDA GPU automatically. If CUDA/Diffusers is not
    # available, use the configured Colab endpoint instead.
    local_error = None
    try:
        import torch
        if torch.cuda.is_available():
            return _generate_local(disease, topic, output_path, torch)
    except ImportError as exc:
        local_error = exc
    except Exception as exc:
        local_error = exc

    if endpoint:
        token = os.getenv("COLAB_IMAGE_API_TOKEN", "").strip()
        if not token:
            raise ImageBackendUnavailable("Set COLAB_IMAGE_API_TOKEN for the Colab image service.")
        try:
            response = requests.post(
                endpoint + "/generate",
                json={"disease": disease, "topic": topic},
                headers={"Authorization": f"Bearer {token}"},
                timeout=(10, 180),
            )
            if not response.ok:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail") or error_data.get("error")
                except (ValueError, AttributeError):
                    detail = response.text[:300]
                detail = str(detail or response.reason)[:300]
                raise ImageBackendUnavailable(
                    f"Colab image service returned HTTP {response.status_code}: {detail}"
                )
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/png"):
                raise ImageBackendUnavailable("Colab service did not return a PNG image.")
            Path(output_path).write_bytes(response.content)
            return str(output_path)
        except requests.RequestException as exc:
            raise ImageBackendUnavailable(f"Colab image service is unavailable: {exc}") from exc

    if local_error:
        detail = f"Local GPU generation failed: {local_error}. "
    else:
        detail = "No local CUDA GPU is available. "
    raise ImageBackendUnavailable(
        detail + "Configure the temporary Colab service, or install the image dependencies and a compatible CUDA-enabled PyTorch build."
    )


def _generate_local(disease, topic, output_path, torch):
    """Load the local Diffusers pipeline once and reuse it for later requests."""
    global _LOCAL_PIPELINE, _LOCAL_PIPELINE_MODEL
    from diffusers import AutoPipelineForText2Image

    model_id = os.getenv("DIFFUSION_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    if _LOCAL_PIPELINE is None or _LOCAL_PIPELINE_MODEL != model_id:
        _LOCAL_PIPELINE = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        _LOCAL_PIPELINE_MODEL = model_id
    image = _LOCAL_PIPELINE(
        prompt=_build_prompt(disease, topic),
        negative_prompt="text, watermark, graphic injury, misleading medical equipment",
        width=512, height=768, num_inference_steps=24,
    ).images[0]
    image.save(output_path)
    return str(output_path)

# Keep the old import name usable for existing scripts.
def generate_image(disease, topic, facts, output_path):
    return generate_illustration(disease, topic, facts, output_path)
