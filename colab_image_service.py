"""Run in the already initialized Google Colab notebook.

Install pyngrok in a Colab cell, set NGROK_AUTHTOKEN and RH_IMAGE_TOKEN in
Colab Secrets, then call start_colab_image_service(pipe, ...). Keep the runtime
and this server cell running while the local Flask app requests illustrations.
"""
import hmac
import os
import random
import re
from io import BytesIO
from threading import Thread

from flask import Flask, jsonify, request, send_file
from pyngrok import conf, ngrok
from werkzeug.serving import make_server

_ALLOWED_TOPICS = {"prevention", "transmission", "awareness", "hygiene"}

def _prompt(disease, topic):
    disease = re.sub(r"[^a-zA-Z0-9 -]", "", str(disease))[:80]
    if topic not in _ALLOWED_TOPICS:
        raise ValueError("Unsupported image topic")
    art_direction = random.choice([
        "vibrant hand-painted gouache illustration",
        "bold editorial screen-print illustration",
        "layered paper-cut illustration with soft shadows",
        "warm contemporary storybook illustration",
    ])
    return (
        f"A {art_direction} for a respectful rural Indian community health campaign. "
        f"Create one coherent vertical scene about {disease}, topic {topic}, "
        "with expressive people, a rich environmental background, and a clear focal point. "
        "Portrait composition, no collage, no grid, no panels, no frame. Artwork only: "
        "no words, letters, labels, numbers, logos, or medical claims."
    )

def start_colab_image_service(pipe, token=None, port=7860):
    """Expose the already loaded Diffusers pipeline through an authenticated tunnel."""
    token = str(token or os.environ.get("RH_IMAGE_TOKEN") or "").strip()
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token or not ngrok_token:
        raise RuntimeError("Set RH_IMAGE_TOKEN and NGROK_AUTHTOKEN in Colab Secrets first.")
    # Use this token for the process without trying to rewrite ngrok.yml.
    # Colab may already have an authtoken saved there, and `set_auth_token`
    # fails with "Authtoken already set in configuration file" in that case.
    ngrok_config = conf.PyngrokConfig(auth_token=ngrok_token)
    api = Flask("ruralhealth_colab_image")

    @api.post("/generate")
    def generate():
        auth = request.headers.get("Authorization", "")
        supplied = auth.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, token):
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True) or {}
        disease = str(payload.get("disease", "")).strip()
        topic = str(payload.get("topic", "")).strip().lower()
        if not disease or len(disease) > 80 or topic not in _ALLOWED_TOPICS:
            return jsonify({"error": "invalid disease or topic"}), 400
        try:
            result = pipe(
                prompt=_prompt(disease, topic),
                negative_prompt="text, watermark, graphic injury, misleading medical equipment",
                width=512, height=768, num_inference_steps=24,
            )
            image = result.images[0]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return send_file(buffer, mimetype="image/png", download_name="illustration.png")
        except Exception as exc:
            api.logger.exception("Image generation failed")
            return jsonify({"error": "generation failed", "detail": str(exc)[:300]}), 500

    server = make_server("0.0.0.0", port, api, threaded=True)
    Thread(target=server.serve_forever, daemon=True).start()
    tunnel = ngrok.connect(port, "http", pyngrok_config=ngrok_config)
    public_url = tunnel.public_url
    print("RuralHealth Vision Colab image endpoint:", public_url + "/generate")
    print("Set COLAB_IMAGE_API_URL to:", public_url)
    print("Stop the notebook cell or disconnect the runtime to close this temporary endpoint.")
    return server, tunnel
