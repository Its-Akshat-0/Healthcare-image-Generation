# RuralHealth Vision

A college-minor-project prototype for source-linked rural health education. It combines:
- a small, clearly labelled starter knowledge base,
- keyword retrieval (baseline; replace/extend with embeddings),
- optional Hugging Face extractive QA model loading,
- optional Diffusers artwork with locally composed, source-grounded poster text,
- Flask web UI and evaluation scaffolding.

**Medical safety:** this is an educational prototype, not a diagnostic tool. The starter data is deliberately small and should be checked against the current official sources before submission or real-world use. Do not report model fine-tuning as completed until you actually run training and evaluate the saved model.

## Quick start (Windows PowerShell)

```powershell
cd RuralHealthVision
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal, then activate again.

## Data collection

1. Review `data/medical_documents.jsonl`.
2. Run `python scripts/validate_dataset.py`.
3. Use `scripts/collect_medlineplus_xml.py` to download MedlinePlus XML if network access is available. Review the source's terms and content before including it.
4. Add only reviewed official source passages. Preserve the original URL, title, date, and source.
5. Do not scrape indiscriminately or generate invented medical claims.

Each JSONL line is one short, focused passage. Keep the existing fields
`document_id`, `disease`, `section`, `text`, `source_name`, `source_url`,
`language`, and `last_verified`. Use sections such as `causes`,
`transmission`, `symptoms`, `prevention`, and `treatment`. Add a new line
with a unique `document_id` for each subject and disease. Keep the source
passage faithful to its linked official page, record when a person reviewed it,
and recheck it periodically. Avoid putting several unrelated answers into one
long passage: retrieval works best when each entry covers one question.

The question box now suggests cause, spread, treatment, and prevention questions.
Retrieval gives matching subject sections a ranking boost. The word “cure” is
treated as a treatment question; the app does not promise that every disease has
a cure or provide individualized treatment instructions. If the dataset has no
relevant entry for a disease, add reviewed source passages first.

### Importing a larger health-topic collection

The optional importer downloads MedlinePlus's latest English and Spanish health
topic archive, keeps English topics with substantive summaries, and stores the
source URL with each passage. Run it from the project folder in PowerShell:

```powershell
python scripts/import_medlineplus_topics.py
```

It preserves manually curated records and refreshes prior MedlinePlus imports.
It creates a general overview from each topic summary and creates cause,
treatment, or prevention entries only when the summary contains sentences
explicitly about that subject. The importer marks these machine-collected
records `NOT_HUMAN_REVIEWED`; review them before using them as health guidance.
The number of topics is determined by the current MedlinePlus archive, and not
every topic is a disease or has all three subject sections. Existing WHO records
remain alongside the MedlinePlus content. Restart Flask after importing.

## Transformer fine-tuning (Google Colab)

Upload `training/qa_train_template.jsonl` and `training/qa_validation_template.jsonl` after replacing the examples with manually checked answer spans. Then run `training/train_qa.py` in Colab. The script expects SQuAD-style fields:
`question`, `context`, `answers: {"text": ["answer"], "answer_start": [character_offset]}`.

The starter examples are templates and are not sufficient for a meaningful medical model. Split data by source/document to reduce leakage. Inspect preprocessing and evaluation carefully, especially long contexts.

## Optional image generation

When diffusion is connected, it creates varied portrait artwork and Flask uses Pillow to overlay readable health copy in one of several poster compositions. Text and citations come from the reviewed local dataset; Diffusers supplies the artwork only. For local generation, a compatible CUDA GPU, model download, and licence review are required; set `ENABLE_DIFFUSION=1` and `DIFFUSION_MODEL_ID` after checking the model card. For a Windows machine without a suitable GPU, use the temporary Colab bridge below. When image generation is unavailable, the app uses the designed SVG infographic fallback.

## Project folders

- `data/medical_documents.jsonl`: small starter set, not a large corpus
- `scripts/`: dataset validation and MedlinePlus XML import scaffold
- `training/`: fine-tuning starter and QA templates
- `image_generation.py`: optional diffusion integration
- `evaluation/evaluate_retrieval.py`: simple retrieval sanity-check scaffold
- `templates/`, `static/`: web UI

## What to report honestly

Report actual counts, training logs, test-set metrics, image latency, and screenshots from your own run. If training is not completed by the deadline, describe it as planned/future work rather than claiming results.


## Stable Diffusion from Colab (optional)

The Flask app, DistilBERT QA model, retrieval, and poster text stay on Windows. Stable Diffusion runs in the Colab GPU runtime and returns portrait artwork. Flask overlays retrieved health copy using varied poster compositions. The Colab endpoint is temporary: it works only while the notebook runtime, server cell, and ngrok tunnel are active. If Colab disconnects, the Flask app still returns source-grounded text and uses the designed SVG infographic fallback.

### Start the image service in Colab

1. Open the notebook where your `StableDiffusionPipeline` is already loaded as `pipe`.
2. Install the small tunnel helper in a cell: `!pip -q install pyngrok`.
3. Add two Colab Secrets: `NGROK_AUTHTOKEN` (from your ngrok account) and `RH_IMAGE_TOKEN` (a long random secret). Enable notebook access to both secrets.
4. Run this in a cell:

```python
import os
from google.colab import userdata
from colab_image_service import start_colab_image_service
os.environ["NGROK_AUTHTOKEN"] = userdata.get("NGROK_AUTHTOKEN")
os.environ["RH_IMAGE_TOKEN"] = userdata.get("RH_IMAGE_TOKEN")
server, tunnel = start_colab_image_service(pipe)
```

If the module file is not in the notebook working directory, upload `colab_image_service.py` from this project first. Keep the cell running. Copy the printed temporary tunnel base URL (without `/generate`).

### Point the Windows Flask app to the active tunnel

In the same PowerShell window used to launch Flask, set the URL printed by Colab and the exact same image token stored in Colab Secrets:

```powershell
$env:COLAB_IMAGE_API_URL = "https://YOUR-TEMPORARY-NGROK-URL"
$env:COLAB_IMAGE_API_TOKEN = "YOUR_LONG_RANDOM_IMAGE_TOKEN"
python app.py
```

These environment values last only for that PowerShell window. Restart Flask after Colab gives you a new tunnel URL. Keep the token private and do not commit it. This bridge accepts only a disease name and a fixed topic; it does not send medical passages to the image model.

The endpoint returns an illustration PNG only. Flask composes a PNG poster from that artwork, overlaid with readable passages and source attribution. The design changes position and color treatment between requests; it does not ask Stable Diffusion to draw words. If Colab is offline, the SVG infographic fallback appears instead. The Stable Diffusion image is illustrative and is not a medical source. Keep the checkbox off to skip image generation when you do not need a poster.
