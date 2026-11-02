# RuralHealth Vision

**GitHub repository:** https://github.com/Its-Akshat-0/Healthcare-image-Generation

This repository contains the source code, medical passage dataset, evaluation and
training scaffolding, and the project report. Add the final presentation slides
to this repository before submission; no slide deck is currently included.

A college-minor-project prototype for source-linked rural health education. It combines:
- a source-linked health passage collection, including imported records that still need human review,
- keyword retrieval (baseline; replace/extend with embeddings),
- optional Hugging Face extractive QA model loading,
- optional Diffusers artwork with locally composed, source-grounded poster text,
- Flask web UI and evaluation scaffolding.

**Medical safety:** this is an educational prototype, not a diagnostic tool. Review the included passages against their current official sources before using them as guidance. Do not report model fine-tuning as completed until you actually run training and evaluate the saved model.

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

When diffusion is connected, it creates varied portrait artwork and Flask uses Pillow to overlay readable health copy in one of several poster compositions. Text and citations come from the reviewed local dataset; Diffusers supplies the artwork only. The image backend automatically uses a local CUDA GPU when CUDA-enabled PyTorch and Diffusers are installed. Otherwise it uses the configured Colab service. If neither backend is available, the app retains the designed SVG infographic fallback. Check the selected model card and licence before use.

### Local GPU setup

Install the image packages in the same Python environment as Flask:

```powershell
pip install -r requirements-image.txt
```

Install the CUDA-enabled PyTorch build appropriate for your GPU using the
official PyTorch installation selector. Set `DIFFUSION_MODEL_ID` if you want a
different Diffusers-compatible model. The first local run downloads the model;
later requests reuse the loaded pipeline. You do not need to set
`ENABLE_DIFFUSION`.

### Google Colab fallback

When the local machine has no usable CUDA GPU, the image backend uses the
Colab endpoint configured in `COLAB_IMAGE_API_URL` and
`COLAB_IMAGE_API_TOKEN`. Open `notebooks/ruralhealthvision_colab.ipynb` in
Google Colab and follow its cells to start the temporary GPU service. Keep the
notebook runtime and tunnel active while requesting images. Copy the printed
URL and configure the two environment variables in PowerShell as shown below.
The Flask app, retrieval, source links, and poster-text composition remain
local. Only the disease name and fixed image topic are sent to Colab.

## Project folders

- `data/medical_documents.jsonl`: source-linked passage collection; review imported records marked `NOT_HUMAN_REVIEWED`
- `scripts/`: dataset validation and MedlinePlus XML import scaffold
- `training/`: fine-tuning starter and QA templates
- `notebooks/ruralhealthvision_colab.ipynb`: Colab GPU image-service setup
- `presentation/`: project presentation slides
- `RuralHealth_Vision_Minor_Project_Report_Updated.docx`: project report
- `image_generation.py`: optional diffusion integration
- `evaluation/evaluate_retrieval.py`: simple retrieval sanity-check scaffold
- `templates/`, `static/`: web UI

## What to report honestly

Report actual counts, training logs, test-set metrics, image latency, and screenshots from your own run. If training is not completed by the deadline, describe it as planned/future work rather than claiming results.


## Configure the Colab fallback

Follow `notebooks/ruralhealthvision_colab.ipynb` to install the service dependencies, load the image model on a Colab GPU, and start the authenticated tunnel. Keep the notebook cell running. Copy the printed temporary tunnel base URL without `/generate`.

In the same PowerShell window used to launch Flask, set the URL printed by Colab and the same image token stored in Colab Secrets:

```powershell
$env:COLAB_IMAGE_API_URL = "https://YOUR-TEMPORARY-NGROK-URL"
$env:COLAB_IMAGE_API_TOKEN = "YOUR_LONG_RANDOM_IMAGE_TOKEN"
python app.py
```

These environment values last only for that PowerShell window. Restart Flask after Colab gives you a new tunnel URL. Keep the token private and do not commit it. The endpoint receives only the disease name and fixed image topic, and returns artwork without medical text. Flask composes the poster locally. If Colab is offline and no local GPU is available, the SVG infographic fallback appears. The generated image is illustrative and is not a medical source.
