import json
import os
import re
import base64
import random
from difflib import SequenceMatcher
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "medical_documents.jsonl"
MODEL_PATH = ROOT / "models" / "ruralhealth-qa"
GENERATED = ROOT / "generated"
GENERATED.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

class _SummaryTextParser(HTMLParser):
    _BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "br", "div"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)

def clean_passage(text):
    """Remove embedded MedlinePlus HTML while preserving readable word spacing."""
    parser = _SummaryTextParser()
    parser.feed(unescape(str(text or "")))
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()

def source_excerpt(doc, max_chars=650):
    text = clean_passage(doc.get("text", ""))
    if len(text) <= max_chars:
        return "The model could not extract a direct answer. Relevant source passage: " + text
    excerpt = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:")
    return "The model could not extract a direct answer. Relevant source excerpt: " + excerpt + "…"

def _poster_summary(text, max_chars=260):
    text = clean_passage(text)
    text = re.sub(
        r"^The model could not extract a direct answer\. Relevant source (?:passage|excerpt):\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "…"

def _poster_key_points(docs, limit=4, max_chars=145):
    """Pick a few short, distinct source facts; skip overview and fragments."""
    points = []
    section_counts = {}
    seen = set()
    for doc in docs:
        section = doc.get("section", "health_information").lower()
        if section in {"overview", "introduction"} or section_counts.get(section, 0) >= 2:
            continue
        text = clean_passage(doc.get("text", ""))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            point = re.sub(r"\s+", " ", sentence).strip(" \t\r\n-•.;:")
            words = re.findall(r"[A-Za-z0-9]+", point)
            if len(words) < 6:
                continue
            key = " ".join(words).lower()
            if key in seen:
                continue
            if len(point) > max_chars:
                point = point[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "…"
            points.append((section.replace("_", " ").title(), point))
            seen.add(key)
            section_counts[section] = section_counts.get(section, 0) + 1
            if len(points) >= limit:
                return points
            if section_counts[section] >= 2:
                break
    return points

def load_docs():
    docs = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc["text"] = clean_passage(doc.get("text", ""))
                docs.append(doc)
    return docs

DOCS = load_docs()
DISEASES = sorted({doc["disease"].strip() for doc in DOCS if doc.get("disease", "").strip()})

def _normalize_topic(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def resolve_disease(value):
    """Return an exact or confidently close topic name, plus a user-facing note."""
    requested = str(value or "").strip()
    normalized = _normalize_topic(requested)
    by_normalized = {_normalize_topic(name): name for name in DISEASES}
    if normalized in by_normalized:
        return by_normalized[normalized], ""
    if not normalized or not DISEASES:
        return requested.lower(), ""

    requested_tokens = " ".join(sorted(normalized.split()))
    best_name, best_score = "", 0.0
    for name in DISEASES:
        candidate = _normalize_topic(name)
        candidate_tokens = " ".join(sorted(candidate.split()))
        score = max(
            SequenceMatcher(None, normalized, candidate).ratio(),
            SequenceMatcher(None, requested_tokens, candidate_tokens).ratio(),
        )
        if score > best_score:
            best_name, best_score = name, score

    cutoff = 0.78 if len(normalized.replace(" ", "")) < 6 else 0.70
    if best_score >= cutoff:
        return best_name, f"No exact match for “{requested}”; showing the closest match, “{best_name}”."
    if best_name:
        return requested.lower(), f"No close match for “{requested}”. The closest topic in the database is “{best_name}”; check the spelling and try again."
    return requested.lower(), ""

# Load lazily so the Flask app still starts without optional ML packages.
_qa_tokenizer = None
_qa_model = None
_qa_load_error = None

def get_qa_model():
    global _qa_tokenizer, _qa_model, _qa_load_error
    if _qa_model is not None:
        return _qa_tokenizer, _qa_model
    if _qa_load_error is not None or not MODEL_PATH.is_dir():
        _qa_load_error = _qa_load_error or f"Model folder not found: {MODEL_PATH}"
        return None, None
    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        _qa_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _qa_model.eval()
        return _qa_tokenizer, _qa_model
    except Exception as exc:
        _qa_load_error = str(exc)
        print("QA model could not be loaded:", exc)
        return None, None

def extract_answer(question, doc):
    """Extract an exact answer span from one retrieved passage."""
    tokenizer, model = get_qa_model()
    if tokenizer is None or model is None:
        return None
    import torch
    try:
        encoded = tokenizer(question, doc["text"], return_tensors="pt",
                            truncation="only_second", max_length=512,
                            return_offsets_mapping=True)
        context_tokens = encoded.sequence_ids(0)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            outputs = model(**encoded)
        starts = torch.softmax(outputs.start_logits[0], dim=0)
        ends = torch.softmax(outputs.end_logits[0], dim=0)
        best = None
        for start, start_offset in enumerate(offsets):
            if context_tokens[start] != 1 or start_offset == [0, 0]:
                continue
            for end in range(start, min(len(offsets), start + 31)):
                if context_tokens[end] != 1 or offsets[end] == [0, 0]:
                    continue
                score = float(starts[start] * ends[end])
                if best is None or score > best[0]:
                    best = (score, start, end)
        if best is None or best[0] < 0.01:
            return None
        answer = doc["text"][offsets[best[1]][0]:offsets[best[2]][1]].strip()
        return answer or None
    except Exception as exc:
        print("QA inference failed:", exc)
        return None

def tokenize(text):
    return set(re.findall(r"[a-zA-Z]{2,}", text.lower()))

def question_intent(question):
    q = question.lower()
    if any(word in q for word in ("prevent", "avoid", "protection", "stop getting")):
        return "prevention"
    if any(word in q for word in ("treat", "treatment", "cure", "medicine", "medication", "recover", "care")):
        return "treatment"
    if any(word in q for word in ("cause", "caused", "why", "reason", "germ", "bacteria", "virus", "parasite")):
        return "causes"
    if any(word in q for word in ("spread", "transmit", "contagious", "catch", "infection")):
        return "transmission"
    return None

def retrieve(query, disease=None, limit=5, intent=None):
    qtokens = tokenize(query)
    scored = []
    for doc in DOCS:
        if disease and disease.lower() not in doc["disease"].lower():
            continue
        dtokens = tokenize(doc["disease"] + " " + doc["section"] + " " + doc["text"])
        score = len(qtokens & dtokens) / max(1, len(qtokens))
        if disease and disease.lower() in doc["disease"].lower():
            score += 0.6
        # Questions about causes, treatment, and prevention should surface
        # passages explicitly curated for that subject before generic overview.
        section = doc.get("section", "").lower()
        if intent == "causes" and section in {"causes", "overview"}:
            score += 0.8 if section == "causes" else 0.25
        elif intent == "treatment" and section in {"treatment", "care_and_warning_signs", "urgent_action"}:
            score += 0.8 if section == "treatment" else 0.2
        elif intent == "prevention" and section in {"prevention", "prevention_and_care"}:
            score += 0.8
        elif intent == "transmission" and section in {"transmission", "overview"}:
            score += 0.5
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:limit]]

def _wrap_words(text, limit):
    words = str(text).split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + bool(line) > limit:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    return lines


def compose_poster(disease, topic, docs, sources, illustration_path, poster_path, summary=None):
    """Render a compact poster with a short summary and selected sourced facts."""
    palette = [
        ("#fff0ef", "#b52b39", "#f8dadd"),
        ("#e9f7f0", "#08745f", "#d2eee3"),
        ("#edf5ff", "#155c9b", "#d8e9fa"),
        ("#fff7dc", "#966600", "#f5e8b9"),
        ("#f5edfb", "#75409a", "#e9dbf5"),
        ("#e9f7f8", "#087e87", "#d2eef0"),
    ]
    section_titles = {
        "overview": "At a glance",
        "prevention": "Prevention",
        "care_and_warning_signs": "Care and warning signs",
        "urgent_action": "When to seek care",
        "prevention_and_care": "Prevention and care",
    }
    intro = _poster_summary(summary) if summary else "Review the key health points and linked sources below."
    cards_by_section = {}
    for title, text in _poster_key_points(docs):
        cards_by_section.setdefault(title, []).append(text)
    cards = list(cards_by_section.items())

    card_h = max(
        250,
        max((sum(len(_wrap_words(text, 46)) for text in texts) * 23 + 125 for _, texts in cards), default=250),
    )
    row_count = (len(cards) + 1) // 2
    source_rows = min(len(sources), 3)
    height = max(900, 440 + row_count * (card_h + 22) + 250)
    width = 1000
    chunks = [f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <linearGradient id="hero" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#e5f6ef"/><stop offset="1" stop-color="#dceefa"/></linearGradient>
  <linearGradient id="title" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#a91f2c"/><stop offset="1" stop-color="#d44445"/></linearGradient>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="5" stdDeviation="7" flood-color="#204d43" flood-opacity=".10"/></filter>
</defs>
<rect width="1000" height="{height}" fill="#f4f8f5"/>
<rect x="22" y="22" width="956" height="{height - 44}" rx="28" fill="#ffffff" filter="url(#shadow)"/>
<circle cx="69" cy="70" r="28" fill="#0a7564"/><path d="M69 52v36M51 70h36" stroke="#fff" stroke-width="8" stroke-linecap="round"/>
<text x="112" y="66" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#075b50">RuralHealth Vision</text>
<text x="112" y="91" font-family="Arial,sans-serif" font-size="13" fill="#52736c">HEALTHY VILLAGES · STRONGER COMMUNITIES</text>
<text x="55" y="166" font-family="Arial,sans-serif" font-size="68" font-weight="800" letter-spacing="1" fill="url(#title)">{escape(disease.upper())}</text>
<rect x="55" y="183" width="500" height="44" rx="20" fill="#08745f"/>
<text x="76" y="212" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#fff">Know the facts  ·  Protect  ·  Stay healthy</text>
<rect x="55" y="250" width="890" height="174" rx="24" fill="url(#hero)"/>
<text x="82" y="289" font-family="Arial,sans-serif" font-size="15" font-weight="700" letter-spacing="1" fill="#08745f">SUMMARY</text>
''']
    intro_lines = _wrap_words(intro, 62)[:4]
    for i, line in enumerate(intro_lines):
        chunks.append(f'<text x="82" y="{324 + i * 27}" font-family="Arial,sans-serif" font-size="18" fill="#173d36">{escape(line)}</text>')

    if illustration_path and Path(illustration_path).is_file():
        encoded = base64.b64encode(Path(illustration_path).read_bytes()).decode("ascii")
        chunks.append(f'<image x="674" y="257" width="250" height="158" preserveAspectRatio="xMidYMid slice" href="data:image/png;base64,{encoded}"/>')
        art_label = "AI-GENERATED ILLUSTRATION"
    else:
        # A friendly, disease-neutral community scene keeps the poster complete offline.
        chunks.append('''<circle cx="800" cy="332" r="71" fill="#ffffff" opacity=".75"/>
<path d="M714 345l85-69 85 69v57h-170z" fill="#15947a"/><path d="M701 347l98-80 98 80" fill="none" stroke="#08745f" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="779" y="350" width="37" height="52" rx="4" fill="#ffcf70"/><rect x="733" y="353" width="30" height="28" rx="3" fill="#d7f3ec"/><rect x="840" y="353" width="30" height="28" rx="3" fill="#d7f3ec"/>
<circle cx="730" cy="291" r="13" fill="#f4b38d"/><path d="M714 326q16-28 32 0v16h-32z" fill="#e86458"/>
<circle cx="867" cy="293" r="13" fill="#b97050"/><path d="M850 328q17-28 34 0v16h-34z" fill="#307fb1"/>
<path d="M723 344l-17 25M876 344l17 25" stroke="#277d63" stroke-width="6" stroke-linecap="round"/>''')
        art_label = "COMMUNITY HEALTH ILLUSTRATION"
    chunks.append(f'<text x="674" y="415" font-family="Arial,sans-serif" font-size="11" font-weight="700" letter-spacing="1" fill="#52736c">{art_label}</text>')

    y0 = 455
    for index, (title, texts) in enumerate(cards):
        col, row = index % 2, index // 2
        x, y = 55 + col * 455, y0 + row * (card_h + 22)
        bg, accent, badge = palette[index % len(palette)]
        chunks.append(f'<rect x="{x}" y="{y}" width="435" height="{card_h}" rx="22" fill="{bg}"/>')
        chunks.append(f'<circle cx="{x + 43}" cy="{y + 43}" r="25" fill="{accent}"/>')
        title_lower = title.lower()
        if "prevent" in title_lower:
            mark = f'<path d="M{x + 43} {y + 27}l13 5v10q0 12-13 18q-13-6-13-18v-10z" fill="none" stroke="#fff" stroke-width="3"/><path d="M{x + 36} {y + 42}l5 5 10-11" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        elif "warning" in title_lower or "urgent" in title_lower or "seek" in title_lower:
            mark = f'<path d="M{x + 43} {y + 29}v17" stroke="#fff" stroke-width="5" stroke-linecap="round"/><circle cx="{x + 43}" cy="{y + 52}" r="2.5" fill="#fff"/>'
        elif "care" in title_lower:
            mark = f'<path d="M{x + 43} {y + 56}s-16-10-16-20a8 8 0 0 1 16-3 8 8 0 0 1 16 3c0 10-16 20-16 20z" fill="none" stroke="#fff" stroke-width="3"/>'
        else:
            mark = f'<path d="M{x + 43} {y + 30}v26M{x + 30} {y + 43}h26" stroke="#fff" stroke-width="5" stroke-linecap="round"/>'
        chunks.append(mark)
        chunks.append(f'<text x="{x + 80}" y="{y + 49}" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="{accent}">{escape(title[:28])}</text>')
        line_y = y + 91
        for text_value in texts:
            lines = _wrap_words(text_value, 46)
            for line in lines:
                if line_y > y + card_h - 18:
                    break
                chunks.append(f'<circle cx="{x + 26}" cy="{line_y - 6}" r="4" fill="{accent}"/>')
                chunks.append(f'<text x="{x + 40}" y="{line_y}" font-family="Arial,sans-serif" font-size="16" fill="#203b35">{escape(line)}</text>')
                line_y += 23
            line_y += 7

    foot_y = y0 + row_count * (card_h + 22) + 5
    chunks.append(f'<text x="55" y="{foot_y}" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#075b50">SOURCES</text>')
    for i, source in enumerate(sources[:3]):
        label = f"{source.get('name', 'Source')}: {source.get('url', '')}"
        lines = _wrap_words(label, 105)
        for j, line in enumerate(lines[:2]):
            chunks.append(f'<text x="65" y="{foot_y + 25 + i * 40 + j * 16}" font-family="Arial,sans-serif" font-size="11" fill="#415c55">{escape(line)}</text>')
    footer_y = height - 38
    chunks.append(f'<path d="M22 {footer_y - 24} H978 V{height - 50} Q978 {height - 22} 950 {height - 22} H50 Q22 {height - 22} 22 {height - 50} Z" fill="#075b50"/>')
    chunks.append(f'<text x="55" y="{footer_y + 3}" font-family="Arial,sans-serif" font-size="13" font-weight="700" fill="#fff">Educational information · Not a diagnosis or replacement for professional care</text>')
    chunks.append('</svg>')
    Path(poster_path).write_text("\n".join(chunks), encoding="utf-8")


def _poster_font(size, bold=False):
    from PIL import ImageFont
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


def _wrap_pixels(draw, text, font, max_width):
    words = str(text).split()
    lines, line = [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=font) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def compose_diffusion_poster(disease, topic, answer, docs, sources, illustration_path, poster_path):
    """Build a varied full-bleed poster from Diffusers artwork and source-grounded copy."""
    from PIL import Image, ImageDraw, ImageOps

    width, height = 1024, 1536
    with Image.open(illustration_path) as source_image:
        canvas = ImageOps.fit(source_image.convert("RGB"), (width, height), centering=(0.5, 0.42)).convert("RGBA")

    styles = [
        {"panel": (7, 56, 58, 232), "body": (255, 255, 248, 255), "accent": (255, 205, 120, 255), "layout": "top_bottom"},
        {"panel": (250, 245, 230, 238), "body": (28, 46, 42, 255), "accent": (170, 42, 52, 255), "layout": "side"},
        {"panel": (20, 39, 68, 232), "body": (255, 255, 255, 255), "accent": (124, 226, 193, 255), "layout": "floating"},
    ]
    style = random.SystemRandom().choice(styles)
    layout = style["layout"]
    heading_font = _poster_font(66 if layout != "side" else 58, bold=True)
    brand_font = _poster_font(22, bold=True)
    label_font = _poster_font(20, bold=True)
    body_font = _poster_font(22 if layout != "side" else 19)
    small_font = _poster_font(15)
    draw = ImageDraw.Draw(canvas, "RGBA")

    def panel(box, radius=30):
        draw.rounded_rectangle(box, radius=radius, fill=style["panel"])

    def paragraph(text, x, y, max_width, font, fill, line_height, max_lines=None):
        lines = _wrap_pixels(draw, text, font, max_width)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" .,;:") + "..."
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        return y

    title = disease.upper()
    topic_label = "Health summary"
    answer = _poster_summary(answer)
    facts = _poster_key_points(docs)

    if layout == "top_bottom":
        panel((34, 34, width - 34, 420), radius=34)
        draw.text((72, 66), "RURALHEALTH VISION  /  HEALTH EDUCATION", font=brand_font, fill=style["accent"])
        draw.text((68, 112), title, font=heading_font, fill=style["body"])
        draw.text((72, 205), f"{topic_label}  ·  Community health guide", font=label_font, fill=style["accent"])
        paragraph(answer, 72, 252, width - 150, body_font, style["body"], 31, max_lines=4)
        panel((34, 790, width - 34, height - 34), radius=34)
        y = 825
        for section, text in facts:
            draw.text((72, y), section.upper(), font=label_font, fill=style["accent"])
            y += 34
            y = paragraph(text, 72, y, width - 148, body_font, style["body"], 29, max_lines=3)
            y += 15
        draw.text((72, height - 95), "SOURCES  " + "  ·  ".join(s.get("name", "Source") for s in sources[:3]), font=small_font, fill=style["accent"])

    elif layout == "side":
        panel((28, 28, 558, height - 28), radius=34)
        draw.text((62, 65), "RURALHEALTH VISION", font=brand_font, fill=style["accent"])
        title_y = paragraph(title, 62, 112, 440, heading_font, style["body"], 68, max_lines=2)
        draw.text((64, title_y + 4), topic_label.upper(), font=label_font, fill=style["accent"])
        y = paragraph(answer, 64, title_y + 45, 430, body_font, style["body"], 26, max_lines=5) + 24
        for section, text in facts:
            draw.text((64, y), section.upper(), font=label_font, fill=style["accent"])
            y += 32
            y = paragraph(text, 64, y, 430, body_font, style["body"], 25, max_lines=3)
            y += 18
        source_names = " · ".join(s.get("name", "Source") for s in sources[:2])
        draw.text((64, height - 78), source_names[:52], font=small_font, fill=style["accent"])

    else:  # floating translucent cards leave most of the generated scene visible
        panel((34, 34, width - 34, 340), radius=36)
        draw.text((72, 66), "RURALHEALTH VISION", font=brand_font, fill=style["accent"])
        draw.text((68, 108), title, font=heading_font, fill=style["body"])
        draw.text((72, 196), topic_label.upper(), font=label_font, fill=style["accent"])
        paragraph(answer, 72, 235, width - 150, body_font, style["body"], 29, max_lines=3)
        card_y = 835
        card_w, card_h, gap = 452, 276, 32
        for index, (section, text) in enumerate(facts):
            col, row = index % 2, index // 2
            x, y = 34 + col * (card_w + gap), card_y + row * (card_h + gap)
            draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=28, fill=style["panel"])
            draw.text((x + 28, y + 24), section.upper()[:27], font=label_font, fill=style["accent"])
            paragraph(text, x + 28, y + 68, card_w - 56, body_font, style["body"], 27, max_lines=5)
        draw.text((72, height - 70), "SOURCES  " + "  ·  ".join(s.get("name", "Source") for s in sources[:3]), font=small_font, fill=style["accent"])

    canvas.convert("RGB").save(poster_path, format="PNG", optimize=True)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generated/<path:filename>")
def generated_file(filename):
    return send_from_directory(GENERATED, filename)

@app.route("/api/health-explain", methods=["POST"])
def health_explain():
    payload = request.get_json(silent=True) or request.form
    requested_disease = str(payload.get("disease", "")).strip()
    disease, match_notice = resolve_disease(requested_disease)
    question = str(payload.get("question", "")).strip()
    make_image = str(payload.get("generate_image", "true")).lower() not in {"false", "0", "no"}
    if not requested_disease or len(requested_disease) > 80 or len(question) > 500:
        return jsonify({"error": "Enter a disease and a question under 500 characters."}), 400

    query = f"{disease} {question}"
    intent = question_intent(question)
    docs = retrieve(query, disease=disease, limit=6, intent=intent)

    if not docs:
        return jsonify({
            "disease": disease,
            "summary": "I could not find a relevant passage in the current starter knowledge base. Please consult a qualified health professional or official health service.",
            "sections": {},
            "sources": [],
            "image_url": None,
            "match_notice": match_notice,
            "notice": "Educational prototype; not a diagnosis or substitute for professional care."
        })

    model_answers = []
    for doc in docs:
        answer = extract_answer(question, doc)
        if answer:
            model_answers.append({"answer": answer, "document_id": doc["document_id"]})

    sections = {}
    for doc in docs:
        section = doc["section"].replace("_", " ").title()
        sections.setdefault(section, [])
        if doc["text"] not in sections[section]:
            sections[section].append(doc["text"])

    all_facts = [doc["text"] for doc in docs]
    sources = []
    seen = set()
    for doc in docs:
        if doc["source_url"] not in seen:
            seen.add(doc["source_url"])
            sources.append({"name": doc["source_name"], "url": doc["source_url"], "document_id": doc["document_id"]})

    answer_summary = (
        " ".join(dict.fromkeys(item["answer"] for item in model_answers))
        if model_answers else source_excerpt(docs[0])
    )

    image_url = None
    illustration_url = None
    image_caption = None
    image_error = None
    if make_image:
        topic = "prevention"
        if any(w in question.lower() for w in ["spread", "transmit", "cause"]):
            topic = "transmission"
        elif any(w in question.lower() for w in ["hygiene", "water", "handwash"]):
            topic = "hygiene"
        stem = re.sub(r"[^a-z0-9_-]", "", disease.replace(" ", "_"))[:50] + "_" + topic
        illustration_path = GENERATED / (stem + "_illustration.png")
        try:
            from image_generation import generate_illustration
            generate_illustration(disease, topic, all_facts, str(illustration_path))
            illustration_url = f"/generated/{illustration_path.name}"
            image_caption = "Stable Diffusion artwork with readable, source-grounded text overlaid in a varied poster layout."
        except Exception as exc:
            print("Illustration generation unavailable:", exc)
            image_error = str(exc)
            illustration_path = None
            image_caption = "Designed vector artwork; health text and source links come from retrieved passages."
        try:
            if illustration_path:
                poster_path = GENERATED / (stem + "_poster.png")
                compose_diffusion_poster(disease, topic, answer_summary, docs, sources, illustration_path, poster_path)
            else:
                poster_path = GENERATED / (stem + "_poster.svg")
                compose_poster(disease, topic, docs, sources, None, poster_path, summary=answer_summary)
            image_url = f"/generated/{poster_path.name}"
        except Exception as exc:
            print("Poster composition failed:", exc)
            image_url = illustration_url
            image_caption = "The poster could not be composed. Showing the illustration only."

    return jsonify({
        "disease": disease,
        "match_notice": match_notice,
        "summary": answer_summary,
        "model_answers": model_answers,
        "sections": sections,
        "sources": sources,
        "image_url": image_url,
        "illustration_url": illustration_url,
        "image_caption": image_caption,
        "image_error": image_error,
        "notice": (
            "Educational prototype only. Information may be incomplete; verify sources and consult a qualified healthcare professional. For urgent symptoms, seek medical care."
            + (
                " Some retrieved MedlinePlus passages were imported automatically and have not been human-reviewed."
                if any(doc.get("last_verified") == "NOT_HUMAN_REVIEWED" for doc in docs)
                else ""
            )
        )
    })

if __name__ == "__main__":
    app.run(debug=True)

