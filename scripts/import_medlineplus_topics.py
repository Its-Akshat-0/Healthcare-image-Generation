"""Refresh the local health-topic corpus from the official MedlinePlus XML.

The importer preserves the project's existing manually curated records. It
replaces only records previously imported from MedlinePlus. Section-specific
passages are copied from source summaries when a summary explicitly discusses
causes, treatment, or prevention; absent sections are omitted.
"""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "medical_documents.jsonl"
INDEX_URL = "https://medlineplus.gov/xml.html"
USER_AGENT = "RuralHealthVision educational prototype (MedlinePlus XML importer)"
MIN_SUMMARY_WORDS = 35
MAX_SECTION_SENTENCES = 5


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def plain_text(markup):
    parser = _TextExtractor()
    parser.feed(markup or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def fetch(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def latest_zip_url(index_html):
    # The link is published by MedlinePlus and includes the generated date.
    matches = re.findall(
        r'href=["\']([^"\']*mplus_topics_compressed_\d{4}-\d{2}-\d{2}\.zip)["\']',
        index_html,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise RuntimeError("Could not find the latest compressed topic XML link on MedlinePlus.")
    return urljoin(INDEX_URL, matches[0])


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.split()) >= 5]


def _records_from_topic(topic):
    if topic.get("language", "").lower() != "english":
        return []
    title = (topic.get("title") or "").strip()
    url = (topic.get("url") or "").strip()
    topic_id = (topic.get("id") or "").strip()
    if not title or not url.startswith("https://") or not topic_id:
        return []

    summary_node = topic.find("full-summary")
    # MedlinePlus stores HTML inside the XML summary. itertext() recovers that
    # embedded markup so HTMLParser can turn headings, links, and lists into
    # readable plain text instead of displaying literal "<p>" and "<a>" tags.
    summary = plain_text(
        " ".join(summary_node.itertext()) if summary_node is not None else ""
    )
    if len(summary.split()) < MIN_SUMMARY_WORDS:
        return []

    source_name = "MedlinePlus, U.S. National Library of Medicine"
    common = {
        "disease": title.lower(),
        "source_name": source_name,
        "source_url": url,
        "language": "en",
        "last_verified": "NOT_HUMAN_REVIEWED",
    }
    records = [{
        **common,
        "document_id": f"medlineplus_{topic_id}_overview",
        "section": "overview",
        "text": summary,
    }]

    # Only create subject-specific retrieval passages when the source summary
    # explicitly contains relevant wording. Keep the original sentences intact.
    patterns = {
        "causes": re.compile(r"\b(caus(?:e|es|ed|ing)|risk factor|due to|result(?:s|ing)? from)\b", re.I),
        "treatment": re.compile(r"\b(treat(?:ment|ed|ing)?|cure[sd]?|medicin(?:e|es)|therapy|manage(?:ment|d))\b", re.I),
        "prevention": re.compile(r"\b(prevent(?:ion|s|ed|ing)?|protect(?:ion|s|ed|ing)?|lower your risk|reduce the risk)\b", re.I),
    }
    sentences = _sentences(summary)
    for section, pattern in patterns.items():
        selected = [sentence for sentence in sentences if pattern.search(sentence)]
        if selected:
            records.append({
                **common,
                "document_id": f"medlineplus_{topic_id}_{section}",
                "section": section,
                "text": " ".join(selected[:MAX_SECTION_SENTENCES]),
            })
    return records


def main():
    print("Reading the latest MedlinePlus XML download page...")
    index = fetch(INDEX_URL).decode("utf-8", errors="replace")
    zip_url = latest_zip_url(index)
    print(f"Downloading {zip_url}")
    archive_bytes = fetch(zip_url)

    imported = []
    with tempfile.TemporaryDirectory(prefix="rhv_medlineplus_") as temp_dir:
        archive_path = Path(temp_dir) / "topics.zip"
        archive_path.write_bytes(archive_bytes)
        with ZipFile(archive_path) as archive:
            xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not xml_names:
                raise RuntimeError("The MedlinePlus ZIP did not contain an XML file.")
            # The compressed topic archive normally contains one topic XML.
            xml_bytes = archive.read(xml_names[0])
    root = ET.fromstring(xml_bytes)
    generated = root.get("dategenerated") or root.get("date-generated") or ""
    for topic in root.findall(".//health-topic"):
        rows = _records_from_topic(topic)
        if generated:
            for row in rows:
                row["source_updated"] = generated
        imported.extend(rows)

    if len(imported) < 100:
        raise RuntimeError(
            f"Only {len(imported)} source records were prepared; leaving the existing dataset untouched."
        )

    existing = []
    if DATA_PATH.exists():
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("document_id", "").startswith("medlineplus_"):
                    existing.append(row)

    combined = existing + imported
    temp_path = DATA_PATH.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in combined:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(DATA_PATH)

    topics = sum(row["section"] == "overview" for row in imported)
    print(f"Imported {topics} English MedlinePlus topics ({len(imported)} total passages).")
    print(f"Preserved {len(existing)} existing project records.")
    print(f"Updated {DATA_PATH}")
    print("Restart Flask to reload the dataset. Review the imported passages before presenting them as clinical guidance.")


if __name__ == "__main__":
    main()
