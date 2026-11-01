"""Basic retrieval smoke test; not a full scientific evaluation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import retrieve

tests = [
    ("dengue", "How can dengue be prevented?", "prevention"),
    ("malaria", "How is malaria transmitted?", "overview"),
    ("rabies", "What should I do after an animal bite?", "urgent_action"),
]
for disease, query, expected_section in tests:
    docs = retrieve(query, disease=disease, limit=5)
    found = any(d["section"] == expected_section for d in docs)
    print(f"{disease}: expected section {expected_section!r} retrieved={found}")
