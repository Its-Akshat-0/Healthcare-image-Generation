import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data" / "medical_documents.jsonl"
required = {"document_id", "disease", "section", "text", "source_name", "source_url", "language"}
seen = set()
errors = []
count = 0

for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
        continue
    count += 1
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        errors.append(f"Line {line_no}: invalid JSON: {exc}")
        continue
    missing = required - row.keys()
    if missing:
        errors.append(f"Line {line_no}: missing fields {sorted(missing)}")
    if row.get("document_id") in seen:
        errors.append(f"Line {line_no}: duplicate document_id {row.get('document_id')}")
    seen.add(row.get("document_id"))
    if not str(row.get("source_url", "")).startswith("https://"):
        errors.append(f"Line {line_no}: source_url should be HTTPS")
    if len(str(row.get("text", "")).split()) < 8:
        errors.append(f"Line {line_no}: suspiciously short passage")

print(f"Documents checked: {count}")
if errors:
    print("Validation issues:")
    print("\n".join(errors))
    raise SystemExit(1)
print("Basic structure checks passed. Medical accuracy and freshness still require human review.")
