"""
Starter importer for MedlinePlus XML.
Check current XML documentation, terms, and redistribution rules before use.
This script downloads only the main health-topic XML index; it does not guarantee
that every linked topic page is downloaded or that all text is appropriate.
"""
from pathlib import Path
import requests

URL = "https://medlineplus.gov/xml.html"
OUT = Path(__file__).resolve().parents[1] / "data" / "source_downloads"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    response = requests.get(URL, timeout=30, headers={"User-Agent": "RuralHealthVision student project"})
    response.raise_for_status()
    target = OUT / "medlineplus_xml_download_page.html"
    target.write_bytes(response.content)
    print(f"Saved source download page to {target}")
    print("Open the page, select an appropriate XML file, and review its terms before importing.")

if __name__ == "__main__":
    main()
