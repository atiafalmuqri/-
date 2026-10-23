"""Central configuration. Paths are relative to the project root so the same
code runs locally and in Google Colab (override with NONPROFIT_PROJECT_ROOT)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NONPROFIT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "data" / "outputs"
OCR_DIR = DATA_PROCESSED / "ocr_pages"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

# Place the original files in data/raw/ under these names (see README). Never committed.
REG_PDF = DATA_RAW / "regulation_executive.pdf"   # اللائحة التنفيذية لنظام الجمعيات والمؤسسات الأهلية
GOV_PDF = DATA_RAW / "governance_rules.pdf"       # قواعد حوكمة الجمعيات والمؤسسات الأهلية
CHARITIES_CSV = DATA_RAW / "Charities_-_Dashboard_2.csv"
SCORES_CSV = DATA_RAW / "governance_scores.csv"   # optional, only if such a file is provided

DOC_LABELS = {"reg": "اللائحة التنفيذية", "gov": "قواعد الحوكمة"}
EXPECTED_ARTICLES = {"reg": 129, "gov": 47}
OCR_DPI = 250
OCR_LANG = "ara"
OCR_PSM = "6"
EXPECTED_PAGES = {"reg": 56, "gov": 17}
