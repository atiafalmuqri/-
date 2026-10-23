"""Extract articles (المواد) from the two governance PDFs.

PDF -> page images (PyMuPDF) -> OCR (Tesseract, Arabic) -> cached page text
-> article segmentation with strict sequence checking -> page references.
OCR-tolerant matching is applied only to heading words, never to article text."""
import difflib
import re
import subprocess
from pathlib import Path

import pandas as pd

from src import config

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def nz(text) -> str:
    """Normalise Arabic for comparison: strip diacritics/tatweel, unify alef, ya, ta marbuta,
    map Arabic-Indic digits to ASCII, collapse spaces. The original text is kept separately."""
    if not isinstance(text, str):
        return ""
    t = _DIACRITICS.sub("", text).translate(_ARABIC_DIGITS)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"\s+", " ", t).strip()


normalize_ar = nz
ARTICLE_WORD = nz("المادة")

# ---------- ordinals ------------------------------------------------------
_UNITS = {}
for _m, _f, _v in [("الاول", "الاولي", 1), ("الثاني", "الثانيه", 2), ("الثالث", "الثالثه", 3),
                   ("الرابع", "الرابعه", 4), ("الخامس", "الخامسه", 5), ("السادس", "السادسه", 6),
                   ("السابع", "السابعه", 7), ("الثامن", "الثامنه", 8), ("التاسع", "التاسعه", 9),
                   ("العاشر", "العاشره", 10), ("الحادي", "الحاديه", 1)]:
    _UNITS[nz(_m)] = _v
    _UNITS[nz(_f)] = _v
_TENS = {nz(w): v for w, v in [("العشرون", 20), ("العشرين", 20), ("الثلاثون", 30), ("الاربعون", 40),
                                ("الخمسون", 50), ("الستون", 60), ("السبعون", 70),
                                ("الثمانون", 80), ("التسعون", 90)]}
_HUNDRED = {nz("المائه"), nz("مائه")}
_TEEN = {nz("عشره"), nz("عشر")}
_AFTER = nz("بعد")
_VOCAB = list(set(_UNITS) | set(_TENS) | _HUNDRED | _TEEN | {_AFTER})


def _canonical(tok: str):
    """Return (known_word, was_fuzzy). OCR confuses letters in ordinals: ت/ث ('التالتة'),
    ي/ث ('الثاليه' for 'الثالثة' after ئ->ي normalisation), and drops letters ('الأريعون').
    Exact matches over all letter variants are tried before any approximate match."""
    if tok in _VOCAB:
        return tok, False
    core = tok[1:] if tok.startswith("و") and len(tok) > 2 else tok
    if core in _VOCAB:
        return core, False
    variants = list(dict.fromkeys([core, core.replace("ت", "ث"), core.replace("ي", "ث"),
                                   core.replace("ت", "ث").replace("ي", "ث")]))
    for v in variants:
        if v in _VOCAB:
            return v, True
    for v in variants:
        cand = difflib.get_close_matches(v, _VOCAB, n=1, cutoff=0.8)
        if cand:
            return cand[0], True
    return None, False


def parse_ordinal_detail(label: str):
    """Return (number, used_fuzzy_match). Number is None when the label is not an ordinal."""
    toks = [t for t in nz(label).replace("(", " ").replace(")", " ").split() if t]
    value, last_unit, seen, fuzzy = 0, None, False, False
    for raw in toks:
        tok, was_fuzzy = _canonical(raw)
        if tok is None:
            return None, False
        fuzzy |= was_fuzzy
        if tok in _UNITS:
            value += _UNITS[tok]
            last_unit = _UNITS[tok]
            seen = True
        elif tok in _TENS:
            value += _TENS[tok]
            seen = True
        elif tok in _TEEN and last_unit is not None and last_unit < 10:
            value += 10          # الحادية عشرة = 11
            last_unit = None
        elif tok in _HUNDRED:
            value += 100
            seen = True
        elif tok == _AFTER:
            continue
        else:
            return None, False
    return (value if seen and value > 0 else None), fuzzy


def parse_ordinal(label: str):
    """Arabic ordinal -> integer, e.g. 'التاسعة والعشرون بعد المائة' -> 129. None if not parsable."""
    return parse_ordinal_detail(label)[0]


# ---------- OCR -------------------------------------------------------------
def ocr_document(pdf_path, doc_key, cache_dir=config.OCR_DIR, dpi=config.OCR_DPI,
                 lang=config.OCR_LANG, psm=config.OCR_PSM) -> list:
    """OCR each page once. Cached as <cache_dir>/<doc_key>_NNN.txt and reused on later runs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(pdf_path)
    cached = sorted(cache_dir.glob(f"{doc_key}_*.txt"))
    if cached and len(cached) == config.EXPECTED_PAGES.get(doc_key, len(cached)):
        return [p.read_text(encoding="utf-8") for p in cached]
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    import fitz  # PyMuPDF
    texts = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            out_txt = cache_dir / f"{doc_key}_{i:03d}.txt"
            if not out_txt.exists():
                png = cache_dir / f"_tmp_{doc_key}.png"
                page.get_pixmap(dpi=dpi).save(png)
                try:
                    subprocess.run(["tesseract", str(png), str(out_txt.with_suffix("")), "-l", lang,
                                    "--psm", psm], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError as exc:
                    raise RuntimeError("Tesseract not found. Install it with the Arabic ('ara') language data.") from exc
                finally:
                    png.unlink(missing_ok=True)
            texts.append(out_txt.read_text(encoding="utf-8"))
    return texts


# ---------- segmentation ----------------------------------------------------
_RUNNING = {nz(x) for x in [
    "قواعد حوكمة الجمعيات والمؤسسات الأهلية",
    "الالئحة التنفيذية لنظام الجمعيات والمؤسسات الأهلية المحدّثة",
    "الائحة التنفيذية لنظام الجمعيات والمؤسسات الأهلية المحدثة"]}
_FOOTER = re.compile(r"^صفحه\s*\d+\s*من\s*\d+$")
_CHAPTER = re.compile(r"^(الباب|الفصل)\s")
_RUNNING_TITLE = re.compile(r"التنفيذيه.*المحدثه|الائحه التنفيذيه لنظام")
_PAGE_MARK = re.compile(r"صفحه.*من|من\s*\d+$")
_HEADER = re.compile(r"^" + re.escape(ARTICLE_WORD) + r"\s*[\(\[]?\s*([^:\n\(\)\[\]]{2,50}?)\s*[\)\]]?\s*:\s*(.*)$")


def _content_lines(page_text: str):
    for line in page_text.splitlines():
        n = nz(line)
        if not n or _FOOTER.match(n) or n in _RUNNING:
            continue
        if _CHAPTER.match(n) and len(n) < 60:   # chapter headings (الباب/الفصل) are not article text
            continue
        if _RUNNING_TITLE.search(n) or (len(n) < 40 and _PAGE_MARK.search(n)):
            continue           # running title and page footers (e.g. "صفحه 47 من 56") seen in regulation OCR
        if len(re.sub(r"[\W_]", "", n)) < 3:
            continue           # stray OCR fragments such as "“]."
        yield line.strip(), n


def parse_articles(pages, doc_key: str):
    """Return (articles_df, rejected_df). A heading is accepted only when its number is the
    next expected article; out-of-sequence headings are logged and kept as body text."""
    articles, rejected = [], []
    current, expected = None, 1
    for page_no, page_text in enumerate(pages, start=1):
        for raw, n in _content_lines(page_text):
            m = _HEADER.match(n)
            value, fuzzy = parse_ordinal_detail(m.group(1)) if m else (None, False)
            if value is not None and value == expected:
                if current:
                    articles.append(current)
                current = {"doc_key": doc_key, "article_no": value, "article_label": f"المادة {m.group(1).strip()}",
                           "page_start": page_no, "page_end": page_no, "header_fuzzy": fuzzy,
                           "text_parts": [m.group(2).strip()] if m.group(2).strip() else [], "raw_parts": []}
                expected += 1
                continue
            if value is not None:
                rejected.append({"doc_key": doc_key, "page": page_no, "parsed": value, "expected": expected})
            if current is not None:
                current["text_parts"].append(n)
                current["raw_parts"].append(raw)
                current["page_end"] = page_no
    if current:
        articles.append(current)
    rows = [{**{k: v for k, v in a.items() if k not in ("text_parts", "raw_parts")},
             "text": "\n".join(a["text_parts"]), "text_raw": "\n".join(a["raw_parts"])} for a in articles]
    return pd.DataFrame(rows), pd.DataFrame(rejected, columns=["doc_key", "page", "parsed", "expected"])


def sequence_report(arts: pd.DataFrame, expected: int) -> dict:
    nums = arts["article_no"].tolist() if len(arts) else []
    found = set(nums)
    return {"expected": expected, "found": len(found),
            "missing": sorted(set(range(1, expected + 1)) - found),
            "is_monotonic": nums == sorted(nums), "duplicates": len(nums) - len(found)}


def arabic_share(text: str) -> float:
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return 0.0
    arabic = re.findall(r"[\u0621-\u064A]", text)
    return round(100 * len(arabic) / len(letters), 2)
