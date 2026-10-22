"""
Legal Text Extraction and Arabic Text Normalization Module
وحدة استخراج ومعالجة وتنظيف النصوص القانونية العربية
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional


def normalize_arabic_text(text: str) -> str:
    """
    Cleans and normalizes Arabic text while preserving core legal terminology.
    - Unifies Alef variants (أ، إ، آ -> ا)
    - Unifies Yaa and Alef Maksura (ى -> ي)
    - Normalizes Taa Marbuta (ة -> ه) for flexible fuzzy search (optional)
    - Strips Kashida / Tatweel (_)
    - Strips Arabic Tashkeel (diacritics)
    - Normalizes multiple spaces and punctuation
    """
    if not isinstance(text, str):
        return ""

    # Strip Tashkeel (harakat)
    tashkeel_regex = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel_regex, '', text)

    # Strip Tatweel
    text = re.sub(r'\u0640', '', text)

    # Unify Alef forms
    text = re.sub(r'[إأآا]', 'ا', text)

    # Unify Alef Maksura
    text = re.sub(r'ى', 'ي', text)

    # Remove non-alphanumeric punctuation except standard legal structure
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_legal_corpus(corpus_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the structured legal dataset containing extracted articles, page numbers,
    chapters, and legal requirements from Governance Rules and Executive Regulations.
    """
    if corpus_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus_path = os.path.join(base_dir, "data", "processed", "combined_legal_articles.csv")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Legal corpus file not found: {corpus_path}")

    df = pd.read_csv(corpus_path, encoding="utf-8-sig")
    df["normalized_text"] = df["text"].apply(normalize_arabic_text)
    return df


def search_legal_articles(query: str, df_corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Verifiable search across legal articles for specific governance keywords or phrases.
    """
    norm_query = normalize_arabic_text(query)
    mask = df_corpus["normalized_text"].str.contains(norm_query, case=False, na=False)
    return df_corpus[mask][["doc_name", "article_num", "page_num", "article_title", "key_metric", "text"]]


if __name__ == "__main__":
    df = load_legal_corpus()
    print(f"Loaded legal corpus with {len(df)} articles across documents.")
    search_res = search_legal_articles("المراجعة", df)
    print(f"Found {len(search_res)} articles referencing 'المراجعة'.")
