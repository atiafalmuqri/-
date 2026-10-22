"""
Unit Tests for Saudi Nonprofit Governance Analysis Pipeline
اختبارات الوحدة للتأكد من صحة دوال المعالجة والتحليل والتصنيف وسلامة المخرجات
"""

import os
import unittest
import pandas as pd

from src.data_cleaning import load_and_clean_charities_data
from src.legal_text_extraction import normalize_arabic_text, load_legal_corpus, search_legal_articles
from src.legal_topic_classification import LegalTopicClassifier, GOVERNANCE_TOPIC_PROFILES
from src.governance_analysis import analyze_charities_governance, analyze_legal_governance_indicators


class TestGovernancePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.raw_csv = os.path.join(cls.base_dir, "data", "raw", "Charities - Dashboard 2.csv")
        cls.processed_csv = os.path.join(cls.base_dir, "data", "processed", "charities_cleaned.csv")
        cls.legal_corpus_csv = os.path.join(cls.base_dir, "data", "processed", "combined_legal_articles.csv")

    def test_01_arabic_normalization(self):
        """Test Arabic text normalization cleans diacritics, tatweel and alefs."""
        raw_text = "الْإِفْصَاحُ وَالشَّفَافِيَّةُ ـــــ لِلْجَمْعِيَّةِ"
        normalized = normalize_arabic_text(raw_text)
        self.assertNotIn("ْ", normalized)
        self.assertNotIn("َ", normalized)
        self.assertNotIn("ـ", normalized)
        self.assertTrue(normalized.startswith("الافصاح"))

    def test_02_charities_data_cleaning(self):
        """Test charities dataset loader and audit summary."""
        df_clean, summary = load_and_clean_charities_data(self.raw_csv)
        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertGreater(len(df_clean), 2500)
        self.assertIn("registration_number", df_clean.columns)
        self.assertIn("organization_name", df_clean.columns)
        self.assertIn("entity_type", df_clean.columns)
        self.assertIn("region", df_clean.columns)
        self.assertEqual(summary["unique_regions_count"], 13)
        self.assertFalse(summary["has_governance_scores"])

    def test_03_legal_corpus_loading(self):
        """Test legal corpus loading and structure."""
        df_legal = load_legal_corpus(self.legal_corpus_csv)
        self.assertIsInstance(df_legal, pd.DataFrame)
        self.assertGreaterEqual(len(df_legal), 70)
        self.assertIn("article_num", df_legal.columns)
        self.assertIn("page_num", df_legal.columns)
        self.assertIn("text", df_legal.columns)
        self.assertIn("key_metric", df_legal.columns)

    def test_04_ai_topic_classifier(self):
        """Test AI NLP Topic Classifier returns valid topic and confidence score."""
        clf = LegalTopicClassifier()
        sample_text = "يجب على لجنة المراجعة تقويم أنظمة الرقابة الداخلية والنظام المحاسبي وفحص المستندات"
        predicted_topic, score, keywords = clf.classify_text(sample_text)
        self.assertIn(predicted_topic, GOVERNANCE_TOPIC_PROFILES.keys())
        self.assertEqual(predicted_topic, "اللجان والرقابة الداخلية والمراجعة")
        self.assertGreater(score, 0.0)
        self.assertIsInstance(keywords, list)

    def test_05_governance_analysis_indicators(self):
        """Test verifiable governance indicators extraction."""
        df_legal = load_legal_corpus(self.legal_corpus_csv)
        legal_res = analyze_legal_governance_indicators(df_legal)
        df_ind = legal_res["df_indicators"]
        self.assertGreaterEqual(len(df_ind), 8)
        # Check that 8M threshold indicator is present
        has_8m = any("8,000,000" in str(row["statutory_threshold"]) for _, row in df_ind.iterrows())
        self.assertTrue(has_8m)

    def test_06_search_legal_articles(self):
        """Test verifiable search in legal articles."""
        df_legal = load_legal_corpus(self.legal_corpus_csv)
        results = search_legal_articles("تعارض المصالح", df_legal)
        self.assertGreater(len(results), 0)
        for _, row in results.iterrows():
            self.assertIsNotNone(row["article_num"])
            self.assertIsNotNone(row["page_num"])


if __name__ == "__main__":
    unittest.main()
