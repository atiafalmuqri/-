"""
Master Pipeline Runner for Saudi Nonprofit Governance Analysis
مشغل خط المعالجة والتحليل الكامل للمشروع
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_cleaning import load_and_clean_charities_data
from src.legal_text_extraction import load_legal_corpus
from src.legal_topic_classification import LegalTopicClassifier
from src.governance_analysis import (
    analyze_charities_governance,
    analyze_legal_governance_indicators,
    save_analysis_outputs
)


def main():
    print("=" * 60)
    print("Starting Saudi Nonprofit Governance Analysis Pipeline...")
    print("=" * 60)

    raw_csv = os.path.join(project_root, "data", "raw", "Charities - Dashboard 2.csv")
    cleaned_csv = os.path.join(project_root, "data", "processed", "charities_cleaned.csv")
    legal_csv = os.path.join(project_root, "data", "processed", "combined_legal_articles.csv")
    output_dir = os.path.join(project_root, "data", "outputs")

    # 1. Clean charities data
    print("\n[Step 1/4] Cleaning and auditing charities dataset...")
    df_charities, audit_summary = load_and_clean_charities_data(raw_csv, cleaned_csv)
    print(f"  -> Total records: {audit_summary['total_records']}")
    print(f"  -> Unique regions: {audit_summary['unique_regions_count']}")
    print(f"  -> Entity types: {audit_summary['entity_type_distribution']}")

    # 2. Load legal corpus
    print("\n[Step 2/4] Loading and normalizing legal texts corpus...")
    df_legal = load_legal_corpus(legal_csv)
    print(f"  -> Total legal articles: {len(df_legal)}")

    # 3. AI / NLP Topic Classification
    print("\n[Step 3/4] Running AI / NLP Topic Classification (TF-IDF + Cosine Similarity)...")
    classifier = LegalTopicClassifier()
    df_classified = classifier.classify_dataframe(df_legal)
    classified_out = os.path.join(project_root, "data", "processed", "legal_articles_ai_classified.csv")
    df_classified.to_csv(classified_out, index=False, encoding="utf-8-sig")
    print(f"  -> AI classification complete! Saved to {classified_out}")

    # 4. Statistical & Governance Analysis
    print("\n[Step 4/4] Computing governance statistics and verifiable indicators...")
    charities_analysis = analyze_charities_governance(df_charities)
    legal_analysis = analyze_legal_governance_indicators(df_legal)
    save_analysis_outputs(charities_analysis, legal_analysis, output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
