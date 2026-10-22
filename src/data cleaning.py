"""
Data Cleaning and Validation Module for Saudi Non-Profit Organizations Dataset
وحدة تنظيف والتحقق من بيانات المنظمات غير الربحية في المملكة العربية السعودية
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple


def load_and_clean_charities_data(
    filepath: str,
    output_filepath: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reads the open data CSV of Saudi charities, cleans strings, handles headers,
    examines missing values and duplicates, and verifies column completeness.

    Parameters:
        filepath: Path to raw CSV file.
        output_filepath: Optional path to save cleaned CSV.

    Returns:
        Tuple of (cleaned_df, audit_summary_dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    # Read file skipping metadata header row if present
    # Check if first line contains 'Charities - Dashboard'
    with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
        first_line = f.readline()
        skiprows = 1 if "Charities" in first_line else 0

    df_raw = pd.read_csv(filepath, skiprows=skiprows, encoding="utf-8")

    # Standardize column names
    col_mapping = {
        df_raw.columns[0]: "registration_number",
        df_raw.columns[1]: "organization_name",
        df_raw.columns[2]: "entity_type",
        df_raw.columns[3]: "region"
    }
    df = df_raw.rename(columns=col_mapping)
    df = df[["registration_number", "organization_name", "entity_type", "region"]].copy()

    total_records = len(df)

    # 1. Clean whitespace and trim text
    for col in ["organization_name", "entity_type", "region"]:
        df[col] = df[col].astype(str).str.strip()

    # 2. Convert registration number to numeric safely
    df["registration_number"] = pd.to_numeric(df["registration_number"], errors="coerce")

    # 3. Check missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_rates = (df.isnull().sum() / total_records * 100).round(2).to_dict()

    # 4. Check for duplicates
    exact_duplicates = int(df.duplicated().sum())
    reg_duplicates = int(df["registration_number"].duplicated().sum())

    # Check whether governance scores exist
    has_governance_score = "governance_score" in [c.lower() for c in df_raw.columns] or any(
        "درجة" in str(c) or "حوك" in str(c) for c in df_raw.columns
    )

    # 5. Entity types distribution
    entity_dist = df["entity_type"].value_counts().to_dict()

    # 6. Regions distribution
    region_dist = df["region"].value_counts().to_dict()

    audit_summary = {
        "total_records": total_records,
        "clean_records": len(df.dropna()),
        "columns_present": list(df.columns),
        "missing_counts": missing_counts,
        "missing_rates_percent": missing_rates,
        "exact_duplicates": exact_duplicates,
        "duplicate_registration_ids": reg_duplicates,
        "has_governance_scores": has_governance_score,
        "governance_score_status": (
            "درجات الحوكمة غير متوفرة في ملف البيانات المفتوحة الحالي "
            "(لا يعتبر غيابها دليلاً على ضعف الحوكمة أو عدم الامتثال)"
        ),
        "entity_type_distribution": entity_dist,
        "region_distribution": region_dist,
        "unique_regions_count": int(df["region"].nunique()),
        "unique_entities_count": int(df["organization_name"].nunique())
    }

    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_csv(output_filepath, index=False, encoding="utf-8-sig")

    return df, audit_summary


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_p = os.path.join(base, "data", "raw", "Charities - Dashboard 2.csv")
    out_p = os.path.join(base, "data", "processed", "charities_cleaned.csv")
    df_clean, summary = load_and_clean_charities_data(raw_p, out_p)
    print("Data cleaning executed successfully!")
    print(f"Total: {summary['total_records']}, Regions: {summary['unique_regions_count']}")
