"""
Governance Analysis Module for Saudi Non-Profit Sector
وحدة التحليل الإحصائي والمقارن لمؤشرات الحوكمة في القطاع غير الربحي
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_charities_governance(df_charities: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive statistical analysis on Track 2 (Charities dataset):
    - Regional distribution
    - Entity type breakdown
    - Duplicate registration number diagnostics
    - Data completeness and missing value rates
    """
    total_entities = len(df_charities)

    # 1. Regional distribution
    region_counts = df_charities["region"].value_counts()
    region_pcts = (region_counts / total_entities * 100).round(2)
    df_regions = pd.DataFrame({
        "region": region_counts.index,
        "count": region_counts.values,
        "percentage": region_pcts.values
    })

    # 2. Entity types breakdown
    type_counts = df_charities["entity_type"].value_counts()
    type_pcts = (type_counts / total_entities * 100).round(2)
    df_types = pd.DataFrame({
        "entity_type": type_counts.index,
        "count": type_counts.values,
        "percentage": type_pcts.values
    })

    # 3. Cross-tabulation: Entity Type by Region
    crosstab_reg_type = pd.crosstab(df_charities["region"], df_charities["entity_type"])

    # 4. Registration number analysis
    unique_reg_ids = df_charities["registration_number"].nunique()
    duplicated_reg_ids = total_entities - unique_reg_ids

    # 5. Data completeness analysis
    completeness = {
        col: {
            "non_null_count": int(df_charities[col].notnull().sum()),
            "null_count": int(df_charities[col].isnull().sum()),
            "completeness_rate_pct": round(df_charities[col].notnull().sum() / total_entities * 100, 2)
        }
        for col in df_charities.columns
    }

    return {
        "total_entities": total_entities,
        "df_regions": df_regions,
        "df_types": df_types,
        "crosstab_reg_type": crosstab_reg_type,
        "unique_reg_ids": unique_reg_ids,
        "duplicated_reg_ids": duplicated_reg_ids,
        "completeness": completeness,
        "has_governance_scores": False,
        "governance_score_note": (
            "ملف البيانات المفتوحة الحالي لا يحتوي على درجات الحوكمة. "
            "لا يعني هذا ضعف الامتثال، وإنما يمثل قيداً بيانياً في المصدر المنشور."
        )
    }


def analyze_legal_governance_indicators(df_legal: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comparative analysis on Track 1 (Legal text corpus):
    - Topic distribution in Governance Rules vs Executive Regulations
    - Verifiable governance indicators and statutory thresholds
    """
    # 1. Topic frequency across all legal texts
    total_articles = len(df_legal)
    topic_dist = df_legal["primary_topic"].value_counts()
    topic_pcts = (topic_dist / total_articles * 100).round(2)

    # 2. Breakdown per document
    crosstab_doc_topic = pd.crosstab(df_legal["primary_topic"], df_legal["doc_name"])

    # 3. Verifiable quantitative indicators extracted from the texts
    verifiable_indicators = [
        {
            "dimension": "الرقابة المالية والمراجعة",
            "statutory_threshold": "إيرادات 8,000,000 ريال سنوياً فأكثر",
            "requirement": "إلزامية تشكيل لجنتي المراجعة والترشيحات والمكافآت بوجود أعضاء مستقلين",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المادة 21",
            "page_num": 8
        },
        {
            "dimension": "المكافآت والحوافز",
            "statutory_threshold": "درجة حوكمة لا تقل عن 85% وإيراد 8 ملايين ريال",
            "requirement": "اشتراط نسبة حوكمة 85% وخلو تقرير المراجع من تحفظات لصرف مكافآت للمجلس",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المادة 29",
            "page_num": 11
        },
        {
            "dimension": "سقوف مكافآت الاجتماعات",
            "statutory_threshold": "1,000 ريال لاجتماع المجلس و500 ريال لاجتماع اللجنة (سقف 6 سنوياً)",
            "requirement": "الحد الأقصى لمكافأة عضو المجلس 6,000 ريال وعضو اللجنة 3,000 ريال سنوياً",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المادة 30",
            "page_num": 11
        },
        {
            "dimension": "الديمقراطية المؤسسية والتصويت",
            "statutory_threshold": "سقف 49% كحد أقصى لأصوات الداعمين مجتمعين",
            "requirement": "تطبيق الأسلوب الموزون التراكمي مع منع سيطرة كبار المانحين على العمومية",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المادتان 6 و14",
            "page_num": 4
        },
        {
            "dimension": "العضو الداعم",
            "statutory_threshold": "تبرع 100,000 ريال سعودي غير مقيد",
            "requirement": "شرط التبرع غير المقيد للحصول على وزن تصويتي إضافي بحد أدنى 100 ألف",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المادة 4",
            "page_num": 3
        },
        {
            "dimension": "حفظ السجلات المالية",
            "statutory_threshold": "عشر (10) سنوات كحد أدنى",
            "requirement": "الاحتفاظ بجميع السجلات والمستندات المالية وملفات الحسابات والهويات",
            "source_doc": "اللائحة التنفيذية",
            "article_num": "المادة 118",
            "page_num": 53
        },
        {
            "dimension": "مهلة تصحيح المخالفات",
            "statutory_threshold": "ثلاثون (30) يوماً",
            "requirement": "إنذار الكيان وإمهاله مدة لا تزيد على 30 يوماً لتصحيح المخالفة أو خطة تصحيحية",
            "source_doc": "اللائحة التنفيذية",
            "article_num": "المادة 123",
            "page_num": 54
        },
        {
            "dimension": "الشفافية وتحديث البيانات",
            "statutory_threshold": "تحديث نصف سنوي إلزامي",
            "requirement": "تزويد المركز ببيانات المستفيدين والمالية والمجالس وتحديثها دورياً",
            "source_doc": "اللائحة التنفيذية",
            "article_num": "المادة 114",
            "page_num": 52
        },
        {
            "dimension": "الطعون الانتخابية",
            "statutory_threshold": "تقديم الطعن خلال 10 أيام والبت خلال 30 يوماً",
            "requirement": "مواعيد حتمية للطعن في نتائج الانتخابات أمام لجنة الطعون الدائمة",
            "source_doc": "قواعد الحوكمة",
            "article_num": "المواد 18-20",
            "page_num": 8
        },
        {
            "dimension": "استقلالية الحسابات الزكوية",
            "statutory_threshold": "فصل بنكي ومحاسبي 100%",
            "requirement": "إلزامية فتح حساب بنكي مستقل لأموال الزكاة وسجل خاص ومصارف شرعية",
            "source_doc": "اللائحة التنفيذية",
            "article_num": "المادتان 31 و61",
            "page_num": 12
        }
    ]

    df_indicators = pd.DataFrame(verifiable_indicators)

    return {
        "total_articles": total_articles,
        "topic_distribution": topic_dist,
        "topic_percentages": topic_pcts,
        "crosstab_doc_topic": crosstab_doc_topic,
        "df_indicators": df_indicators
    }


def save_analysis_outputs(
    charities_analysis: Dict[str, Any],
    legal_analysis: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Saves all analysis tables to data/outputs/ as CSV and JSON summaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Regions table
    charities_analysis["df_regions"].to_csv(
        os.path.join(output_dir, "entities_by_region.csv"), index=False, encoding="utf-8-sig"
    )

    # 2. Entity types table
    charities_analysis["df_types"].to_csv(
        os.path.join(output_dir, "entities_by_type.csv"), index=False, encoding="utf-8-sig"
    )

    # 3. Cross-tabulation Region x Type
    charities_analysis["crosstab_reg_type"].to_csv(
        os.path.join(output_dir, "crosstab_region_entity_type.csv"), encoding="utf-8-sig"
    )

    # 4. Verifiable indicators table
    legal_analysis["df_indicators"].to_csv(
        os.path.join(output_dir, "verifiable_governance_indicators.csv"), index=False, encoding="utf-8-sig"
    )

    # 5. Legal topics comparative table
    legal_analysis["crosstab_doc_topic"].to_csv(
        os.path.join(output_dir, "legal_topics_comparison.csv"), encoding="utf-8-sig"
    )

    # 6. JSON summary
    summary = {
        "total_charities": charities_analysis["total_entities"],
        "total_regions": int(len(charities_analysis["df_regions"])),
        "total_legal_articles_analyzed": legal_analysis["total_articles"],
        "top_region": charities_analysis["df_regions"].iloc[0]["region"],
        "top_region_count": int(charities_analysis["df_regions"].iloc[0]["count"]),
        "entity_types": charities_analysis["df_types"].to_dict(orient="records"),
        "key_statutory_thresholds_count": len(legal_analysis["df_indicators"])
    }
    with open(os.path.join(output_dir, "executive_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Analysis outputs successfully saved to data/outputs/!")


if __name__ == "__main__":
    from src.data_cleaning import load_and_clean_charities_data
    from src.legal_text_extraction import load_legal_corpus

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base, "data", "raw", "Charities - Dashboard 2.csv")
    df_c, _ = load_and_clean_charities_data(raw_csv)
    df_l = load_legal_corpus()

    c_res = analyze_charities_governance(df_c)
    l_res = analyze_legal_governance_indicators(df_l)

    out_dir = os.path.join(base, "data", "outputs")
    save_analysis_outputs(c_res, l_res, out_dir)
