"""End-to-end pipeline:  python run_pipeline.py
Inputs : data/raw/ (see README)   Intermediate: data/processed/ (git-ignored)
Outputs: data/outputs/ (tables, metrics.json)   Figures: reports/figures/"""
import json

import pandas as pd

from src import config
from src import data_cleaning as dc
from src import governance_analysis as ga
from src import legal_text_extraction as lte
from src import legal_topic_classification as ltc
from src import visualizations as viz
from src import audit


def ensure_dirs():
    for d in [config.DATA_PROCESSED, config.OUTPUTS_DIR, config.FIGURES_DIR, config.OCR_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_charities(metrics: dict):
    raw = dc.load_raw(config.CHARITIES_CSV)
    clean = dc.clean_charities(raw)
    clean.to_csv(config.DATA_PROCESSED / "charities_clean.csv", index=False, encoding="utf-8-sig")  # git-ignored

    comp = dc.completeness(raw)
    checks = dc.quality_checks(clean)
    names = dc.name_stats(clean)
    by_type, by_region = dc.by_type(clean), dc.by_region(clean)
    cross = dc.region_by_type(clean)
    out = config.OUTPUTS_DIR
    comp.to_csv(out / "charities_completeness.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(out / "charities_quality_checks.csv", index=False, encoding="utf-8-sig")
    names.to_csv(out / "charities_name_stats.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(out / "charities_by_type.csv", index=False, encoding="utf-8-sig")
    by_region.to_csv(out / "charities_by_region.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(out / "charities_region_by_type.csv", index=False, encoding="utf-8-sig")

    status = ga.governance_scores_status(config.SCORES_CSV, charities_has_score_column=False)
    status.to_csv(out / "governance_scores_status.csv", index=False, encoding="utf-8-sig")

    f = config.FIGURES_DIR
    viz.hbar(by_region["region"].str.replace("منطقة ", "", regex=False), by_region["total"],
             "توزيع الجهات حسب المنطقة", "عدد الجهات", f / "charities_by_region.png")
    viz.hbar(by_type["entity_type"], by_type["total"], "توزيع الجهات حسب نوع الكيان", "عدد الجهات",
             f / "charities_by_type.png", color="#c9742f")
    d = by_region.assign(region=by_region["region"].str.replace("منطقة ", "", regex=False))
    viz.stacked_hbar(d["region"], {"جمعيات أهلية": d["associations"], "مؤسسات أهلية": d["foundations"]},
                     "الجهات حسب المنطقة ونوع الكيان", "عدد الجهات", f / "charities_region_by_type.png")
    viz.completeness(comp, f / "charities_completeness.png")

    metrics["charities"] = {
        "rows": int(len(clean)),
        "entity_counts": by_type.set_index("entity_type")["total"].to_dict(),
        "top_regions": by_region.head(3).set_index("region")["total"].to_dict(),
        "reg_number_shared_rows": int(clean["reg_number_shared"].sum()),
        "reg_number_unique_values": int(clean["reg_number"].nunique()),
        "truncated_names": int(clean["name_truncated"].sum()),
        "exact_duplicate_group_rows": int(clean["exact_duplicate_group"].sum()),
        "governance_score_columns": 0,
        "governance_score_analysis": "لم يُنفَّذ: الملف لا يحتوي على درجات حوكمة، ولا يوجد ملف درجات منفصل.",
    }


def run_legal(metrics: dict):
    out = config.OUTPUTS_DIR
    pages = {"reg": lte.ocr_document(config.REG_PDF, "reg"), "gov": lte.ocr_document(config.GOV_PDF, "gov")}
    frames, rejected, quality, seq = [], [], [], {}
    for key, page_texts in pages.items():
        arts, rej = lte.parse_articles(page_texts, key)
        expected = config.EXPECTED_ARTICLES[key]
        seq[key] = lte.sequence_report(arts, expected)
        frames.append(arts)
        rejected.append(rej)
        full = " ".join(page_texts)
        quality.append({"doc_key": key, "pages": len(page_texts), "articles_found": seq[key]["found"],
                        "max_article_no": int(arts["article_no"].max()), "missing_article_numbers": len(seq[key]["missing"]),
                        "rejected_header_candidates": len(rej), "articles_matched_by_ocr_fuzzy": int(arts["header_fuzzy"].sum()),
                        "arabic_letter_share_pct": lte.arabic_share(full)})

    arts = pd.concat(frames, ignore_index=True)
    arts = ltc.tfidf_topic_check(ltc.classify_articles(arts))

    rej_all = pd.concat(rejected, ignore_index=True)
    rej_all.to_csv(out / "rejected_header_candidates.csv", index=False, encoding="utf-8-sig")
    review = audit.classification_review_sample(arts)
    review.to_csv(config.DATA_PROCESSED / "classification_review_sample.csv", index=False, encoding="utf-8-sig")  # includes source words
    review.drop(columns=["text_first_20_words"]).to_csv(out / "classification_review_sample.csv", index=False, encoding="utf-8-sig")
    art_audit = audit.article_audit(arts)
    art_audit.to_csv(out / "extraction_audit.csv", index=False, encoding="utf-8-sig")
    art_audit[art_audit["needs_human_review"]].to_csv(out / "human_review_queue.csv", index=False, encoding="utf-8-sig")
    seq_sum = audit.sequence_summary(arts, rej_all, config.EXPECTED_ARTICLES)
    seq_sum["flagged_for_review"] = seq_sum["doc_key"].map(art_audit.groupby("doc_key")["needs_human_review"].sum()).fillna(0).astype(int)
    seq_sum.to_csv(out / "extraction_sequence_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality).to_csv(out / "ocr_quality.csv", index=False, encoding="utf-8-sig")

    index = arts[["doc_key", "article_no", "article_label", "page_start", "page_end", "topics",
                  "primary_topic", "topic_hits", "tfidf_topic", "tfidf_score", "agrees_with_rules"]].copy()
    index["topics"] = index["topics"].map(lambda t: "; ".join(t))
    index["topic_hits"] = index["topic_hits"].map(lambda h: json.dumps(h, ensure_ascii=False))
    index.to_csv(out / "legal_articles_index.csv", index=False, encoding="utf-8-sig")
    arts.drop(columns=["text_raw"]).to_csv(config.DATA_PROCESSED / "legal_articles_full.csv", index=False,
                                           encoding="utf-8-sig")  # git-ignored, contains article text

    req = ltc.extract_requirements(arts)
    req.insert(0, "indicator_id", [f"{d}-{i:04d}" for i, d in enumerate(req["doc_key"], start=1)])
    # Snippets are fragments of the source text: kept in git-ignored processed/, not in published outputs.
    req.to_csv(config.DATA_PROCESSED / "requirement_snippets.csv", index=False, encoding="utf-8-sig")
    req.drop(columns=["snippet_15_words"]).to_csv(out / "requirement_indicators.csv", index=False, encoding="utf-8-sig")

    counts = (arts[["doc_key", "topics"]].explode("topics").dropna().groupby(["topics", "doc_key"]).size()
              .unstack(fill_value=0).reindex(columns=["gov", "reg"], fill_value=0))
    topic_table = []
    for topic, row in counts.iterrows():
        topic_table.append({"topic": topic, "gov_articles": int(row["gov"]), "reg_articles": int(row["reg"]),
                            "gov_share_pct": round(100 * row["gov"] / seq["gov"]["found"], 1),
                            "reg_share_pct": round(100 * row["reg"] / seq["reg"]["found"], 1)})
    topic_table = pd.DataFrame(topic_table).sort_values("gov_articles", ascending=False)
    topic_table.to_csv(out / "topic_counts_by_document.csv", index=False, encoding="utf-8-sig")

    matches, sens = ga.governance_to_regulation(arts)
    matches.to_csv(out / "governance_to_regulation_matches.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out / "similarity_sensitivity.csv", index=False, encoding="utf-8-sig")
    terms = ga.top_terms(arts)
    terms.to_csv(out / "top_terms_by_document.csv", index=False, encoding="utf-8-sig")

    f = config.FIGURES_DIR
    viz.grouped(topic_table["topic"], {"قواعد الحوكمة": topic_table["gov_articles"],
                                      "اللائحة التنفيذية": topic_table["reg_articles"]},
                "عدد المواد لكل موضوع في كل وثيقة", "عدد المواد", f / "legal_topics_by_document.png")
    rt = req.groupby(["doc_key", "requirement_type"]).size().unstack(fill_value=0)
    viz.grouped(rt.columns.tolist(), {config.DOC_LABELS[d]: rt.loc[d].tolist() for d in rt.index},
                "أنواع المتطلبات في كل وثيقة", "عدد الجمل", f / "legal_requirement_types.png")
    viz.similarity_hist(matches["similarity"], 0.25, f / "governance_regulation_similarity.png")

    metrics["legal"] = {
        "sequence_checks": seq,
        "ocr_quality": quality,
        "articles_total": int(len(arts)),
        "articles_with_keyword_topic": int(arts["primary_topic"].notna().sum()),
        "rule_vs_tfidf_agreement_pct": round(100 * arts.loc[arts["primary_topic"].notna(), "agrees_with_rules"].mean(), 1),
        "requirements_total": int(len(req)),
        "requirements_by_type": req.groupby(["doc_key", "requirement_type"]).size().reset_index(name="n")
                                  .to_dict(orient="records"),
        "topic_table": topic_table.to_dict(orient="records"),
        "matches_above_threshold_0_25": int(matches["above_threshold"].sum()),
        "governance_articles_compared": int(len(matches)),
        "matches_with_shared_topic": int(matches["shared_topic"].sum()),
        "topics_absent_in_one_document": sorted(
            set(topic_table.loc[topic_table["gov_articles"] == 0, "topic"]) |
            set(topic_table.loc[topic_table["reg_articles"] == 0, "topic"])),
    }


def main():
    ensure_dirs()
    metrics = {}
    run_charities(metrics)
    run_legal(metrics)
    (config.OUTPUTS_DIR / "run_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
