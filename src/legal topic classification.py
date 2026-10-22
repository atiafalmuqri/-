"""
AI / NLP Legal Topic Classification Module
وحدة تصنيف النصوص القانونية واستخراج الموضوعات بالذكاء الاصطناعي (TF-IDF والتشابه الدلالي)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.legal_text_extraction import normalize_arabic_text


# Predefined Governance Topic Profiles (Standard Taxonomy)
GOVERNANCE_TOPIC_PROFILES = {
    "مجلس الإدارة ومجلس الأمناء": (
        "مجلس الادارة مجلس الامناء اختصاصات المجلس اعضاء المجلس تشكيل المجلس الرئيس النائب "
        "اجتماعات المجلس دورة المجلس مسؤوليات العضو عزل المجلس النصاب واجبات الامناء"
    ),
    "اللجان والرقابة الداخلية والمراجعة": (
        "لجنة المراجعة لجنة الترشيحات والمكافات الرقابة الداخلية المراجع الداخلي المراجع الخارجي "
        "تقويم انظمة الرقابة المحاسبية فحص المستندات لجان المجلس اعضاء مستقلون تفتيش فحص"
    ),
    "الإفصاح والشفافية وتعارض المصالح": (
        "الافصاح الشفافية تعارض المصالح مصلحة شخصية تقرير سنوي بطلان التصرف سرية المعلومات "
        "نشر القوائم المالية نشر الاسماء اقرار خطي استغلال العضوية"
    ),
    "العضوية والجمعية العمومية والانتخابات": (
        "الجمعية العمومية شروط العضوية فئات العضوية العضو العادي العضو الداعم الاسلوب الموزون "
        "التراكمي التصويت الانتخابات الترشح لجنة الانتخابات الطعون التظلمات الاقتراع السري"
    ),
    "الرقابة المالية والمكافآت والميزانية": (
        "المكافات مخصصات مالية حساب بنكي الزكاة اموال الزكاة ميزانية قوائم مالية حساب ختامي "
        "استدامة مالية ايرادات ثمانية ملايين استثمار فائض الاموال رهن الاصول"
    ),
    "الجهاز التنفيذي والوظائف القيادية": (
        "الجهاز التنفيذي المدير العام الرئيس التنفيذي الوظائف القيادية المدير المالي تفرغ "
        "الهيكل التنظيمي ادارة العمل اليومي تنفيذ الخطط تقارير دورية"
    ),
    "حقوق المستفيدين وأصحاب المصلحة": (
        "المستفيدين اصحاب المصلحة حماية الحقوق تسوية الشكاوى النسبة الادارية من التبرعات "
        "سلوك مهني سرية بيانات المانحين العدالة والمساواة"
    ),
    "التأسيس والترخيص والحل والدمج": (
        "تاسيس ترخيص تجديد الترخيص الغاء الترخيص حل الجمعية تصفية دمج تعليق النشاط انذار "
        "مخالفات جزاءات مواءمة الاوضاع"
    ),
    "المجالس التنسيقية والفروع": (
        "مجلس المؤسسات الاهلية مجلس الجمعيات الاهلية المجالس الفرعية مناطقية تخصصية "
        "فروع تمثيل القطاع الشأن العام تنسيق الجهود"
    )
}

ARABIC_LEGAL_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "أو", "ثم", "أن", "إن",
    "لا", "ما", "لم", "لن", "كل", "جميع", "أحد", "أي", "بما", "كما", "هو", "هي", "هم", "يكون",
    "تكون", "كان", "كانت", "له", "لها", "لهم", "به", "بها", "وفق", "وفقا", "بموجب", "أمام", "بعد",
    "قبل", "حتى", "دون", "حالة", "حالات", "غير", "لدى", "بين", "منها", "فيه", "فيها", "إلا",
    "اذا", "إذا", "ذات", "ذو", "ذوي", "التي", "الذي", "الذين", "اللاتي", "أينما", "وردت", "يقصد",
    "الاحكام", "المادة", "الفقرة", "النظام", "الائحة", "القواعد", "المركز"
}


class LegalTopicClassifier:
    """
    NLP Engine for classifying Arabic legal articles into Governance Dimensions
    using TF-IDF Vectorization and Cosine Similarity.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(ARABIC_LEGAL_STOPWORDS),
            max_features=1500
        )
        self.topics = list(GOVERNANCE_TOPIC_PROFILES.keys())
        self.profile_texts = [
            normalize_arabic_text(GOVERNANCE_TOPIC_PROFILES[t]) for t in self.topics
        ]
        self._fit()

    def _fit(self):
        # Fit vectorizer on all profiles
        self.tfidf_matrix_profiles = self.vectorizer.fit_transform(self.profile_texts)

    def classify_text(self, text: str) -> Tuple[str, float, List[str]]:
        """
        Classifies an article into the best matching governance topic.
        Returns:
            Tuple of (assigned_topic, confidence_score, top_evidence_keywords)
        """
        norm_text = normalize_arabic_text(text)
        if not norm_text.strip():
            return "التعريفات والمفاهيم العامة", 0.0, []

        text_vec = self.vectorizer.transform([norm_text])
        similarities = cosine_similarity(text_vec, self.tfidf_matrix_profiles)[0]

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        assigned_topic = self.topics[best_idx]

        # Extract top keywords from article present in vectorizer vocabulary
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = text_vec.toarray()[0]
        top_indices = tfidf_scores.argsort()[-5:][::-1]
        top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

        return assigned_topic, round(best_score, 4), top_keywords

    def classify_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Applies AI classification across the entire legal DataFrame.
        """
        results = []
        for _, row in df.iterrows():
            topic, score, keywords = self.classify_text(row[text_col])
            results.append({
                "ai_predicted_topic": topic,
                "ai_confidence_score": score,
                "ai_evidence_keywords": ", ".join(keywords)
            })

        df_res = df.copy()
        for col in ["ai_predicted_topic", "ai_confidence_score", "ai_evidence_keywords"]:
            df_res[col] = [r[col] for r in results]

        return df_res


if __name__ == "__main__":
    from src.legal_text_extraction import load_legal_corpus
    corpus = load_legal_corpus()
    clf = LegalTopicClassifier()
    df_classified = clf.classify_dataframe(corpus)
    print("AI Classification complete!")
    print(df_classified[["doc_id", "article_num", "primary_topic", "ai_predicted_topic", "ai_confidence_score"]].head())
