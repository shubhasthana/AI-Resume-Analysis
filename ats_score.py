"""ATS Compatibility Scoring Engine for AI Resume Analyzer.

Calculates multi-factor compatibility score based on:
1. Skill Match (50%)
2. Keyword & Semantic Match (20%)
3. Education & Experience Match (15%)
4. Resume Structure & Formatting Health (15%)
"""

import re
import math
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import (
    clean_text,
    extract_tokens,
    extract_keywords,
    segment_resume_sections,
    extract_contact_info,
    calculate_text_metrics,
    ACTION_VERBS
)
from skill_match import SkillMatcher

# Standard education and degree keywords
DEGREE_KEYWORDS = [
    "bachelor", "bachelors", "b.tech", "btech", "b.e", "be", "b.s", "bs", "b.sc", "bsc", "bca",
    "master", "masters", "m.tech", "mtech", "m.e", "me", "m.s", "ms", "m.sc", "msc", "mca", "mba",
    "phd", "doctorate", "diploma", "associate", "computer science", "information technology",
    "data science", "software engineering", "electrical", "electronics", "mathematics", "statistics"
]

EXPERIENCE_KEYWORDS = [
    "intern", "internship", "developer", "engineer", "analyst", "lead", "architect",
    "junior", "senior", "mid-level", "associate", "specialist", "consultant", "manager"
]


class ATSScoreCalculator:
    """Calculates weighted ATS compatibility score between Resume and Job Description."""

    WEIGHT_SKILLS = 0.50
    WEIGHT_KEYWORDS = 0.20
    WEIGHT_EDUCATION_EXP = 0.15
    WEIGHT_STRUCTURE = 0.15

    @classmethod
    def calculate_keyword_score(cls, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Calculates semantic and TF-IDF keyword overlap between Resume and JD."""
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(jd_text)

        if not cleaned_resume or not cleaned_jd:
            return {"score": 0.0, "cosine_sim": 0.0, "keyword_match_rate": 0.0, "matched_keywords": []}

        # 1. Cosine similarity using TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
            cosine_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            cosine_sim = 0.0

        # 2. Keyphrase overlap
        jd_keywords = [k for k, _ in extract_keywords(cleaned_jd, top_n=30)]
        resume_tokens = set(extract_tokens(cleaned_resume))
        
        matched_kw = [kw for kw in jd_keywords if kw in resume_tokens or kw in cleaned_resume]
        kw_rate = (len(matched_kw) / len(jd_keywords)) if jd_keywords else 1.0

        # Weighted combination: 60% TF-IDF Cosine Similarity + 40% Keyword Hit Rate
        combined_score = (cosine_sim * 0.60 + kw_rate * 0.40) * 100
        combined_score = min(max(combined_score, 0.0), 100.0)

        return {
            "score": round(combined_score, 1),
            "cosine_sim": round(cosine_sim * 100, 1),
            "keyword_match_rate": round(kw_rate * 100, 1),
            "top_jd_keywords": jd_keywords[:15],
            "matched_keywords": matched_kw
        }

    @classmethod
    def calculate_education_exp_score(cls, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Evaluates degree matches, experience depth, action verbs, and seniority indicators."""
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()

        # 1. Degree Match
        resume_degrees = [deg for deg in DEGREE_KEYWORDS if deg in resume_lower]
        jd_degrees = [deg for deg in DEGREE_KEYWORDS if deg in jd_lower]

        if jd_degrees:
            degree_overlap = [deg for deg in jd_degrees if deg in resume_degrees]
            degree_score = (len(degree_overlap) / len(jd_degrees)) * 100
        else:
            degree_score = 90.0 if resume_degrees else 60.0

        # 2. Action Verbs Count (Quantified achievements indicator)
        action_verb_count = sum(1 for v in ACTION_VERBS if re.search(r'\b' + re.escape(v) + r'\b', resume_lower))
        # 8+ action verbs gives full action verb score
        action_score = min((action_verb_count / 8.0) * 100, 100.0)

        # 3. Experience & Project mention check
        has_dates = bool(re.search(r'\b(20\d\d|19\d\d|present|current)\b', resume_lower))
        has_metrics = bool(re.search(r'\b(\d+%\s*|\$\d+|\d+\+?\s*(?:users|clients|requests|ms|sec|x|k))\b', resume_lower))

        depth_score = (50.0 if has_dates else 20.0) + (50.0 if has_metrics else 25.0)

        # Composite Education / Experience score
        composite = (degree_score * 0.40) + (action_score * 0.35) + (depth_score * 0.25)
        composite = min(max(composite, 0.0), 100.0)

        return {
            "score": round(composite, 1),
            "degree_score": round(degree_score, 1),
            "action_score": round(action_score, 1),
            "detected_degrees": resume_degrees,
            "action_verb_count": action_verb_count,
            "has_metrics": has_metrics
        }

    @classmethod
    def calculate_structure_score(cls, resume_text: str) -> Dict[str, Any]:
        """Evaluates resume structure, sections, length, and contact details."""
        sections = segment_resume_sections(resume_text)
        contacts = extract_contact_info(resume_text)
        metrics = calculate_text_metrics(resume_text)

        # 1. Section presence points (max 50 pts)
        sec_points = 0
        if sections.get("skills") and len(sections["skills"]) > 10:
            sec_points += 12
        if sections.get("experience") and len(sections["experience"]) > 20:
            sec_points += 12
        if sections.get("education") and len(sections["education"]) > 10:
            sec_points += 10
        if sections.get("projects") and len(sections["projects"]) > 15:
            sec_points += 10
        if sections.get("summary") and len(sections["summary"]) > 20:
            sec_points += 6

        # 2. Contact details points (max 25 pts)
        contact_points = 0
        if contacts.get("email"):
            contact_points += 8
        if contacts.get("phone"):
            contact_points += 7
        if contacts.get("linkedin"):
            contact_points += 5
        if contacts.get("github") or contacts.get("portfolio"):
            contact_points += 5

        # 3. Word count & formatting health (max 25 pts)
        wc = metrics["word_count"]
        if 300 <= wc <= 900:
            wc_points = 25
        elif 200 <= wc < 300 or 900 < wc <= 1300:
            wc_points = 18
        elif 100 <= wc < 200:
            wc_points = 10
        else:
            wc_points = 5

        total_structure_score = min(max(sec_points + contact_points + wc_points, 0), 100)

        return {
            "score": round(float(total_structure_score), 1),
            "section_points": sec_points,
            "contact_points": contact_points,
            "length_points": wc_points,
            "word_count": wc,
            "contacts": contacts
        }

    @classmethod
    def calculate_ats_score(cls, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Calculates the complete ATS Compatibility report.

        Args:
            resume_text: Raw plain text of resume.
            jd_text: Target Job Description text.

        Returns:
            Dict containing composite score, breakdown, status badge, and sub-metrics.
        """
        # 1. Skill Match Score (50%)
        skill_res = SkillMatcher.match_skills(resume_text, jd_text)
        skill_score = skill_res["match_percentage"]

        # 2. Keyword Match Score (20%)
        kw_res = cls.calculate_keyword_score(resume_text, jd_text)
        kw_score = kw_res["score"]

        # 3. Education / Experience Match Score (15%)
        edu_res = cls.calculate_education_exp_score(resume_text, jd_text)
        edu_score = edu_res["score"]

        # 4. Resume Structure Score (15%)
        struct_res = cls.calculate_structure_score(resume_text)
        struct_score = struct_res["score"]

        # Composite calculation
        composite_score = (
            (skill_score * cls.WEIGHT_SKILLS) +
            (kw_score * cls.WEIGHT_KEYWORDS) +
            (edu_score * cls.WEIGHT_EDUCATION_EXP) +
            (struct_score * cls.WEIGHT_STRUCTURE)
        )
        composite_score = round(min(max(composite_score, 0.0), 100.0), 1)

        # Rating Tier & Badge
        if composite_score >= 85:
            rating_label = "Outstanding Match"
            rating_color = "#00F5D4" # Neon Cyan / Mint
            rating_grade = "A+"
        elif composite_score >= 70:
            rating_label = "Strong Match"
            rating_color = "#00BBF9" # Electric Blue
            rating_grade = "A"
        elif composite_score >= 50:
            rating_label = "Moderate Match"
            rating_color = "#FEE440" # Amber Yellow
            rating_grade = "B"
        else:
            rating_label = "Needs Improvement"
            rating_color = "#F15BB5" # Neon Coral / Pink
            rating_grade = "C"

        return {
            "ats_score": int(round(composite_score)),
            "ats_score_exact": composite_score,
            "rating_label": rating_label,
            "rating_color": rating_color,
            "rating_grade": rating_grade,
            "breakdown": {
                "skills": {
                    "weight_percent": 50,
                    "score": skill_score,
                    "contribution": round(skill_score * cls.WEIGHT_SKILLS, 1),
                    "matched_count": len(skill_res["matched_skills"]),
                    "missing_count": len(skill_res["missing_skills"])
                },
                "keywords": {
                    "weight_percent": 20,
                    "score": kw_score,
                    "contribution": round(kw_score * cls.WEIGHT_KEYWORDS, 1),
                    "cosine_similarity": kw_res["cosine_sim"],
                    "keyword_hit_rate": kw_res["keyword_match_rate"]
                },
                "education_experience": {
                    "weight_percent": 15,
                    "score": edu_score,
                    "contribution": round(edu_score * cls.WEIGHT_EDUCATION_EXP, 1),
                    "action_verbs": edu_res["action_verb_count"]
                },
                "structure": {
                    "weight_percent": 15,
                    "score": struct_score,
                    "contribution": round(struct_score * cls.WEIGHT_STRUCTURE, 1),
                    "word_count": struct_res["word_count"]
                }
            },
            "skill_details": skill_res,
            "keyword_details": kw_res,
            "education_details": edu_res,
            "structure_details": struct_res
        }

