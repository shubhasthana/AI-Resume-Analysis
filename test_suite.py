"""Comprehensive Automated Test Suite for AI Resume Analyzer.

Tests all core components:
1. User Authentication & Password Hashing
2. Text Processing & NLP Utilities
3. PDF Parser & Metadata Extractor
4. Skill Taxonomy & Matching Engine
5. Multi-Factor ATS Compatibility Engine
6. ML Model Category Prediction & Evaluation
7. Recommendation Engine
8. 3D Visuals & HTML WebGL Generators
"""

import os
import io
import unittest
import numpy as np

# Core module imports
from auth import AuthManager
from utils.text_processing import (
    clean_text,
    extract_tokens,
    extract_keywords,
    extract_contact_info,
    segment_resume_sections,
    calculate_text_metrics
)
from resume_parser import ResumeParser
from skill_match import SkillExtractor, SkillMatcher
from ats_score import ATSScoreCalculator
from model import ResumeClassifier, get_classifier
from recommendations import RecommendationEngine
from utils.visuals_3d import (
    get_3d_hero_header_html,
    get_3d_score_orb_html,
    get_3d_skill_constellation_html,
    get_3d_login_portal_html,
    plot_3d_category_vector_space,
    plot_3d_capability_mesh
)


class TestAuthManager(unittest.TestCase):
    """Tests authentication, password hashing, and secure registration flow."""

    def test_no_default_demo_account_exists(self):
        users = AuthManager._load_users()
        self.assertNotIn("demo_user", users)

    def test_user_registration_and_login(self):
        import time
        uid = str(int(time.time() * 1000))
        test_username = f"user_{uid}"
        test_email = f"user_{uid}@example.com"
        test_pwd = "SecurePassword123!"

        # Register
        success, msg = AuthManager.register_user(
            username=test_username,
            email=test_email,
            password=test_pwd,
            full_name="Test Engineer",
            target_role="Data Scientist"
        )
        self.assertTrue(success)

        # Login
        success, msg, profile = AuthManager.authenticate_user(test_username, test_pwd)
        self.assertTrue(success)
        self.assertEqual(profile["username"], test_username)
        self.assertEqual(profile["email"], test_email)

        # Scan increment
        scans = AuthManager.increment_user_scans(test_username)
        self.assertGreaterEqual(scans, 1)


class TestTextProcessing(unittest.TestCase):
    """Tests text processing and NLP extraction utilities."""

    def test_clean_text(self):
        raw = "Check out https://github.com/test for Python &amp; Django @ user@test.com (555) 123-4567!"
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("user@test.com", cleaned)
        self.assertIn("python", cleaned)
        self.assertIn("django", cleaned)

    def test_extract_contact_info(self):
        text = "Jane Doe | Email: jane.doe@example.com | Phone: 555-123-4567 | LinkedIn: linkedin.com/in/janedoe | GitHub: github.com/janedoe"
        contacts = extract_contact_info(text)
        self.assertEqual(contacts["email"], "jane.doe@example.com")
        self.assertIn("555", contacts["phone"])
        self.assertIn("janedoe", contacts["linkedin"])
        self.assertIn("janedoe", contacts["github"])

    def test_segment_resume_sections(self):
        sample = """
        John Doe
        Summary:
        Experienced developer building web apps.
        Technical Skills:
        Python, Django, PostgreSQL
        Work Experience:
        Software Engineer at Acme Corp 2021-2023. Built APIs.
        Education:
        B.Tech Computer Science
        Projects:
        E-commerce backend in FastAPI.
        """
        sections = segment_resume_sections(sample)
        self.assertTrue(len(sections["summary"]) > 0)
        self.assertTrue(len(sections["skills"]) > 0)
        self.assertTrue(len(sections["experience"]) > 0)
        self.assertTrue(len(sections["education"]) > 0)

    def test_calculate_text_metrics(self):
        text = "Developed automated CI/CD pipelines. Scaled microservices handling 50k users. Improved database queries."
        metrics = calculate_text_metrics(text)
        self.assertGreater(metrics["word_count"], 5)
        self.assertGreater(metrics["action_verb_count"], 0)


class TestResumeParser(unittest.TestCase):
    """Tests PDF parsing, OCR fallback, and structural health audit."""

    def test_normal_text_pdf(self):
        sample_pdf = os.path.join(os.path.dirname(__file__), "resumes", "sample_python_developer.pdf")
        if os.path.exists(sample_pdf):
            res = ResumeParser.parse(sample_pdf)
            self.assertTrue(res["success"])
            self.assertEqual(res["extraction_method"], "native")
            self.assertFalse(res["ocr_used"])
            self.assertIn("ALEXANDER", res["text"])
            self.assertGreater(res["metrics"]["word_count"], 100)
            self.assertTrue(res["section_status"]["has_skills_section"])

    def test_empty_and_corrupt_input_handling(self):
        # Empty bytes
        res_empty = ResumeParser.parse(b"")
        self.assertFalse(res_empty["success"])
        self.assertIsNotNone(res_empty["error"])

        # Corrupted random bytes
        res_corrupt = ResumeParser.parse(b"This is not a valid PDF file content.")
        self.assertFalse(res_corrupt["success"])
        self.assertIsNotNone(res_corrupt["error"])

    def test_scanned_image_pdf_fallback(self):
        # Generate an in-memory image-only PDF (no embedded text)
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "SCANNED RESUME CONTENT", fill=(0, 0, 0))
        
        pdf_buf = io.BytesIO()
        img.save(pdf_buf, format="PDF")
        scanned_pdf_bytes = pdf_buf.getvalue()

        # Parse scanned PDF
        res = ResumeParser.extract_text_from_pdf(scanned_pdf_bytes)
        # Should detect that native text is empty/insufficient and invoke OCR
        self.assertEqual(res["extraction_method"], "ocr")
        self.assertTrue(res["ocr_used"])

    def test_multipage_scanned_pdf_page_limit(self):
        # Generate a multi-page image-only PDF (6 pages)
        from PIL import Image
        img = Image.new("RGB", (400, 600), color=(255, 255, 255))
        images = [img.copy() for _ in range(6)]
        
        pdf_buf = io.BytesIO()
        images[0].save(pdf_buf, format="PDF", save_all=True, append_images=images[1:])
        multi_pdf_bytes = pdf_buf.getvalue()

        res = ResumeParser.extract_text_from_pdf(multi_pdf_bytes)
        self.assertEqual(res["page_count"], 6)
        # If OCR ran or attempted, total_pages is 6 and exceeds MAX_OCR_PAGES (5)
        if res.get("ocr_warning"):
            self.assertIn("5 pages", res["ocr_warning"])


class TestSkillMatching(unittest.TestCase):
    """Tests skill extraction and matching engine."""

    def test_skill_extraction_and_aliases(self):
        text = "Built web apps with React.js, K8s, Python3, Postgres, REST APIs, and Docker."
        skills = SkillExtractor.extract_skills(text)
        self.assertIn("React", skills)
        self.assertIn("Kubernetes", skills)
        self.assertIn("Python", skills)
        self.assertIn("PostgreSQL", skills)
        self.assertIn("REST API", skills)
        self.assertIn("Docker", skills)

    def test_skill_matcher(self):
        resume_text = "Experienced in Python, SQL, Git, and Pandas."
        jd_text = "Looking for Python Developer with Python, Django, SQL, Git, and REST API."
        res = SkillMatcher.match_skills(resume_text, jd_text)
        
        self.assertIn("Python", res["matched_skills"])
        self.assertIn("SQL", res["matched_skills"])
        self.assertIn("Git", res["matched_skills"])
        self.assertIn("Django", res["missing_skills"])
        self.assertIn("REST API", res["missing_skills"])
        self.assertIn("Pandas", res["additional_skills"])
        self.assertGreater(res["match_percentage"], 0.0)


class TestATSScoreCalculator(unittest.TestCase):
    """Tests 4-factor ATS score calculations."""

    def test_ats_score_bounds_and_breakdown(self):
        resume_text = """
        Alexander Rivers
        Email: alex@example.com | Phone: 555-019-2834 | GitHub: github.com/alex
        Summary:
        Backend Python developer with experience in Django, FastAPI, PostgreSQL, Docker, and REST APIs.
        Technical Skills:
        Python, Django, FastAPI, PostgreSQL, Docker, Git, PyTest, Redis
        Experience:
        Python Developer | TechCorp. Built scalable microservices, reduced latency by 35%.
        Education:
        Bachelor of Technology in Computer Science
        Projects:
        Distributed E-commerce Engine in Python.
        """
        jd_text = """
        Looking for a Python Developer with experience in Python, Django, PostgreSQL, Docker, and REST API.
        Requires Bachelor's in Computer Science.
        """
        ats_res = ATSScoreCalculator.calculate_ats_score(resume_text, jd_text)
        self.assertGreaterEqual(ats_res["ats_score"], 60)
        self.assertLessEqual(ats_res["ats_score"], 100)
        self.assertIn(ats_res["rating_grade"], ["A+", "A", "B", "C"])
        self.assertIn("skills", ats_res["breakdown"])
        self.assertIn("keywords", ats_res["breakdown"])
        self.assertIn("education_experience", ats_res["breakdown"])
        self.assertIn("structure", ats_res["breakdown"])


class TestMLModel(unittest.TestCase):
    """Tests ML category classification and evaluation metrics."""

    def test_classifier_accuracy_and_prediction(self):
        clf = get_classifier()
        metrics = clf.get_metrics()
        self.assertGreaterEqual(metrics["accuracy"], 80.0)
        self.assertGreaterEqual(metrics["f1_score"], 80.0)
        
        text = "Built deep learning neural networks with PyTorch, TensorFlow, Scikit-Learn, and Python for NLP and Computer Vision."
        pred = clf.predict_category(text)
        self.assertIn(pred["predicted_category"], ["Data Science", "Machine Learning Engineer"])
        self.assertGreater(pred["confidence_score"], 0.0)


class TestRecommendations(unittest.TestCase):
    """Tests recommendation generation and ethical constraints."""

    def test_recommendation_engine(self):
        ats_data = {
            "ats_score": 75,
            "skill_details": {
                "matched_skills": ["Python", "SQL"],
                "missing_skills": ["Docker", "Kubernetes"],
                "additional_skills": ["Pandas"]
            },
            "keyword_details": {
                "top_jd_keywords": ["python", "docker", "kubernetes", "microservices"],
                "matched_keywords": ["python"]
            },
            "education_details": {
                "action_verb_count": 3,
                "has_metrics": False
            },
            "structure_details": {
                "word_count": 220,
                "contacts": {"email": "test@example.com"}
            }
        }
        resume_metadata = {
            "section_status": {"has_projects_section": False}
        }
        recs = RecommendationEngine.generate_recommendations(ats_data, resume_metadata)
        self.assertGreater(len(recs["suggestions"]), 0)
        self.assertIn("Ethical Reminder", recs["ethical_note"])


class Test3DVisuals(unittest.TestCase):
    """Tests 3D visual HTML generators and Plotly 3D visual models."""

    def test_3d_html_generators(self):
        login_html = get_3d_login_portal_html()
        self.assertIn("three-login-portal", login_html)

        hero_html = get_3d_hero_header_html()
        self.assertIn("three-hero-container", hero_html)

        score_orb_html = get_3d_score_orb_html(85, "Outstanding Match", "#00F5D4", "A+")
        self.assertIn("85", score_orb_html)
        self.assertIn("three-score-orb-container", score_orb_html)

        constellation_html = get_3d_skill_constellation_html(
            matched_skills=["Python", "SQL"],
            missing_skills=["Docker"],
            additional_skills=["Pandas"]
        )
        self.assertIn("three-skill-constellation-container", constellation_html)

    def test_plotly_3d_plots(self):
        clf = get_classifier()
        fig = plot_3d_category_vector_space(
            resume_text="Python developer with Django and PostgreSQL.",
            jd_text="Python backend developer wanted.",
            classifier_obj=clf
        )
        self.assertIsNotNone(fig)
        self.assertTrue(len(fig.data) >= 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
