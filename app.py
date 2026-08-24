"""AI Resume Analyzer - 3D Streamlit Web Application with User Authentication.

A modern, AI-powered Resume Analyzer using NLP, Scikit-learn Machine Learning,
PyPDF text extraction, interactive WebGL / Three.js 3D Visualizations, and secure User Authentication.
"""

import os
import io
import json
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# Import custom core modules
from auth import AuthManager
from resume_parser import ResumeParser
from skill_match import SkillMatcher, SKILL_TAXONOMY
from ats_score import ATSScoreCalculator
from model import get_classifier, ResumeClassifier, SAMPLE_RESUME_DATA
from recommendations import RecommendationEngine
from utils.text_processing import (
    clean_text,
    extract_keywords,
    extract_contact_info,
    calculate_text_metrics,
    ACTION_VERBS
)
from utils.visuals_3d import (
    get_3d_hero_header_html,
    get_3d_score_orb_html,
    get_3d_skill_constellation_html,
    get_3d_login_portal_html,
    plot_3d_category_vector_space,
    plot_3d_capability_mesh
)

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer 3D",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glassmorphic & 3D CSS Theme
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Montserrat:wght@700;900&display=swap');

    /* Global styling */
    .stApp {
        background-color: #05070e;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header gradient */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }

    /* Glassmorphic 3D Card */
    .glass-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(13, 20, 36, 0.6) 100%);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 245, 212, 0.05);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 245, 212, 0.45);
        box-shadow: 0 15px 40px rgba(0, 245, 212, 0.15);
    }

    /* Metric card */
    .metric-box {
        background: rgba(10, 16, 30, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-value {
        font-family: 'Montserrat', sans-serif;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Badges */
    .skill-chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px 4px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .skill-chip-matched {
        background: rgba(0, 245, 212, 0.12);
        color: #00F5D4;
        border: 1px solid rgba(0, 245, 212, 0.4);
        box-shadow: 0 0 10px rgba(0, 245, 212, 0.15);
    }
    .skill-chip-missing {
        background: rgba(255, 0, 127, 0.12);
        color: #FF007F;
        border: 1px solid rgba(255, 0, 127, 0.4);
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.15);
    }
    .skill-chip-bonus {
        background: rgba(0, 187, 249, 0.12);
        color: #00BBF9;
        border: 1px solid rgba(0, 187, 249, 0.4);
    }

    /* Streamlit tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94A3B8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(123, 44, 191, 0.3) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(0, 245, 212, 0.4) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00F5D4 0%, #00BBF9 50%, #7B2CBF 100%);
        color: #05070e;
        font-weight: 700;
        font-size: 15px;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 20px rgba(0, 245, 212, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 30px rgba(0, 245, 212, 0.5);
        transform: scale(1.02);
        color: #000000;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00F5D4, #7B2CBF);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_resume_text_image(resume_text: str, title: str = "Extracted Resume Text"):
    """Render resume text into a clean image preview for better readability."""
    if not resume_text or not resume_text.strip():
        return None

    base_width = 1200
    margin = 60
    line_spacing = 22
    title_height = 90
    wrap_width = 92

    wrapped_lines = []
    for para in resume_text.splitlines():
        if not para.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(para, width=wrap_width, break_long_words=False, replace_whitespace=False))

    line_count = max(1, len(wrapped_lines) + 4)
    height = max(500, title_height + (line_count * line_spacing) + margin)

    img = Image.new("RGB", (base_width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.rounded_rectangle(
        [20, 20, base_width - 20, height - 20],
        radius=20,
        fill=(248, 250, 252),
        outline=(148, 163, 184),
        width=2,
    )

    draw.text((margin, 30), title, fill=(15, 23, 42), font=title_font)

    current_y = title_height
    for line in wrapped_lines:
        draw.text((margin, current_y), line, fill=(15, 23, 42), font=body_font)
        current_y += line_spacing

    return img

# Preset Job Description Templates
JD_TEMPLATES = {
    "Select a Preset Template or Paste Custom JD...": "",
    "Python Backend Developer": """We are looking for a skilled Python Backend Developer to join our engineering team.
Requirements:
- Strong proficiency in Python, Django, FastAPI, or Flask.
- Experience with relational databases such as PostgreSQL or MySQL and caching with Redis.
- Solid understanding of RESTful APIs, microservices architecture, and asynchronous programming.
- Hands-on experience with Docker, Git, CI/CD pipelines, and AWS cloud deployment.
- Familiarity with unit testing frameworks like PyTest.
- Bachelor's degree in Computer Science, Engineering, or related technical field.""",

    "Data Scientist": """We are seeking an experienced Data Scientist to build predictive models and analytics solutions.
Requirements:
- Proficient in Python, R, SQL, Pandas, NumPy, Scikit-Learn, and PyTorch or TensorFlow.
- Experience in statistical modeling, machine learning, deep learning, NLP, and feature engineering.
- Ability to build data pipelines and deploy ML models using FastAPI, Docker, and AWS SageMaker.
- Strong data visualization skills with Matplotlib, Seaborn, or Tableau.
- Master's or Bachelor's degree in Data Science, Statistics, Computer Science, or Mathematics.""",

    "Full Stack Web Developer": """Seeking a talented Full Stack Web Developer to build scalable, responsive web applications.
Requirements:
- Strong hands-on experience with React, Next.js, TypeScript, JavaScript, HTML5, CSS3, and Tailwind CSS.
- Backend proficiency in Node.js, Express.js, GraphQL, and REST APIs.
- Experience with databases including MongoDB, PostgreSQL, and Redis.
- Knowledge of Docker, Git, GitHub Actions, AWS S3, and modern frontend testing (Jest, Cypress).
- Passion for responsive design, web performance optimization, and accessibility.""",

    "DevOps & Cloud Engineer": """Looking for a Cloud & DevOps Engineer to design and maintain automated cloud infrastructure.
Requirements:
- Extensive experience with AWS or Azure cloud environments.
- Deep expertise in Docker, Kubernetes, Terraform, Ansible, and Helm.
- Proven experience building CI/CD pipelines using Jenkins, GitHub Actions, or GitLab CI.
- Strong Linux system administration and Bash / Python scripting skills.
- Knowledge of Prometheus, Grafana, and ELK monitoring stacks.""",

    "Cybersecurity Analyst": """We are hiring a Cybersecurity Analyst to strengthen enterprise security operations.
Requirements:
- Experience in vulnerability assessment, penetration testing, network security, and SIEM tools (Splunk).
- Hands-on knowledge of OWASP Top 10, ethical hacking, Wireshark, and Metasploit.
- Understanding of Identity & Access Management (IAM), Firewalls, and Zero Trust architecture.
- Industry certifications such as CEH, CompTIA Security+, or CISSP preferred.""",

    "Mobile App Developer (Flutter/React Native)": """Looking for a Mobile App Developer to build next-generation cross-platform mobile apps.
Requirements:
- Experience developing iOS and Android applications with Flutter (Dart) or React Native.
- Strong understanding of mobile UI/UX, state management (Bloc, Provider, Redux), and REST APIs.
- Experience integrating Firebase authentication, push notifications, and local SQLite caching.
- Familiarity with Google Play Store and Apple App Store deployment procedures."""
}

# Preset Sample Resume Options
SAMPLE_RESUME_PATHS = {
    "Select or Upload your own PDF...": None,
    "Sample 1: Python Backend Developer": os.path.join(os.path.dirname(__file__), "resumes", "sample_python_developer.txt"),
    "Sample 2: Senior Data Scientist": os.path.join(os.path.dirname(__file__), "resumes", "sample_data_scientist.txt"),
    "Sample 3: Full Stack Web Developer": os.path.join(os.path.dirname(__file__), "resumes", "sample_web_developer.txt")
}

# Initialize Application State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

classifier = get_classifier()


# ==============================================================================
# VIEW 1: 3D LOGIN & REGISTRATION SCREEN (WHEN NOT AUTHENTICATED)
# ==============================================================================
if not st.session_state.authenticated:
    # 3D Quantum Login Portal Header
    components.html(get_3d_login_portal_html(), height=385, scrolling=False)

    col_left, col_center, col_right = st.columns([1, 1.8, 1])

    with col_center:
        st.markdown("<br>", unsafe_allow_html=True)
        login_tab, register_tab = st.tabs(["🔐 Sign In", "✨ Create Account"])

        # TAB: Sign In
        with login_tab:
            st.markdown("#### Welcome Back")
            st.caption("Enter your credentials to access your AI resume workspace.")
            
            with st.form("login_form"):
                login_identifier = st.text_input("Username or Email", placeholder="e.g. alexrivers or alex@example.com")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In to Workspace", use_container_width=True)

            if submit_login:
                success, msg, profile = AuthManager.authenticate_user(login_identifier, login_password)
                if success and profile:
                    st.session_state.authenticated = True
                    st.session_state.current_user = profile
                    st.toast(f"Welcome back, {profile['full_name']}!", icon="👋")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        # TAB: Create Account
        with register_tab:
            st.markdown("#### New User Registration")
            st.caption("Create your account to save scans and access 3D AI features.")
            
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Alex Rivers")
                reg_username = st.text_input("Username", placeholder="e.g. alexrivers")
                reg_email = st.text_input("Email Address", placeholder="e.g. alex@example.com")
                reg_role = st.selectbox(
                    "Primary Career Target",
                    [
                        "Python Backend Developer",
                        "Data Scientist",
                        "Machine Learning Engineer",
                        "Full Stack Web Developer",
                        "DevOps & Cloud Engineer",
                        "Cybersecurity Analyst",
                        "Data Analyst",
                        "Mobile App Developer",
                        "Software QA Engineer"
                    ]
                )
                reg_password = st.text_input("Password (min 6 chars)", type="password", placeholder="Choose a secure password")
                submit_register = st.form_submit_button("Create Account", use_container_width=True)

            if submit_register:
                success, msg = AuthManager.register_user(
                    username=reg_username,
                    email=reg_email,
                    password=reg_password,
                    full_name=reg_name,
                    target_role=reg_role
                )
                if success:
                    st.success(f"✅ {msg}")
                    st.info("You can now switch to the 'Sign In' tab to log in with your new credentials.")
                else:
                    st.error(f"❌ {msg}")

    # Footer note
    st.markdown("<br><div style='text-align: center; color: #64748B; font-size: 12px;'>AI Resume Analyzer 3D • Built with Python, Streamlit, Three.js & Scikit-Learn</div>", unsafe_allow_html=True)
    st.stop()


# ==============================================================================
# VIEW 2: AUTHENTICATED APPLICATION WORKSPACE
# ==============================================================================

current_user = st.session_state.current_user or {"full_name": "Candidate", "username": "user", "scans_count": 0, "target_role": "Developer"}

# Sidebar Navigation & Settings
with st.sidebar:
    # User Profile Card
    st.markdown(f"""
    <div class="glass-card" style="padding: 14px 16px; margin-bottom: 15px; border-left: 3px solid #00F5D4;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #00F5D4, #7B2CBF); display: flex; align-items: center; justify-content: center; font-weight: 800; color: #05070e; font-size: 16px;">
                {current_user['full_name'][0].upper()}
            </div>
            <div>
                <div style="font-weight: 700; font-size: 14px; color: #FFFFFF;">{current_user['full_name']}</div>
                <div style="font-size: 11px; color: #00F5D4;">@{current_user['username']} • {current_user.get('target_role', 'Engineer')}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 11px; color: #94A3B8; display: flex; justify-content: space-between;">
            <span>Resumes Analyzed:</span>
            <b style="color: #00F5D4;">{current_user.get('scans_count', 0)} scans</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔒 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = None
        st.toast("Signed out successfully.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🌌 AI RESUME 3D")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🚀 Resume Analyzer",
            "🌌 3D Skill Constellation & Galaxy",
            "🧪 Machine Learning Studio",
            "📖 ATS Optimization Guide"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Presets")
    selected_sample = st.selectbox("Load Sample Resume", list(SAMPLE_RESUME_PATHS.keys()))
    if selected_sample != "Select or Upload your own PDF..." and SAMPLE_RESUME_PATHS[selected_sample]:
        with open(SAMPLE_RESUME_PATHS[selected_sample], "r", encoding="utf-8") as f:
            st.session_state.sample_content = f.read()
            st.success(f"Loaded {selected_sample.split(':')[1].strip()}!")
    
    selected_jd_template = st.selectbox("Load Job Description Template", list(JD_TEMPLATES.keys()))
    
    st.markdown("---")
    st.markdown("#### 🛡️ Privacy & Compliance")
    st.caption("Resumes are processed in-memory and are never stored or exposed to third parties.")


# Top 3D Hero Header
components.html(get_3d_hero_header_html(), height=255, scrolling=False)


# TAB 1: RESUME ANALYZER
if app_mode == "🚀 Resume Analyzer":
    col_input_left, col_input_right = st.columns([1, 1], gap="large")

    with col_input_left:
        st.markdown("### 📄 1. Upload Resume")
        uploaded_file = st.file_uploader("Upload Resume in PDF format", type=["pdf"], help="Text-based PDF resumes recommended for best accuracy.")
        
        resume_raw_text = ""
        resume_source_name = "Uploaded Resume"
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF (with OCR fallback if scanned)..."):
                pdf_bytes = uploaded_file.getvalue()
                parse_res = ResumeParser.parse(pdf_bytes)
                if not parse_res["success"]:
                    st.error(f"⚠️ {parse_res['error']}")
                else:
                    resume_raw_text = parse_res["text"]
                    resume_source_name = uploaded_file.name
                    method_label = "OCR Engine" if parse_res.get("extraction_method") == "ocr" else "Native PDF Text"
                    st.success(f"✅ Successfully parsed {uploaded_file.name} ({parse_res['metrics']['word_count']} words, {parse_res['page_count']} page(s) via {method_label})")
                    
                    if parse_res.get("extraction_method") == "ocr":
                        st.info("ℹ️ This scanned PDF was processed with OCR. Please verify names, email addresses, phone numbers, and skills because OCR may contain mistakes.")
                    
                    if parse_res.get("ocr_warning"):
                        st.warning(f"⚠️ {parse_res['ocr_warning']}")
        elif "sample_content" in st.session_state and st.session_state.sample_content:
            resume_raw_text = st.session_state.sample_content
            resume_source_name = "Loaded Preset Sample"
            st.info("ℹ️ Using selected Sample Resume. You can also upload your own PDF above.")

        # Show preview expander
        if resume_raw_text:
            with st.expander("👁️ View Extracted Resume Text", expanded=False):
                preview_img = render_resume_text_image(resume_raw_text, "Extracted Resume Text")
                if preview_img is not None:
                    st.image(preview_img, use_container_width=True, caption="Rendered text preview")
                else:
                    st.text_area("Extracted Plain Text", resume_raw_text, height=180, disabled=True)

    with col_input_right:
        st.markdown("### 🎯 2. Target Job Description")
        
        initial_jd_text = JD_TEMPLATES.get(selected_jd_template, "")
        jd_input = st.text_area(
            "Paste Target Job Description (JD)",
            value=initial_jd_text,
            height=210,
            placeholder="Paste the job description or role requirements here..."
        )

    # Analyze Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("⚡ Analyze Resume with AI & 3D", use_container_width=True)

    if analyze_clicked:
        if not resume_raw_text.strip():
            st.warning("⚠️ Please upload a valid PDF resume or choose a sample resume before analyzing.")
        elif not jd_input.strip():
            st.warning("⚠️ Please enter or select a target Job Description before analyzing.")
        else:
            with st.spinner("🌌 Running Deep NLP Extraction, ATS Scoring, ML Classification & 3D Spatial Geometry..."):
                # Increment scan counter for current user
                new_scan_count = AuthManager.increment_user_scans(current_user["username"])
                st.session_state.current_user["scans_count"] = new_scan_count

                # 1. Parse resume structure & metadata
                resume_parsed = ResumeParser.parse(resume_raw_text.encode('utf-8'))
                if not resume_parsed["success"]:
                    resume_parsed = {
                        "text": resume_raw_text,
                        "cleaned_text": clean_text(resume_raw_text),
                        "contacts": extract_contact_info(resume_raw_text),
                        "metrics": calculate_text_metrics(resume_raw_text),
                        "section_status": {"has_contact_info": True, "has_skills_section": True, "has_experience_section": True, "has_education_section": True, "has_projects_section": True},
                        "structure_health_score": 85
                    }

                # 2. ATS Scoring
                ats_data = ATSScoreCalculator.calculate_ats_score(resume_raw_text, jd_input)

                # 3. ML Category Prediction
                pred_data = classifier.predict_category(resume_raw_text)

                # 4. Recommendation Engine
                rec_data = RecommendationEngine.generate_recommendations(ats_data, resume_parsed)

                st.session_state.analysis_results = {
                    "resume_parsed": resume_parsed,
                    "ats_data": ats_data,
                    "pred_data": pred_data,
                    "rec_data": rec_data,
                    "jd_text": jd_input,
                    "resume_text": resume_raw_text,
                    "source_name": resume_source_name
                }
                st.session_state.analysis_complete = True
                st.toast("Analysis Complete!", icon="✨")

    # DISPLAY ANALYSIS RESULTS DASHBOARD
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        res = st.session_state.analysis_results
        ats = res["ats_data"]
        pred = res["pred_data"]
        rec = res["rec_data"]
        parsed = res["resume_parsed"]
        skill_det = ats["skill_details"]
        kw_det = ats["keyword_details"]

        st.markdown("<hr style='border: 1px solid rgba(0, 245, 212, 0.2); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 25px;'>📊 RESUME ANALYSIS & ATS REPORT</h2>", unsafe_allow_html=True)

        if parsed.get("extraction_method") == "ocr":
            st.info("ℹ️ **OCR Extraction Notice**: This scanned PDF was processed with OCR. Please verify names, email addresses, phone numbers, and skills because OCR may contain mistakes.")

        # TOP ROW: 3D Holographic Score Orb + Quick Stats
        col_orb, col_stats = st.columns([1, 1.2], gap="large")

        with col_orb:
            components.html(
                get_3d_score_orb_html(
                    ats_score=ats["ats_score"],
                    rating_label=ats["rating_label"],
                    rating_color=ats["rating_color"],
                    rating_grade=ats["rating_grade"]
                ),
                height=325,
                scrolling=False
            )

        with col_stats:
            # Predicted Category Box
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 12px; padding: 16px 20px;">
                <div style="font-size: 11px; font-weight: 700; color: #00F5D4; letter-spacing: 1.5px; text-transform: uppercase;">
                    ML PREDICTED CAREER CATEGORY
                </div>
                <div style="font-size: 26px; font-weight: 900; color: #FFFFFF; margin: 4px 0;">
                    💼 {pred['predicted_category']}
                </div>
                <div style="font-size: 12px; color: #94A3B8;">
                    Classification Confidence: <b style="color: #00F5D4;">{pred['confidence_score']}%</b> • Based on TF-IDF ML Model
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 Metric Highlights Grid
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #00F5D4;">{len(skill_det['matched_skills'])}</div>
                    <div class="metric-label">Matched Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #FF007F;">{len(skill_det['missing_skills'])}</div>
                    <div class="metric-label">Missing Skills</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #00BBF9;">{int(kw_det['score'])}%</div>
                    <div class="metric-label">Keyword Match</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: #FEE440;">{parsed.get('structure_health_score', 85)}%</div>
                    <div class="metric-label">Format Health</div>
                </div>
                """, unsafe_allow_html=True)

            # Score Breakdown Bars
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            b = ats["breakdown"]
            st.write(f"**Skill Match Weight (50%):** {int(b['skills']['score'])}%")
            st.progress(min(int(b['skills']['score']), 100))

            st.write(f"**Semantic & Keyword Match Weight (20%):** {int(b['keywords']['score'])}%")
            st.progress(min(int(b['keywords']['score']), 100))

            st.write(f"**Education & Experience Weight (15%):** {int(b['education_experience']['score'])}%")
            st.progress(min(int(b['education_experience']['score']), 100))
            st.markdown("</div>", unsafe_allow_html=True)

        # 3D INTERACTIVE SKILL CONSTELLATION
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌌 3D Interactive Skill Constellation")
        st.caption("Rotate, zoom, and explore how your skills align with the target job requirements in real-time 3D space.")
        components.html(
            get_3d_skill_constellation_html(
                matched_skills=skill_det["matched_skills"],
                missing_skills=skill_det["missing_skills"],
                additional_skills=skill_det["additional_skills"]
            ),
            height=430,
            scrolling=False
        )

        # SKILL CHIPS BREAKDOWN
        col_matched, col_missing, col_additional = st.columns([1, 1, 1], gap="medium")

        with col_matched:
            st.markdown("#### ✅ Matched Skills")
            if skill_det["matched_skills"]:
                chips_html = "".join([f"<span class='skill-chip skill-chip-matched'>✓ {s}</span>" for s in skill_det["matched_skills"]])
                st.markdown(f"<div style='line-height: 2.2;'>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No direct skill matches detected. Consider aligning your terminology with the job description.")

        with col_missing:
            st.markdown("#### ❌ Missing Skills from JD")
            if skill_det["missing_skills"]:
                chips_html = "".join([f"<span class='skill-chip skill-chip-missing'>✗ {s}</span>" for s in skill_det["missing_skills"]])
                st.markdown(f"<div style='line-height: 2.2;'>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.success("🎉 Outstanding! Your resume covers all required skills detected in the JD.")

        with col_additional:
            st.markdown("#### ➕ Additional Skills Found")
            if skill_det["additional_skills"]:
                chips_html = "".join([f"<span class='skill-chip skill-chip-bonus'>+ {s}</span>" for s in skill_det["additional_skills"][:12]])
                st.markdown(f"<div style='line-height: 2.2;'>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("No extra predefined skills found.")

        # CATEGORY BREAKDOWN TABLE & RADAR
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 Skill Domain Coverage Breakdown")
        
        cat_breakdown = skill_det.get("category_breakdown", {})
        if cat_breakdown:
            table_rows = []
            for cat_name, data in cat_breakdown.items():
                matched_str = ", ".join(data["matched"]) if data["matched"] else "None"
                missing_str = ", ".join(data["missing"]) if data["missing"] else "None"
                table_rows.append({
                    "Domain": cat_name,
                    "Matched Skills": matched_str,
                    "Missing Skills": missing_str,
                    "Coverage": f"{data['coverage_percent']}%",
                    "Matched / Required": f"{data['total_matched']} / {data['total_required']}"
                })
            df_cat = pd.DataFrame(table_rows)
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

        # ACTIONABLE RECOMMENDATIONS & ETHICAL GUIDE
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Actionable Improvement Recommendations")
        
        for sug in rec["suggestions"]:
            badge_color = "#FF007F" if sug["priority"] == "HIGH" else ("#FEE440" if sug["priority"] == "MEDIUM" else "#00F5D4")
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {badge_color}; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 15px; color: #FFFFFF;">{sug['icon']} {sug['title']}</span>
                    <span style="font-size: 10px; font-weight: 800; background: rgba(0,0,0,0.5); color: {badge_color}; padding: 2px 8px; border-radius: 8px; border: 1px solid {badge_color};">{sug['priority']} PRIORITY</span>
                </div>
                <p style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;">{sug['description']}</p>
                <div style="font-size: 12px; color: #00F5D4; font-weight: 600;">🎯 Action Item: {sug['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info(rec["ethical_note"])

        # DOWNLOAD REPORT
        st.markdown("<br>", unsafe_allow_html=True)
        report_text = f"""# AI Resume Analyzer - ATS Compatibility Audit Report
Date: 2026-08-23
Candidate: {current_user['full_name']} (@{current_user['username']})
Resume Source: {res['source_name']}

==================================================
1. ATS COMPATIBILITY SUMMARY
==================================================
ATS Score: {ats['ats_score']}% ({ats['rating_grade']} - {ats['rating_label']})
Predicted Category: {pred['predicted_category']} (Confidence: {pred['confidence_score']}%)

==================================================
2. SKILL ANALYSIS
==================================================
Matched Skills ({len(skill_det['matched_skills'])}):
{', '.join(skill_det['matched_skills']) if skill_det['matched_skills'] else 'None'}

Missing Skills ({len(skill_det['missing_skills'])}):
{', '.join(skill_det['missing_skills']) if skill_det['missing_skills'] else 'None'}

Additional Skills Found:
{', '.join(skill_det['additional_skills']) if skill_det['additional_skills'] else 'None'}

==================================================
3. SCORE BREAKDOWN
==================================================
- Skill Match (50%): {ats['breakdown']['skills']['score']}%
- Keyword Match (20%): {ats['breakdown']['keywords']['score']}%
- Education & Experience (15%): {ats['breakdown']['education_experience']['score']}%
- Resume Structure (15%): {ats['breakdown']['structure']['score']}%

==================================================
4. KEY RECOMMENDATIONS
==================================================
{chr(10).join(['- ' + s['title'] + ': ' + s['action'] for s in rec['suggestions']])}

{rec['ethical_note']}
"""
        st.download_button(
            label="📥 Download Full Resume Audit Report (.txt)",
            data=report_text,
            file_name=f"AI_Resume_Audit_{current_user['username']}.txt",
            mime="text/plain",
            use_container_width=True
        )


# TAB 2: 3D SKILL GALAXY & SEMANTIC SPACE
elif app_mode == "🌌 3D Skill Constellation & Galaxy":
    st.markdown("## 🌌 3D Semantic Vector Space & Skill Galaxy")
    st.caption("Explore multidimensional TF-IDF latent space and 3D skill surface topology.")

    resume_text_to_use = st.session_state.resume_text or (st.session_state.analysis_results["resume_text"] if st.session_state.analysis_results else "")
    jd_text_to_use = st.session_state.analysis_results["jd_text"] if st.session_state.analysis_results else JD_TEMPLATES["Python Backend Developer"]

    if not resume_text_to_use:
        with open(SAMPLE_RESUME_PATHS["Sample 1: Python Backend Developer"], "r", encoding="utf-8") as f:
            resume_text_to_use = f.read()

    col_3d_1, col_3d_2 = st.columns([1.3, 1], gap="large")

    with col_3d_1:
        st.markdown("#### 🪐 3D Category Latent Space (Resume vs JD vs 12 Roles)")
        fig_3d = plot_3d_category_vector_space(resume_text_to_use, jd_text_to_use, classifier)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_3d_2:
        st.markdown("#### 🏔️ 3D Skill Topology & Depth Mesh")
        skill_res = SkillMatcher.match_skills(resume_text_to_use, jd_text_to_use)
        fig_mesh = plot_3d_capability_mesh(skill_res.get("category_breakdown", {}))
        st.plotly_chart(fig_mesh, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔭 Interactive 3D Orbit Galaxy")
    components.html(
        get_3d_skill_constellation_html(
            matched_skills=skill_res["matched_skills"],
            missing_skills=skill_res["missing_skills"],
            additional_skills=skill_res["additional_skills"]
        ),
        height=450,
        scrolling=False
    )


# TAB 3: MACHINE LEARNING STUDIO
elif app_mode == "🧪 Machine Learning Studio":
    st.markdown("## 🧪 Machine Learning Classifier & Model Hub")
    st.caption("Inspect live model evaluation metrics, test category prediction on arbitrary text, and retrain the classifier.")

    metrics = classifier.get_metrics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #00F5D4;">{metrics['accuracy']}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #00BBF9;">{metrics['precision']}%</div>
            <div class="metric-label">Weighted Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #7B2CBF;">{metrics['recall']}%</div>
            <div class="metric-label">Weighted Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #FEE440;">{metrics['f1_score']}%</div>
            <div class="metric-label">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    col_cm, col_play = st.columns([1.2, 1], gap="large")

    with col_cm:
        st.markdown("#### 📊 Confusion Matrix Heatmap")
        labels = metrics["labels"]
        cm_data = np.array(metrics["confusion_matrix"])
        
        fig_cm = px.imshow(
            cm_data,
            x=labels,
            y=labels,
            labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
            color_continuous_scale="Viridis",
            text_auto=True
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(10, 14, 26, 0.9)",
            height=440,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_play:
        st.markdown("#### 🎮 Live Category Prediction Playground")
        test_text = st.text_area(
            "Enter arbitrary resume or bio text:",
            value="Software engineer experienced with PyTorch, Transformers, LLMs, LangChain, and Python NLP models.",
            height=130
        )
        if st.button("🔮 Predict Category", use_container_width=True):
            pred_out = classifier.predict_category(test_text)
            st.success(f"Predicted Category: **{pred_out['predicted_category']}** ({pred_out['confidence_score']}%)")
            
            top_df = pd.DataFrame(pred_out["top_categories"], columns=["Category", "Probability (%)"])
            fig_bar = px.bar(
                top_df,
                x="Probability (%)",
                y="Category",
                orientation="h",
                color="Probability (%)",
                color_continuous_scale="Tealgrn"
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(10, 14, 26, 0.9)",
                height=250,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Retrain Classifier on Dataset", use_container_width=True):
            with st.spinner("Retraining model..."):
                new_metrics = classifier.train_and_save()
                st.success(f"Model successfully retrained! New Accuracy: {new_metrics['accuracy']}%")


# TAB 4: ATS GUIDE & TIPS
elif app_mode == "📖 ATS Optimization Guide":
    st.markdown("## 📖 ATS Optimization & Resume Guide")
    st.markdown("""
    ### How Modern Applicant Tracking Systems (ATS) Work
    Applicant Tracking Systems scan incoming resumes for relevant keywords, technical proficiencies, job titles, and educational background before human recruiters review them.

    ---

    ### 🎯 Top 5 Golden Rules for ATS-Friendly Resumes

    1. **Use Standard Section Headers**
       - Use clear titles like `Work Experience`, `Technical Skills`, `Education`, and `Projects`. Avoid unconventional names like *"Where I've Been"* or *"My Passions"*.

    2. **Quantify Your Achievements (The STAR Method)**
       - Don't just list responsibilities. Showcase outcomes with numbers:
         - ❌ *"Responsible for improving web page load times."*
         - ✅ *"Reduced initial page load latency by 42% through code splitting and Redis caching, improving conversions by 15%."*

    3. **Incorporate Job Description Keywords Naturally**
       - Align your skills and terminology with the job description (e.g., `FastAPI`, `Docker`, `CI/CD`).
       - Never 'keyword-stuff' white text into margins—modern ATS parsers detect this instantly.

    4. **Maintain Clean Single or Two-Column Text Formats**
       - Avoid complex multi-layered text boxes, intricate tables, or graphic-heavy canvases that confuse standard PDF parsers.

    5. **Provide Active Links to Verified Work**
       - Include direct links to your GitHub profile, LinkedIn URL, and deployed portfolio applications.
    """)
