"""AI resume drafting utilities for generating an optimized resume draft from a parsed resume and JD."""

from __future__ import annotations

import io
import re
import textwrap
from typing import List

from PIL import Image, ImageDraw, ImageFont

from skill_match import SkillMatcher
from utils.text_processing import extract_contact_info, segment_resume_sections


def _extract_name(resume_text: str) -> str:
    """Best-effort extraction of the candidate name from the resume header."""
    if not resume_text:
        return "Candidate"

    for line in resume_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if re.search(r'@|linkedin|github|http|phone|tel', candidate, flags=re.IGNORECASE):
            continue
        if len(candidate.split()) <= 3 and not candidate.endswith(":"):
            return candidate
    return "Candidate"


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        clean = item.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def build_ai_resume_draft(resume_text: str, jd_text: str) -> str:
    """Build a polished, ATS-aware resume draft using the extracted resume and target JD."""
    if not resume_text or not resume_text.strip():
        return ""

    contacts = extract_contact_info(resume_text)
    sections = segment_resume_sections(resume_text)
    matching = SkillMatcher.match_skills(resume_text, jd_text)

    matched_skills = matching.get("matched_skills", [])
    missing_skills = matching.get("missing_skills", [])
    additional_skills = matching.get("additional_skills", [])
    canonical_skills = _dedupe(matched_skills + additional_skills + missing_skills)
    skills_to_show = canonical_skills[:15] if canonical_skills else ["Python", "SQL", "Problem Solving", "Communication"]

    summary_block = sections.get("summary", "").strip()
    if not summary_block:
        summary_block = (
            f"Results-driven professional with experience in {', '.join(skills_to_show[:5])} and a track record of "
            "building reliable, scalable solutions that improve business outcomes."
        )

    experience_block = sections.get("experience", "").strip()
    if not experience_block:
        experience_block = (
            "Software Engineer\n"
            "- Designed and developed applications using " + ", ".join(skills_to_show[:5]) + ".\n"
            "- Improved system performance, reliability, and maintainability through clean architecture and automation.\n"
            "- Collaborated with cross-functional teams to deliver user-focused solutions and measurable business impact."
        )

    projects_block = sections.get("projects", "").strip()
    if not projects_block:
        projects_block = (
            "Key Projects\n"
            "- Built a data-driven application using " + ", ".join(skills_to_show[:4]) + " to streamline workflows and improve efficiency.\n"
            "- Delivered a production-ready solution with monitoring, testing, and deployment automation."
        )

    education_block = sections.get("education", "").strip()
    if not education_block:
        education_block = "Bachelor's Degree / Relevant Technical Education"

    header_lines = [
        _extract_name(resume_text),
        " | ".join(part for part in [
            contacts.get("email"),
            contacts.get("phone"),
            f"LinkedIn: {contacts.get('linkedin')}" if contacts.get("linkedin") else None,
            f"GitHub: {contacts.get('github')}" if contacts.get("github") else None,
            f"Portfolio: {contacts.get('portfolio')}" if contacts.get("portfolio") else None,
        ] if part),
    ]

    draft_lines = [
        *header_lines,
        "",
        "PROFESSIONAL SUMMARY",
        summary_block,
        "",
        "TECHNICAL SKILLS",
        ", ".join(skills_to_show),
        "",
        "PROFESSIONAL EXPERIENCE",
        experience_block,
        "",
        "PROJECTS",
        projects_block,
        "",
        "EDUCATION",
        education_block,
    ]

    if missing_skills:
        draft_lines.extend([
            "",
            "TARGETED IMPROVEMENTS",
            "Prioritize continued development in: " + ", ".join(missing_skills[:5]) + ".",
        ])

    return "\n".join(draft_lines).strip()


def build_ai_resume_pdf(resume_text: str, jd_text: str) -> bytes:
    """Render the AI-generated resume draft as a PDF-ready image document."""
    draft = build_ai_resume_draft(resume_text, jd_text)
    if not draft:
        return b""

    base_width = 1200
    margin = 60
    line_spacing = 24
    title_height = 90
    wrap_width = 95

    wrapped_lines = []
    for para in draft.splitlines():
        if not para.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(para, width=wrap_width, break_long_words=False, replace_whitespace=False))

    line_count = max(1, len(wrapped_lines) + 4)
    height = max(900, title_height + (line_count * line_spacing) + margin)

    img = Image.new("RGB", (base_width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.rounded_rectangle([20, 20, base_width - 20, height - 20], radius=18, fill=(248, 250, 252), outline=(148, 163, 184), width=2)
    draw.text((margin, 28), "AI Resume Draft", fill=(15, 23, 42), font=title_font)

    current_y = title_height
    for line in wrapped_lines:
        draw.text((margin, current_y), line, fill=(15, 23, 42), font=body_font)
        current_y += line_spacing

    pdf_buffer = io.BytesIO()
    img.save(pdf_buffer, format="PDF", resolution=200.0)
    return pdf_buffer.getvalue()
