"""Recommendation Engine for AI Resume Analyzer.

Generates tailored, actionable resume improvement suggestions based on:
1. Missing technical skills & relevant project ideas
2. Quantifiable metrics and action verb enhancements (STAR method)
3. Resume structure, section completeness, and contact links
4. Target job description keyword alignment
"""

from typing import Dict, List, Any


class RecommendationEngine:
    """Generates context-aware, ethical resume improvement suggestions."""

    @classmethod
    def generate_recommendations(
        cls,
        ats_data: Dict[str, Any],
        resume_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates structured suggestions grouped by category and priority.

        Args:
            ats_data: Full ATS scoring output from ATSScoreCalculator.calculate_ats_score.
            resume_metadata: Full metadata from ResumeParser.parse.

        Returns:
            Dict containing lists of prioritized suggestions and actionable checklists.
        """
        suggestions: List[Dict[str, str]] = []
        action_checklist: List[str] = []

        skill_details = ats_data.get("skill_details", {})
        missing_skills = skill_details.get("missing_skills", [])
        matched_skills = skill_details.get("matched_skills", [])
        
        kw_details = ats_data.get("keyword_details", {})
        top_jd_kw = kw_details.get("top_jd_keywords", [])
        matched_kw = kw_details.get("matched_keywords", [])
        missing_kw = [k for k in top_jd_kw if k not in matched_kw][:6]

        edu_details = ats_data.get("education_details", {})
        action_verb_count = edu_details.get("action_verb_count", 0)
        has_metrics = edu_details.get("has_metrics", False)

        struct_details = ats_data.get("structure_details", {})
        word_count = struct_details.get("word_count", 0)
        contacts = struct_details.get("contacts", {})
        sec_status = resume_metadata.get("section_status", {})

        # 1. Missing Skills Suggestions (High Priority)
        if missing_skills:
            top_missing = missing_skills[:4]
            missing_str = ", ".join(top_missing)
            suggestions.append({
                "category": "Skills Alignment",
                "priority": "HIGH",
                "icon": "⚡",
                "title": f"Address Missing Core Skills: {missing_str}",
                "description": f"The job description specifically requests **{missing_str}**. If you have working experience or have built projects with these technologies, ensure they are explicitly highlighted in your Skills and Project bullet points.",
                "action": f"Incorporate {missing_str} where genuinely applicable."
            })
            action_checklist.append(f"Add projects demonstrating {missing_str} if applicable.")
        else:
            suggestions.append({
                "category": "Skills Alignment",
                "priority": "LOW",
                "icon": "✨",
                "title": "Superb Skill Coverage",
                "description": "Your resume covers all major technical skills required by this job description!",
                "action": "Maintain clear, concise descriptions of how you utilized these skills."
            })

        # 2. Measurable Metrics & Action Verbs (Medium/High Priority)
        if not has_metrics or action_verb_count < 5:
            suggestions.append({
                "category": "Impact & Metrics",
                "priority": "HIGH" if not has_metrics else "MEDIUM",
                "icon": "📈",
                "title": "Quantify Achievements with Data & STAR Method",
                "description": "Recruiters and ATS algorithms favor resumes that demonstrate quantifiable business impact (e.g., *'Reduced API response latency by 35%'* or *'Scaled system to handle 50k+ daily users'*).",
                "action": "Use the Situation-Task-Action-Result (STAR) format with percentages, numbers, or time savings in your bullet points."
            })
            action_checklist.append("Add measurable metrics (% improvements, user counts, latency reductions) to project descriptions.")

        # 3. Contact & Profile Completeness
        missing_profiles = []
        if not contacts.get("github"):
            missing_profiles.append("GitHub profile")
        if not contacts.get("linkedin"):
            missing_profiles.append("LinkedIn URL")
        if not contacts.get("portfolio"):
            missing_profiles.append("Portfolio / Live Demo link")

        if missing_profiles:
            profiles_str = " and ".join(missing_profiles)
            suggestions.append({
                "category": "Online Presence",
                "priority": "MEDIUM",
                "icon": "🔗",
                "title": f"Add {profiles_str}",
                "description": f"Providing active links to your **{profiles_str}** allows technical recruiters and hiring managers to quickly verify your open-source contributions and portfolio code.",
                "action": f"Include clean clickable hyperlinks to your {profiles_str} in your contact header."
            })
            action_checklist.append(f"Add {profiles_str} to your resume header.")

        # 4. Resume Length & Formatting Health
        if word_count < 250:
            suggestions.append({
                "category": "Resume Structure",
                "priority": "HIGH",
                "icon": "📝",
                "title": "Expand Resume Content & Project Details",
                "description": f"Your resume contains only ~{word_count} words. An effective 1-page technical resume typically contains between 350 and 700 words detailing technical stack, architecture, and responsibilities.",
                "action": "Elaborate on your key projects, technical challenges solved, and specific tools used."
            })
            action_checklist.append("Expand project descriptions with architecture, technical challenges, and solutions.")
        elif word_count > 1200:
            suggestions.append({
                "category": "Resume Structure",
                "priority": "MEDIUM",
                "icon": "✂️",
                "title": "Streamline Resume for Better Conciseness",
                "description": f"Your resume has ~{word_count} words, which may span 2-3 pages. Consider tightening your bullet points to focus on the most impactful and recent accomplishments.",
                "action": "Condense older or less relevant roles to keep the resume crisp and focused."
            })
            action_checklist.append("Condense bullet points to maintain high readability.")

        # 5. Missing Sections Check
        if not sec_status.get("has_projects_section", True):
            suggestions.append({
                "category": "Resume Sections",
                "priority": "HIGH",
                "icon": "🛠️",
                "title": "Add a Dedicated 'Projects' Section",
                "description": "A dedicated Technical Projects section is critical for students, freshers, and developers to showcase practical engineering ability.",
                "action": "Create a 'Projects' section highlighting 2-3 substantial full-stack or data projects."
            })
            action_checklist.append("Add a dedicated Projects section with tech stacks and GitHub repository links.")

        # 6. Keyword Alignment
        if missing_kw:
            kw_str = ", ".join(missing_kw)
            suggestions.append({
                "category": "Keyword Optimization",
                "priority": "MEDIUM",
                "icon": "🎯",
                "title": f"Target Important JD Terminology: {kw_str}",
                "description": f"The job description frequently emphasizes terms like **{kw_str}**. Review your project summaries to naturally incorporate these keywords where true.",
                "action": f"Align terminology with {kw_str} in context."
            })
            action_checklist.append(f"Incorporate target keywords ({kw_str}) in context.")

        # Ethical guideline reminder (as required by PRD Section 17 & 5.2)
        ethical_note = "🛡️ **Ethical Reminder**: Always represent your qualifications authentically. Never add skills, libraries, or experiences that you do not genuinely possess or cannot explain during a technical interview."

        return {
            "suggestions": suggestions,
            "action_checklist": action_checklist,
            "ethical_note": ethical_note
        }

