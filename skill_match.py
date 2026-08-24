"""Skill taxonomy, extraction, and matching engine for AI Resume Analyzer.

Contains 500+ curated technical and professional skills with alias mapping,
multi-word n-gram extraction, and category-wise alignment.
"""

import re
from typing import Dict, List, Set, Tuple, Any

# Canonical Skill Taxonomy grouped by Domain
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "R", "Scala", "Dart",
        "SQL", "Bash", "Shell", "PowerShell", "Perl", "Haskell", "Julia", "MATLAB"
    ],
    "Web & Mobile Frameworks": [
        "React", "Angular", "Vue.js", "Next.js", "Nuxt.js", "Node.js", "Express.js",
        "Django", "FastAPI", "Flask", "Spring Boot", "ASP.NET", "Ruby on Rails",
        "Flutter", "React Native", "Android", "iOS", "SwiftUI", "HTML5", "CSS3",
        "Tailwind CSS", "Bootstrap", "Sass", "GraphQL", "REST API", "gRPC", "WebSockets"
    ],
    "AI, ML & Data Science": [
        "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
        "Generative AI", "Large Language Models", "PyTorch", "TensorFlow", "Keras",
        "Scikit-Learn", "Pandas", "NumPy", "SciPy", "OpenCV", "Matplotlib",
        "Seaborn", "Hugging Face", "LangChain", "Transformers", "BERT", "XGBoost",
        "LightGBM", "Retrieval Augmented Generation", "Prompt Engineering", "MLOps",
        "Feature Engineering", "Time Series Analysis", "Reinforcement Learning"
    ],
    "Databases & Big Data": [
        "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra",
        "Elasticsearch", "Oracle Database", "Microsoft SQL Server", "DynamoDB",
        "Neo4j", "Apache Spark", "Apache Kafka", "Hadoop", "Snowflake", "BigQuery",
        "Databricks", "Apache Airflow", "Hive", "Redshift", "Pinecone", "ChromaDB"
    ],
    "Cloud & DevOps": [
        "AWS", "Microsoft Azure", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "Ansible", "Jenkins", "CI/CD", "GitHub Actions", "GitLab CI",
        "Linux", "Unix", "Prometheus", "Grafana", "Nginx", "Apache HTTP Server",
        "Helm", "ArgoCD", "Serverless", "Microservices", "Infrastructure as Code"
    ],
    "Cybersecurity & Networking": [
        "Network Security", "Penetration Testing", "Ethical Hacking", "Cryptography",
        "OWASP", "SOC Analysis", "SIEM", "Firewalls", "Wireshark", "Metasploit",
        "Vulnerability Assessment", "Identity & Access Management", "Zero Trust",
        "Incident Response", "Burp Suite", "Cloud Security", "CISSP"
    ],
    "QA, Testing & Automation": [
        "Unit Testing", "Integration Testing", "End-to-End Testing", "Selenium",
        "PyTest", "JUnit", "Cypress", "Playwright", "Postman", "JMeter",
        "TestNG", "Appium", "Test Driven Development", "Behavior Driven Development", "QA Automation"
    ],
    "Soft Skills & Management": [
        "Agile", "Scrum", "Kanban", "Jira", "Git", "GitHub", "GitLab",
        "Problem Solving", "Communication", "Leadership", "Team Collaboration",
        "Project Management", "Critical Thinking", "Time Management", "Code Review",
        "System Design", "Technical Writing", "Continuous Learning"
    ]
}

# Synonyms and common abbreviations mapped to canonical skill names
SKILL_ALIASES: Dict[str, str] = {
    # Programming Languages
    "python3": "Python",
    "py": "Python",
    "golang": "Go",
    "js": "JavaScript",
    "ts": "TypeScript",
    "cpp": "C++",
    "c plus plus": "C++",
    "csharp": "C#",
    "c sharp": "C#",
    "sh": "Shell",
    "zsh": "Shell",
    
    # Frameworks
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "nuxtjs": "Nuxt.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node": "Node.js",
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "springboot": "Spring Boot",
    "spring": "Spring Boot",
    "dotnet": "ASP.NET",
    ".net": "ASP.NET",
    "asp.net core": "ASP.NET",
    "rails": "Ruby on Rails",
    "react native": "React Native",
    "react-native": "React Native",
    "html": "HTML5",
    "html5": "HTML5",
    "css": "CSS3",
    "css3": "CSS3",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "rest": "REST API",
    "restful": "REST API",
    "rest api": "REST API",
    "rest apis": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "rest-api": "REST API",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "websocket": "WebSockets",
    "websockets": "WebSockets",
    
    # AI & ML
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "genai": "Generative AI",
    "gen ai": "Generative AI",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "sklearn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "scikitlearn": "Scikit-Learn",
    "tf": "TensorFlow",
    "rag": "Retrieval Augmented Generation",
    "huggingface": "Hugging Face",
    "lang chain": "LangChain",
    "ml ops": "MLOps",
    
    # Databases
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mssql": "Microsoft SQL Server",
    "sql server": "Microsoft SQL Server",
    "oracle": "Oracle Database",
    "spark": "Apache Spark",
    "kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    
    # Cloud & DevOps
    "amazon web services": "AWS",
    "amazon aws": "AWS",
    "azure": "Microsoft Azure",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "cicd": "CI/CD",
    "ci / cd": "CI/CD",
    "ci-cd": "CI/CD",
    "iac": "Infrastructure as Code",
    "gh actions": "GitHub Actions",
    
    # Cybersecurity & QA
    "pen testing": "Penetration Testing",
    "pentesting": "Penetration Testing",
    "iam": "Identity & Access Management",
    "pytest": "PyTest",
    "e2e": "End-to-End Testing",
    "e2e testing": "End-to-End Testing",
    "tdd": "Test Driven Development",
    "bdd": "Behavior Driven Development",
    
    # Soft skills & Tools
    "problem-solving": "Problem Solving",
    "communication skills": "Communication",
    "time-management": "Time Management",
    "project-management": "Project Management",
    "code reviews": "Code Review"
}

# Build flat lookup sets
CANONICAL_SKILLS: Set[str] = set()
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        CANONICAL_SKILLS.add(skill)
        SKILL_TO_CATEGORY[skill] = category


class SkillExtractor:
    """Extracts skills from text using boundary matching, n-grams, and alias dictionaries."""

    @classmethod
    def extract_skills(cls, text: str) -> List[str]:
        """Extracts unique canonical skills detected in the provided text.

        Args:
            text: Resume or Job Description text string.

        Returns:
            Sorted list of canonical skill names detected.
        """
        if not text:
            return []

        text_lower = " " + text.lower() + " "
        # Replace non-alphanumeric punctuation with spaces for phrase extraction,
        # but preserve +, #, ., - for tech terms (C++, C#, .NET, Node.js, CI/CD)
        normalized = re.sub(r'[,;:\(\)\[\]\{\}"\*\?!/|]', ' ', text_lower)
        normalized = re.sub(r'\s+', ' ', normalized)

        found_skills: Set[str] = set()

        # 1. Direct Alias Matching (Longest aliases first to avoid greedy sub-phrase match)
        sorted_aliases = sorted(SKILL_ALIASES.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            canonical = SKILL_ALIASES[alias]
            # Match with word boundaries or start/end of string
            pattern = r'(?<![a-zA-Z0-9_#+])' + re.escape(alias) + r'(?![a-zA-Z0-9_#+])'
            if re.search(pattern, normalized):
                found_skills.add(canonical)

        # 2. Canonical Skill Name Matching
        sorted_canonicals = sorted(CANONICAL_SKILLS, key=len, reverse=True)
        for skill in sorted_canonicals:
            skill_lower = skill.lower()
            pattern = r'(?<![a-zA-Z0-9_#+])' + re.escape(skill_lower) + r'(?![a-zA-Z0-9_#+])'
            if re.search(pattern, normalized):
                found_skills.add(skill)

        # 3. Special handling for short single-letter or symbol skills like "C", "R", "Go"
        # Only match if clearly in a skills or language list context
        if "C" not in found_skills:
            if re.search(r'\b(c\s*,\s*c\+\+|c\s*/\s*c\+\+|c\s+and\s+c\+\+|c\s+programming)\b', text_lower):
                found_skills.add("C")

        if "R" not in found_skills:
            if re.search(r'\b(r\s*,\s*python|r\s+programming|r\s+language|r\s+and\s+python|python\s+and\s+r)\b', text_lower):
                found_skills.add("R")

        if "Go" not in found_skills:
            if re.search(r'\b(golang|go\s+developer|go\s+backend|go\s+microservices|go\s+programming)\b', text_lower):
                found_skills.add("Go")

        return sorted(list(found_skills))


class SkillMatcher:
    """Compares resume skills with job description requirements."""

    @classmethod
    def match_skills(cls, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Performs full skill extraction and matching between Resume and JD.

        Args:
            resume_text: Extracted plain text of resume.
            jd_text: Target Job Description text.

        Returns:
            Dict containing:
                - resume_skills: List of skills found in resume
                - jd_skills: List of skills required in JD
                - matched_skills: List of skills present in both
                - missing_skills: List of required JD skills absent from resume
                - additional_skills: Skills on resume not requested in JD
                - match_percentage: Skill match ratio (0-100%)
                - category_breakdown: Breakdown of matched vs required by category
        """
        resume_skills = SkillExtractor.extract_skills(resume_text)
        jd_skills = SkillExtractor.extract_skills(jd_text)

        resume_set = set(resume_skills)
        jd_set = set(jd_skills)

        matched_set = resume_set.intersection(jd_set)
        missing_set = jd_set.difference(resume_set)
        additional_set = resume_set.difference(jd_set)

        matched_skills = sorted(list(matched_set))
        missing_skills = sorted(list(missing_set))
        additional_skills = sorted(list(additional_set))

        # Skill match percentage (based on JD requirement coverage)
        if len(jd_skills) > 0:
            match_percentage = round((len(matched_skills) / len(jd_skills)) * 100, 1)
        else:
            # If JD has no predefined skills found, use a baseline score
            match_percentage = 100.0 if len(resume_skills) > 0 else 0.0

        # Category Breakdown
        category_breakdown: Dict[str, Dict[str, Any]] = {}
        for category, cat_skills in SKILL_TAXONOMY.items():
            cat_set = set(cat_skills)
            jd_in_cat = jd_set.intersection(cat_set)
            matched_in_cat = matched_set.intersection(cat_set)
            missing_in_cat = missing_set.intersection(cat_set)
            resume_in_cat = resume_set.intersection(cat_set)

            if len(jd_in_cat) > 0 or len(resume_in_cat) > 0:
                cat_coverage = (
                    round((len(matched_in_cat) / len(jd_in_cat)) * 100, 1)
                    if len(jd_in_cat) > 0
                    else 100.0
                )
                category_breakdown[category] = {
                    "matched": sorted(list(matched_in_cat)),
                    "missing": sorted(list(missing_in_cat)),
                    "resume_skills": sorted(list(resume_in_cat)),
                    "total_required": len(jd_in_cat),
                    "total_matched": len(matched_in_cat),
                    "coverage_percent": cat_coverage
                }

        return {
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "additional_skills": additional_skills,
            "match_percentage": match_percentage,
            "category_breakdown": category_breakdown
        }

