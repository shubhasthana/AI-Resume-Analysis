"""Text processing and NLP utility functions for AI Resume Analyzer.

Provides text cleaning, keyword extraction, contact information detection,
section segmentation, and tokenization.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Any

# Standard English stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves", "also", "using", "used", "work",
    "worked", "working", "responsible", "including", "across", "within", "per", "via", "well", "etc",
    "year", "years", "month", "months", "day", "days", "experience", "experienced", "role", "roles"
}

# Strong action verbs for ATS scoring and recommendation
ACTION_VERBS = [
    "accelerated", "accomplished", "achieved", "acquired", "adapted", "administered", "advanced",
    "analyzed", "architected", "automated", "built", "championed", "collaborated", "constructed",
    "created", "customized", "debugged", "decreased", "delivered", "deployed", "designed", "developed",
    "devised", "directed", "doubled", "engineered", "enhanced", "established", "evaluated", "executed",
    "expanded", "expedited", "fabricated", "formulated", "founded", "generated", "guided", "implemented",
    "improved", "increased", "initiated", "innovated", "installed", "instituted", "integrated", "invented",
    "launched", "led", "leveraged", "maintained", "managed", "maximized", "mentored", "migrated", "minimized",
    "modernized", "negotiated", "optimized", "orchestrated", "organized", "overhauled", "oversaw",
    "partnered", "pioneered", "planned", "programmed", "published", "re-engineered", "reduced", "refactored",
    "refined", "reformed", "remodeled", "resolved", "restructured", "revamped", "scaled", "simplified",
    "spearheaded", "standardized", "streamlined", "strengthened", "structured", "supervised", "synthesized",
    "trained", "transformed", "troubleshot", "upgraded", "validated", "yielded"
]


def clean_text(text: str) -> str:
    """Cleans text for NLP and ML vectorization.
    
    Removes URLs, emails, special punctuation, and normalizes whitespace.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    cleaned = text.lower()
    
    # Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    
    # Remove email addresses
    cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', cleaned)
    
    # Remove phone numbers
    cleaned = re.sub(r'\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,9}', ' ', cleaned)
    
    # Replace slashes and hyphens with spaces for compound terms
    cleaned = re.sub(r'[/\\_]', ' ', cleaned)
    
    # Remove special punctuation but keep letters, numbers, and basic spaces
    cleaned = re.sub(r'[^\w\s\+#]', ' ', cleaned)
    
    # Normalize multiple whitespace characters
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def extract_tokens(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenizes text into words, optionally filtering out stopwords."""
    cleaned = clean_text(text)
    tokens = [tok for tok in cleaned.split() if len(tok) > 1]
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def extract_keywords(text: str, top_n: int = 25) -> List[Tuple[str, int]]:
    """Extracts top keywords with their frequency from the text."""
    tokens = extract_tokens(text, remove_stopwords=True)
    counter = Counter(tokens)
    return counter.most_common(top_n)


def extract_contact_info(text: str) -> Dict[str, Any]:
    """Extracts contact details such as email, phone, LinkedIn, and GitHub."""
    contacts = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "portfolio": None
    }
    
    if not text:
        return contacts
    
    # Email detection
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    if emails:
        contacts["email"] = emails[0]
        
    # Phone number detection (various formats)
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    if phones:
        contacts["phone"] = phones[0]
        
    # LinkedIn
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|profile)/[a-zA-Z0-9_-]+'
    linkedins = re.findall(linkedin_pattern, text, re.IGNORECASE)
    if linkedins:
        contacts["linkedin"] = linkedins[0]
    elif "linkedin.com" in text.lower() or "linkedin" in text.lower():
        contacts["linkedin"] = "Mentioned"
        
    # GitHub
    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+'
    githubs = re.findall(github_pattern, text, re.IGNORECASE)
    if githubs:
        contacts["github"] = githubs[0]
    elif "github.com" in text.lower() or "github" in text.lower():
        contacts["github"] = "Mentioned"
        
    # Portfolio / Website
    portfolio_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.(?:io|me|dev|com|org|tech)/?[a-zA-Z0-9_-]*'
    portfolios = re.findall(portfolio_pattern, text, re.IGNORECASE)
    portfolios = [p for p in portfolios if "linkedin" not in p and "github" not in p and "@" not in p]
    if portfolios:
        contacts["portfolio"] = portfolios[0]
        
    return contacts


def segment_resume_sections(text: str) -> Dict[str, str]:
    """Segments resume text into common standard sections.
    
    Detects: Summary, Skills, Experience, Education, Projects, Certifications.
    """
    sections = {
        "summary": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": ""
    }
    
    if not text:
        return sections

    lines = text.splitlines()
    current_section = "summary"
    
    section_headers = {
        "summary": ["summary", "objective", "professional summary", "about me", "profile", "overview"],
        "skills": ["skills", "technical skills", "skills & tools", "core competencies", "technologies", "proficiencies"],
        "experience": ["experience", "work experience", "professional experience", "employment history", "work history", "internships"],
        "education": ["education", "academic background", "qualifications", "academics", "education & training"],
        "projects": ["projects", "personal projects", "academic projects", "key projects", "notable projects"],
        "certifications": ["certifications", "certificates", "licenses", "awards", "achievements", "honors"]
    }
    
    collected: Dict[str, List[str]] = {sec: [] for sec in sections}
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        lower_line = stripped.lower()
        cleaned_header = re.sub(r'[^a-z\s]', '', lower_line).strip()
        
        # Check if line matches a standard section header
        matched_header = False
        for sec, headers in section_headers.items():
            if cleaned_header in headers or any(stripped.lower().startswith(h + ":") for h in headers):
                current_section = sec
                matched_header = True
                break
        
        if not matched_header:
            collected[current_section].append(stripped)
            
    for sec in sections:
        sections[sec] = "\n".join(collected[sec])
        
    return sections


def calculate_text_metrics(text: str) -> Dict[str, Any]:
    """Calculates word count, sentence count, reading time, and bullet points."""
    if not text:
        return {
            "word_count": 0,
            "char_count": 0,
            "reading_time_min": 0,
            "bullet_count": 0,
            "action_verb_count": 0
        }
        
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    reading_time = round(word_count / 200, 1) # Avg reading speed 200 wpm
    
    # Bullet points detection
    bullet_pattern = r'(?:^|\n)\s*([•\-\*\u2022\u2023\u25E6\u2043\u2219]|\d+\.)\s+'
    bullets = re.findall(bullet_pattern, text)
    bullet_count = len(bullets)
    
    # Action verbs detected
    text_lower = text.lower()
    action_verb_count = sum(1 for verb in ACTION_VERBS if re.search(r'\b' + re.escape(verb) + r'\b', text_lower))
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "reading_time_min": reading_time,
        "bullet_count": bullet_count,
        "action_verb_count": action_verb_count
    }

