"""Generates sample PDF resumes for testing using standard compliant PDF syntax."""

import os

RESUMES_DIR = os.path.join(os.path.dirname(__file__), "resumes")
os.makedirs(RESUMES_DIR, exist_ok=True)

SAMPLE_RESUMES = {
    "sample_python_developer": """ALEXANDER RIVERS
Email: alex.rivers@example.com | Phone: (555) 019-2834
LinkedIn: linkedin.com/in/alex-rivers-dev | GitHub: github.com/alexrivers-code

PROFESSIONAL SUMMARY
Dynamic Python Backend Developer with 3+ years of experience building high-throughput microservices, RESTful APIs, and database architectures using Python, Django, FastAPI, PostgreSQL, and Docker. Proven track record of improving system latency by 35% and automating CI/CD deployments.

TECHNICAL SKILLS
- Programming Languages: Python, JavaScript, TypeScript, SQL, Bash
- Web Frameworks: Django, FastAPI, Flask, REST API, WebSockets
- Databases: PostgreSQL, MySQL, Redis, MongoDB
- Cloud and DevOps: Docker, AWS, CI/CD, Git, GitHub Actions, Linux
- Testing and Tools: PyTest, Postman, Celery, Jira, Agile, Scrum

WORK EXPERIENCE
Python Developer | TechCorp Solutions
July 2022 - Present
- Architected scalable REST APIs using FastAPI and PostgreSQL, serving 120,000+ daily active users.
- Reduced database query response times by 42% through query optimization and Redis caching layers.
- Built automated CI/CD deployment pipelines using GitHub Actions and Docker, reducing release cycle time by 50%.
- Integrated Stripe payment gateways and third-party webhooks with Celery asynchronous task queues.

Junior Software Engineer | CloudNova Inc.
August 2020 - June 2022
- Developed backend modules in Django and Flask for enterprise inventory management systems.
- Wrote comprehensive unit tests using PyTest, achieving 94% test code coverage across 15 core services.
- Collaborated in an Agile Scrum environment with cross-functional teams of engineers and product managers.

KEY PROJECTS
- Distributed E-Commerce Microservices Engine (Python, FastAPI, Redis, Docker, PostgreSQL)
- Intelligent Log Aggregator and Alerting System (Python, Elasticsearch, Docker, AWS)

EDUCATION
Bachelor of Technology in Computer Science and Engineering
State University of Technology | 2016 - 2020 | GPA: 3.8 / 4.0

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate
- Professional Python Programmer Certification
""",

    "sample_data_scientist": """PRIYA SHARMA
Email: priya.sharma@example.com | Phone: (555) 789-4321
LinkedIn: linkedin.com/in/priya-sharma-ds | GitHub: github.com/priyasharma-ml

PROFESSIONAL SUMMARY
Results-oriented Data Scientist with expertise in Machine Learning, Natural Language Processing (NLP), Deep Learning, and predictive modeling. Proficient in Python, PyTorch, Scikit-Learn, Pandas, NumPy, and SQL. Spearheaded ML solutions that boosted customer retention by 28%.

TECHNICAL SKILLS
- Machine Learning: Supervised Learning, Unsupervised Learning, Random Forest, XGBoost, LightGBM, Clustering
- Deep Learning and NLP: PyTorch, TensorFlow, Transformers, BERT, LLMs, Hugging Face, OpenCV
- Data Science Tools: Python, R, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Jupyter
- Databases and Big Data: SQL, PostgreSQL, Snowflake, Apache Spark
- MLOps and Deployment: Docker, FastAPI, MLflow, AWS, Git

WORK EXPERIENCE
Lead Data Scientist | DataSphere Analytics
March 2021 - Present
- Developed predictive churn models using XGBoost and PyTorch, increasing customer retention by 28%.
- Designed end-to-end NLP sentiment classification pipeline using BERT and Hugging Face on 500,000+ customer reviews.
- Deployed ML inference microservices with FastAPI and Docker on AWS, reducing latency to <45ms.
- Collaborated with business stakeholders to translate analytical findings into strategic executive dashboards.

Data Analyst | Insight Metrics Corp
January 2019 - February 2021
- Built automated SQL queries and ETL pipelines in Snowflake, reducing weekly reporting time by 15 hours.
- Created executive Tableau dashboards to track multi-channel marketing campaigns and ROI metrics.

KEY PROJECTS
- Clinical Text Summarization with Transformers (PyTorch, Hugging Face, Streamlit, Python)
- Real-Time Credit Card Fraud Detection (Scikit-Learn, LightGBM, Kafka, Docker)

EDUCATION
Master of Science in Data Science and Machine Learning
University of Engineering and Tech | 2017 - 2019
Bachelor of Science in Mathematics and Statistics | 2013 - 2017
""",

    "sample_web_developer": """JORDAN BLAKE
Email: jordan.blake@example.com | Phone: (555) 321-9876
LinkedIn: linkedin.com/in/jordan-blake-web | GitHub: github.com/jordanblake-dev

PROFESSIONAL SUMMARY
Full Stack Web Developer with 4+ years of hands-on experience building responsive, accessible, and high-performance web applications using React, Next.js, TypeScript, Node.js, Tailwind CSS, and MongoDB.

TECHNICAL SKILLS
- Frontend: React, Next.js, JavaScript, TypeScript, HTML5, CSS3, Tailwind CSS, Redux, Sass
- Backend: Node.js, Express.js, GraphQL, REST API, WebSockets
- Databases: MongoDB, PostgreSQL, Firebase, Redis
- Tools and Cloud: Git, GitHub Actions, Docker, AWS, Figma, Jest, Cypress

WORK EXPERIENCE
Senior Full Stack Developer | PixelWave Studios
September 2021 - Present
- Built modular SaaS application with Next.js, React, Tailwind CSS, and TypeScript serving 80,000+ monthly users.
- Improved frontend Core Web Vitals (LCP and INP) by 45% through code-splitting and asset optimization.
- Implemented secure JWT authentication and Stripe subscription billing using Node.js and Express.

Frontend Developer | Horizon Web Lab
June 2019 - August 2021
- Designed 25+ pixel-perfect, responsive client websites from Figma mockups with HTML5, CSS3, and React.
- Integrated GraphQL APIs and Apollo Client for real-time collaborative workspace tools.
- Wrote automated unit and end-to-end test suites using Jest and Cypress.

KEY PROJECTS
- AI Collaborative Document Workspace (React, Next.js, Node.js, WebSockets, MongoDB, Tailwind)
- Crypto Portfolio and Analytics Dashboard (React, TypeScript, Tailwind CSS, REST APIs)

EDUCATION
Bachelor of Science in Computer Science
Metro State University | 2015 - 2019
"""
}


def create_simple_pdf(text: str, filepath: str):
    """Writes a valid uncompressed PDF file containing plain text."""
    lines = text.strip().splitlines()
    stream_lines = ["BT", "/F1 10 Tf", "30 760 Td", "13 TL"]
    for line in lines[:55]:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({safe}) '")
    stream_lines.append("ET")
    stream_data = "\n".join(stream_lines)
    stream_len = len(stream_data)

    pdf_body = f"""%PDF-1.4
1 0 obj
<<
  /Type /Catalog
  /Pages 2 0 R
>>
endobj
2 0 obj
<<
  /Type /Pages
  /Kids [3 0 R]
  /Count 1
>>
endobj
3 0 obj
<<
  /Type /Page
  /Parent 2 0 R
  /MediaBox [0 0 612 792]
  /Contents 4 0 R
  /Resources <<
    /Font <<
      /F1 <<
        /Type /Font
        /Subtype /Type1
        /BaseFont /Helvetica
      >>
    >>
  >>
>>
endobj
4 0 obj
<<
  /Length {stream_len}
>>
stream
{stream_data}
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000306 00000 n 
trailer
<<
  /Size 5
  /Root 1 0 R
>>
startxref
{306 + stream_len + 30}
%%EOF
"""
    with open(filepath, "w", encoding="latin-1") as f:
        f.write(pdf_body)

    # Also save .txt version for raw text loading
    txt_path = filepath.replace(".pdf", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


if __name__ == "__main__":
    for name, content in SAMPLE_RESUMES.items():
        pdf_path = os.path.join(RESUMES_DIR, f"{name}.pdf")
        create_simple_pdf(content, pdf_path)
        print(f"Created: {pdf_path}")

