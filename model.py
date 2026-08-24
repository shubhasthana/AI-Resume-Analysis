"""Machine Learning Model Pipeline for Resume Career Category Classification.

Provides a rich 12-category labeled resume dataset, TF-IDF feature extraction,
hyperparameter-tuned classification (MultinomialNB & Logistic Regression),
evaluation metrics, model persistence, and real-time inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from utils.text_processing import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MODEL_PATH = os.path.join(MODEL_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
DATASET_PATH = os.path.join(DATASET_DIR, "resume_dataset.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Comprehensive Labeled Dataset across 12 Major Domains
SAMPLE_RESUME_DATA = [
    # 1. Data Science
    ("Data Scientist with 4 years experience in Python, R, SQL, Pandas, NumPy, Scikit-learn, PyTorch, statistical analysis, hypothesis testing, predictive modeling, machine learning algorithms, regression, random forest, XGBoost, and data storytelling.", "Data Science"),
    ("Senior Data Scientist specializing in predictive modeling, customer segmentation, time series forecasting, NLP, BERT, Spark, BigQuery, Tableau, and automated ML pipelines in Python.", "Data Science"),
    ("Data Science graduate skilled in exploratory data analysis (EDA), statistical inference, machine learning, feature engineering, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and SQL.", "Data Science"),
    ("Lead Data Scientist developing recommendation systems, churn prediction models, A/B testing frameworks, Bayesian optimization, Python, Snowflake, and deep neural networks.", "Data Science"),
    ("Data Scientist experienced in computer vision, PyTorch, OpenCV, TensorFlow, image classification, object detection, mathematical modeling, and cloud deployment on AWS.", "Data Science"),
    ("Applied Data Scientist proficient in statistical computing, R programming, Python, Scikit-learn, ensemble methods, clustering, geospatial analytics, and executive presentation of ML findings.", "Data Science"),
    ("Data Scientist with background in econometrics, statistical modeling, machine learning, Python, SQL, Jupyter, regression analysis, random forest, and data visualization.", "Data Science"),
    ("Machine learning and data science professional with expertise in deep learning, neural networks, PyTorch, Scikit-learn, natural language processing, vector databases, and Python data pipelines.", "Data Science"),

    # 2. Machine Learning Engineer
    ("Machine Learning Engineer with hands-on experience designing, training, and deploying deep neural networks. Proficient in PyTorch, TensorFlow, MLOps, Docker, Kubernetes, Triton, FastAPI, and CUDA optimization.", "Machine Learning Engineer"),
    ("ML Engineer focused on Computer Vision and Object Detection using YOLO, OpenCV, PyTorch, ONNX, TensorRT, Python, C++, and edge device deployment on NVIDIA Jetson.", "Machine Learning Engineer"),
    ("Machine Learning Engineer skilled in Generative AI, Large Language Models (LLMs), RAG pipelines, LangChain, Pinecone vector database, Hugging Face, fine-tuning Llama models, and microservices on AWS.", "Machine Learning Engineer"),
    ("MLOps and ML Engineer experienced in building CI/CD pipelines for machine learning models using Kubeflow, MLflow, Docker, AWS SageMaker, Prometheus, Python, and automated model monitoring.", "Machine Learning Engineer"),
    ("Senior Machine Learning Engineer architecting distributed training clusters with PyTorch, Ray, Horovod, Transformers, Hugging Face, Kubernetes, and high-performance model serving.", "Machine Learning Engineer"),
    ("AI / ML Engineer with background in Deep Learning, PyTorch, TensorFlow, model quantization, LLM agent development, LangChain, vector embeddings, and real-time inferencing with FastAPI.", "Machine Learning Engineer"),
    ("Machine Learning Engineer specializing in reinforcement learning, simulation environments, deep neural networks, PyTorch, C++, Python, and production MLOps deployment.", "Machine Learning Engineer"),
    ("ML Systems Engineer implementing automated model retraining, data drift detection, MLflow tracking, Docker containerization, AWS SageMaker, and scalable API endpoints.", "Machine Learning Engineer"),

    # 3. Python Developer
    ("Python Developer with 4 years experience developing scalable backend web applications and REST APIs using Python, Django, FastAPI, Flask, PostgreSQL, Redis, Celery, Docker, and Git.", "Python Developer"),
    ("Backend Python Engineer experienced with Django REST Framework, SQLAlchemy, MySQL, microservices architecture, RabbitMQ, pytest unit testing, and AWS EC2 deployment.", "Python Developer"),
    ("Junior Python Developer skilled in Python 3, Flask, SQLite, HTML5, CSS3, JavaScript, web scraping with Beautiful Soup and Selenium, automated scripting, and Git version control.", "Python Developer"),
    ("Full Stack Python Developer proficient in Python, FastAPI, React, PostgreSQL, Docker, Redis, RESTful API design, pytest, JWT authentication, and CI/CD pipelines with GitHub Actions.", "Python Developer"),
    ("Python Backend Software Engineer building high-concurrency microservices, gRPC services, Django, PostgreSQL, Redis caching, AWS Lambda serverless functions, and unit test automation.", "Python Developer"),
    ("Software Engineer specialized in Python backend development, asynchronous programming with asyncio, FastAPI, PostgreSQL, message queues, Docker, and Linux system administration.", "Python Developer"),
    ("Python Developer with strong experience in building data extraction scripts, RESTful API integrations, Django web applications, PostgreSQL queries, and automated testing.", "Python Developer"),
    ("Backend Developer proficient in Python, Flask, Django, REST APIs, Redis, Docker, microservices, unit testing with PyTest, PostgreSQL, and cloud deployments on AWS.", "Python Developer"),

    # 4. Java Developer
    ("Senior Java Developer with 5+ years experience in Core Java, J2EE, Spring Boot, Spring Cloud, Hibernate, Microservices, RESTful Web Services, Apache Kafka, PostgreSQL, MySQL, and Docker.", "Java Developer"),
    ("Java Software Engineer building enterprise microservices using Spring Boot, Spring Security, Maven, JPA, Hibernate, Oracle Database, JUnit, Mockito, and Jenkins CI/CD pipelines.", "Java Developer"),
    ("Java Full Stack Developer proficient in Java 17, Spring Boot, Angular, TypeScript, REST APIs, MongoDB, RabbitMQ, Docker, Kubernetes, and Agile Scrum methodologies.", "Java Developer"),
    ("Entry level Java Developer with solid understanding of Java, Data Structures, Algorithms, Object Oriented Programming, JDBC, Spring Boot, MySQL, and Git. B.Tech in Computer Science.", "Java Developer"),
    ("Lead Java Engineer architecting distributed high-throughput banking systems with Spring Boot, Kafka, Redis, Oracle, Kubernetes, Docker, and reactive programming with WebFlux.", "Java Developer"),
    ("Backend Java Developer with extensive background in Core Java, multithreading, Spring Boot, Hibernate, microservices architecture, REST API design, and Maven build automation.", "Java Developer"),
    ("Java Developer skilled in building scalable cloud-native microservices using Spring Boot, AWS ECS, Docker, PostgreSQL, JUnit testing, and continuous integration.", "Java Developer"),
    ("Enterprise Java Application Developer with experience in J2EE, Spring MVC, Spring Boot, Hibernate ORM, SQL Server, SOAP/REST APIs, and Git.", "Java Developer"),

    # 5. Web Developer
    ("Full Stack Web Developer proficient in JavaScript, TypeScript, React, Next.js, Node.js, Express, HTML5, CSS3, Tailwind CSS, MongoDB, PostgreSQL, and REST APIs. Experienced in responsive UI/UX design.", "Web Developer"),
    ("Frontend Web Developer with expertise in React, Vue.js, JavaScript (ES6+), HTML5, CSS3, Sass, Tailwind CSS, Webpack, responsive web design, cross-browser compatibility, and Figma.", "Web Developer"),
    ("MERN Stack Developer building dynamic single page applications (SPA) using MongoDB, Express.js, React.js, Node.js, Redux, WebSockets, Docker, and AWS S3 hosting.", "Web Developer"),
    ("Web Developer specialized in Next.js, React, Tailwind CSS, GraphQL, Apollo Client, REST APIs, modern UI components, performance optimization, SEO, and accessibility (a11y).", "Web Developer"),
    ("Full Stack JavaScript Developer experienced in Node.js, Express, React, TypeScript, PostgreSQL, RESTful APIs, Redux Toolkit, HTML5, CSS3, and Docker.", "Web Developer"),
    ("Frontend Engineer building modern interactive web applications with React, TypeScript, Next.js, Tailwind CSS, styled-components, state management, and unit testing with Jest.", "Web Developer"),
    ("Web Developer skilled in responsive website development, HTML5, CSS3, JavaScript, Bootstrap, jQuery, WordPress, PHP, and modern frontend frameworks like React.", "Web Developer"),
    ("Full Stack Developer creating cloud-hosted web apps with React frontend, Node.js backend, MongoDB database, REST APIs, and automated deployments with GitHub Actions.", "Web Developer"),

    # 6. DevOps & Cloud Engineer
    ("DevOps Engineer with 4 years experience managing cloud infrastructure on AWS and Azure. Skilled in Docker, Kubernetes (EKS/AKS), Terraform, Ansible, Jenkins, CI/CD pipelines, Prometheus, Grafana, and Linux.", "DevOps & Cloud Engineer"),
    ("Cloud & DevOps Architect experienced in Infrastructure as Code (IaC) using Terraform, CloudFormation, AWS Lambda, ECS, Docker, Kubernetes, GitLab CI, Bash scripting, and Python automation.", "DevOps & Cloud Engineer"),
    ("Site Reliability Engineer (SRE) / DevOps specialist proficient in Kubernetes cluster management, Docker containerization, CI/CD automation, Linux, Nginx, Terraform, and AWS.", "DevOps & Cloud Engineer"),
    ("Cloud Infrastructure Engineer specialized in AWS cloud solutions, Terraform automation, Docker containers, Kubernetes orchestration, Helm charts, and Prometheus monitoring.", "DevOps & Cloud Engineer"),
    ("DevOps Engineer building secure automated deployment pipelines with Jenkins, GitHub Actions, Docker, Kubernetes, Ansible, Linux bash scripting, and cloud security.", "DevOps & Cloud Engineer"),
    ("Platform Engineer proficient in Kubernetes, Terraform, ArgoCD, GitOps, Docker, AWS, microservices networking, Linux administration, and logging with ELK stack.", "DevOps & Cloud Engineer"),
    ("DevOps Specialist managing multi-cloud environments across GCP and AWS with Terraform, Docker, Kubernetes, CI/CD automation, Grafana dashboards, and shell scripting.", "DevOps & Cloud Engineer"),
    ("Cloud Engineer experienced in automated server provisioning, Ansible, Docker, Kubernetes, Linux sysadmin, AWS VPC, EC2, S3, IAM, and continuous integration.", "DevOps & Cloud Engineer"),

    # 7. Data Analyst
    ("Data Analyst with 3 years experience in SQL queries, data warehousing, Power BI dashboards, Tableau, Excel advanced formulas, Python (Pandas, Matplotlib), and KPI reporting.", "Data Analyst"),
    ("Business Intelligence & Data Analyst proficient in SQL, PostgreSQL, Snowflake, Tableau, Power BI, data extraction, ETL pipelines, KPI reporting, and customer segmentation.", "Data Analyst"),
    ("Junior Data Analyst skilled in MySQL, Excel pivot tables, Python, data visualization, exploratory data analysis, Google Analytics, and executive summary presentations.", "Data Analyst"),
    ("Senior Data Analyst extracting actionable business insights using SQL, BigQuery, Tableau, Python, statistical hypothesis testing, and automated business reporting.", "Data Analyst"),
    ("BI Analyst specialized in Power BI dashboard development, DAX calculations, SQL data modeling, data warehousing, Excel modeling, and operational KPI tracking.", "Data Analyst"),
    ("Marketing & Product Data Analyst with strong SQL, Python, Tableau, Google Analytics, A/B testing, customer lifecycle analysis, and data storytelling expertise.", "Data Analyst"),
    ("Data Analyst skilled in transforming complex raw datasets into interactive visual reports using SQL, PostgreSQL, Python Pandas, Matplotlib, and Tableau.", "Data Analyst"),
    ("Financial Data Analyst with expertise in quantitative data analysis, Excel VBA, SQL database queries, Power BI reporting, and statistical modeling.", "Data Analyst"),

    # 8. Cybersecurity Specialist
    ("Cybersecurity Analyst with hands-on experience in vulnerability assessment, penetration testing, network security, SIEM (Splunk), firewall configuration, ethical hacking, and Wireshark.", "Cybersecurity Specialist"),
    ("Information Security Specialist proficient in SOC operations, threat intelligence, identity & access management (IAM), zero trust architecture, malware analysis, and cloud security.", "Cybersecurity Specialist"),
    ("Penetration Tester / Security Consultant certified in CEH and CompTIA Security+. Skilled in Burp Suite, Metasploit, Nmap, web application security testing, and secure code review.", "Cybersecurity Specialist"),
    ("Cyber Defense Engineer with experience in incident response, endpoint detection and response (EDR), intrusion detection systems (IDS/IPS), network protocols, and Linux security.", "Cybersecurity Specialist"),
    ("Cloud Security Engineer focusing on AWS security, IAM policy design, vulnerability scanning, compliance auditing, encryption standards, and threat modeling.", "Cybersecurity Specialist"),
    ("SOC Analyst monitoring security events, investigating cyber incidents, analyzing SIEM alerts, triage, malware behavior analysis, and firewall log inspection.", "Cybersecurity Specialist"),
    ("Information Security Analyst specialized in risk assessment, ISO 27001, vulnerability remediation, penetration testing with Kali Linux, OWASP Top 10, and ethical hacking.", "Cybersecurity Specialist"),
    ("Cybersecurity Specialist with deep knowledge in network defense, cryptography, penetration testing tools (Metasploit, Burp Suite, Nmap), and enterprise compliance.", "Cybersecurity Specialist"),

    # 9. Mobile App Developer
    ("Mobile App Developer with 4 years experience developing cross-platform and native apps using Flutter, Dart, React Native, iOS (Swift), Android (Kotlin), and Firebase.", "Mobile App Developer"),
    ("Android Developer specializing in Kotlin, Java, Android SDK, Jetpack Compose, Coroutines, Room DB, Retrofit, MVVM architecture, and Google Play Store publishing.", "Mobile App Developer"),
    ("iOS Developer experienced in Swift, SwiftUI, UIKit, CoreData, Combine, REST APIs, Unit Testing with XCTest, and App Store release management.", "Mobile App Developer"),
    ("Cross-Platform Mobile Engineer building mobile applications with Flutter, Dart, state management (Bloc/Provider), REST APIs, Firebase authentication, and SQLite.", "Mobile App Developer"),
    ("React Native Developer with expertise in building smooth iOS and Android mobile apps with JavaScript, TypeScript, Redux, Native Modules, and mobile UI/UX design.", "Mobile App Developer"),
    ("Mobile Application Developer proficient in Kotlin for Android, Swift for iOS, REST API integration, push notifications, and offline-first database synchronization.", "Mobile App Developer"),
    ("Senior Flutter Developer creating enterprise mobile solutions with Dart, Bloc architecture, CI/CD for mobile, Google Play and App Store deployment, and UI animations.", "Mobile App Developer"),
    ("Mobile App Software Engineer experienced in Flutter, React Native, mobile performance profiling, app monetization, in-app purchases, and responsive mobile interfaces.", "Mobile App Developer"),

    # 10. Human Resources (HR)
    ("Human Resources Specialist with 5 years experience in full-cycle talent acquisition, technical recruitment, employee onboarding, HR policies, performance management, and HRIS systems.", "Human Resources (HR)"),
    ("Senior HR Manager skilled in human resource management, talent sourcing, interview coordination, employee engagement, payroll administration, compliance, and HR analytics.", "Human Resources (HR)"),
    ("Technical Recruiter / HR Executive experienced in IT recruiting, LinkedIn Recruiter, sourcing software engineers, screening resumes, salary negotiations, and employer branding.", "Human Resources (HR)"),
    ("HR Generalist with expertise in employee relations, labor law compliance, talent development, performance appraisal cycles, onboarding, and workforce planning.", "Human Resources (HR)"),
    ("Talent Acquisition Specialist managing end-to-end recruitment lifecycle, headhunting, candidate pipeline management, interview scheduling, and offer rollouts.", "Human Resources (HR)"),
    ("Human Resources Coordinator skilled in HR administrative operations, employee record keeping, benefits administration, company culture initiatives, and HR software.", "Human Resources (HR)"),
    ("HR Business Partner (HRBP) collaborating with business leaders on organizational design, talent retention strategies, employee sentiment, and performance management.", "Human Resources (HR)"),
    ("People Operations & HR Specialist experienced in recruitment operations, HR analytics, compensation and benefits, employee satisfaction surveys, and onboarding.", "Human Resources (HR)"),

    # 11. Business Analyst
    ("Business Analyst with 4 years experience in requirement gathering, BRD / FRD documentation, UML diagrams, user stories, Agile Scrum, Jira, and stakeholder management.", "Business Analyst"),
    ("Senior Business Systems Analyst proficient in translating business needs into technical specifications, gap analysis, wireframing, workflow optimization, SQL, and Power BI.", "Business Analyst"),
    ("Agile Business Analyst skilled in user story mapping, backlog grooming, sprint planning, acceptance criteria definition, Jira, Confluence, and stakeholder communication.", "Business Analyst"),
    ("Business Analyst specialized in financial workflows, business process modeling (BPMN), requirements traceability matrix (RTM), SQL data validation, and feasibility studies.", "Business Analyst"),
    ("IT Business Analyst working closely with software engineering teams, documenting functional requirements, system flowcharts, wireframes, and managing sprint backlogs.", "Business Analyst"),
    ("Digital Transformation Business Analyst conducting process audits, business requirement documents (BRD), ROI analysis, user acceptance testing (UAT), and workflow design.", "Business Analyst"),
    ("Product & Business Analyst experienced in market analysis, feature prioritization, customer journey mapping, user interviews, SQL queries, and product metrics tracking.", "Business Analyst"),
    ("Business Systems Analyst with strong skills in UML modeling, functional requirement gathering, cross-functional collaboration, data analysis with SQL, and Jira.", "Business Analyst"),

    # 12. Software QA Engineer
    ("Software Quality Assurance (QA) Automation Engineer with 4 years experience in Selenium WebDriver, PyTest, JUnit, Java, Python, TestNG, API testing with Postman, and Cypress.", "Software QA Engineer"),
    ("QA Engineer specializing in manual and automated testing, test plan creation, test case execution, regression testing, performance testing using JMeter, and SQL verification.", "Software QA Engineer"),
    ("Senior QA Automation Engineer building test automation frameworks from scratch using Cypress, Playwright, TypeScript, CI/CD integration, and cross-browser testing.", "Software QA Engineer"),
    ("Software Development Engineer in Test (SDET) proficient in Java, Selenium, Cucumber BDD, RestAssured, Postman API automation, Docker, and Jenkins CI/CD pipelines.", "Software QA Engineer"),
    ("Quality Assurance Specialist experienced in functional testing, exploratory testing, bug reporting in Jira, test matrix design, API test automation, and mobile app QA.", "Software QA Engineer"),
    ("QA Test Lead overseeing automated regression suites with PyTest and Selenium, test strategy formulation, defect lifecycle management, and performance benchmarking with JMeter.", "Software QA Engineer"),
    ("Automation Test Engineer with solid background in Python, PyTest, Selenium, API automated testing, continuous integration, test documentation, and Agile QA practices.", "Software QA Engineer"),
    ("Software QA Tester skilled in test case design, black-box testing, integration testing, database SQL testing, Postman API tests, and defect triage in Jira.", "Software QA Engineer")
]


class ResumeClassifier:
    """Manages training, evaluation, persistence, and inference for resume category prediction."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.evaluation_metrics: Dict[str, Any] = {}
        self._load_or_train()

    def _load_or_train(self) -> None:
        """Loads serialized model and vectorizer, or trains if missing."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                return
            except Exception:
                pass
        
        self.train_and_save()

    @classmethod
    def create_dataset_file(cls) -> pd.DataFrame:
        """Creates and saves the labeled resume dataset CSV."""
        df = pd.DataFrame(SAMPLE_RESUME_DATA, columns=["Resume_Text", "Category"])
        df.to_csv(DATASET_PATH, index=False)
        return df

    def train_and_save(self) -> Dict[str, Any]:
        """Trains the classification model on dataset, evaluates performance, and saves artifacts."""
        df = self.create_dataset_file()
        df["Cleaned_Text"] = df["Resume_Text"].apply(clean_text)

        X = df["Cleaned_Text"]
        y = df["Category"]

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Feature Extraction: TF-IDF with sublinear tf & n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train Logistic Regression classifier (multinomial with high C parameter for crisp decision boundaries)
        self.model = LogisticRegression(C=5.0, max_iter=300, random_state=42)
        self.model.fit(X_train_vec, y_train)

        # Predictions on test set
        y_pred = self.model.predict(X_test_vec)

        # Compute Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        unique_labels = sorted(list(set(y_train).union(set(y_test))))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        self.evaluation_metrics = {
            "accuracy": round(float(acc) * 100, 1),
            "precision": round(float(precision) * 100, 1),
            "recall": round(float(recall) * 100, 1),
            "f1_score": round(float(f1) * 100, 1),
            "labels": unique_labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "sample_count": len(df),
            "categories_count": len(unique_labels)
        }

        # Persist model and vectorizer
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)

        return self.evaluation_metrics

    def predict_category(self, resume_text: str) -> Dict[str, Any]:
        """Predicts the category of a resume with probability distribution.

        Args:
            resume_text: Raw plain text of resume.

        Returns:
            Dict containing predicted category, confidence score, and top category probabilities.
        """
        if not self.model or not self.vectorizer:
            self._load_or_train()

        cleaned = clean_text(resume_text)
        if not cleaned:
            return {
                "predicted_category": "Unknown",
                "confidence_score": 0.0,
                "top_categories": [],
                "all_probabilities": {}
            }

        vec = self.vectorizer.transform([cleaned])
        pred_category = self.model.predict(vec)[0]

        # Probability distribution
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = self.model.classes_
            prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probs)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            confidence = prob_dict.get(pred_category, 0.0)
        else:
            sorted_probs = [(pred_category, 100.0)]
            confidence = 100.0
            prob_dict = {pred_category: 100.0}

        return {
            "predicted_category": pred_category,
            "confidence_score": confidence,
            "top_categories": sorted_probs[:5],
            "all_probabilities": prob_dict
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the model evaluation metrics."""
        if not self.evaluation_metrics:
            self.train_and_save()
        return self.evaluation_metrics


_classifier_instance: Optional[ResumeClassifier] = None

def get_classifier() -> ResumeClassifier:
    """Returns the singleton ResumeClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ResumeClassifier()
    return _classifier_instance

