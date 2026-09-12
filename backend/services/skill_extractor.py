import re
from collections import Counter

SKILL_CATALOG = {
    "Programming": ["Python", "Java", "C", "C++", "JavaScript"],
    "Data": ["SQL", "Pandas", "NumPy", "Excel", "Power BI", "Tableau", "Matplotlib", "Seaborn"],
    "Machine Learning": [
        "Machine Learning", "Regression", "Classification", "Linear Regression",
        "Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM",
        "Naive Bayes", "XGBoost", "Gradient Boosting", "Clustering", "K-Means",
        "DBSCAN", "PCA", "Scikit-learn",
    ],
    "Deep Learning": ["ANN", "Artificial Neural Network", "CNN", "RNN", "TensorFlow", "PyTorch", "Keras"],
    "NLP": ["NLP", "Natural Language Processing", "TF-IDF", "Text Classification", "Tokenization", "Sentiment Analysis"],
    "GenAI": ["LLM", "RAG", "Generative AI", "Prompt Engineering"],
    "Development": ["HTML", "CSS", "React", "FastAPI", "Flask", "Git", "GitHub", "Docker", "REST API"],
    "Database": ["MySQL", "PostgreSQL", "MongoDB", "SQLite", "Redis"],
    "Cloud": ["AWS", "Azure", "Google Cloud", "GCP", "Kubernetes"],
    "Other": ["DSA", "Data Structures", "Algorithms", "Statistics", "A/B Testing", "Model Deployment"],
}


def _pattern(skill: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])")


def extract_skills(text: str) -> dict:
    text = text or ""
    found = {}
    for category, skills in SKILL_CATALOG.items():
        category_hits = []
        for skill in skills:
            matches = list(_pattern(skill).finditer(text.lower()))
            if matches:
                category_hits.append(skill)
                found[skill] = {"category": category, "count": len(matches), "positions": [m.start() for m in matches]}
        if category_hits:
            found.setdefault("_categories", {})[category] = category_hits
    return {"skills": found, "flat": [k for k in found if k != "_categories"], "categories": found.get("_categories", {})}


def analyze_skill_match(resume_text: str, jd_text: str) -> dict:
    resume = extract_skills(resume_text)
    required = extract_skills(jd_text)
    candidate_set = set(resume["flat"])
    required_set = set(required["flat"])
    matched = sorted(candidate_set & required_set)
    missing = sorted(required_set - candidate_set)
    additional = sorted(candidate_set - required_set)
    counts = Counter()
    for skill in required["flat"]:
        counts[skill] = required["skills"][skill]["count"]
    priorities = {
        skill: "HIGH" if counts[skill] >= 3 else "MEDIUM" if counts[skill] == 2 else "LOW"
        for skill in missing
    }
    return {
        "candidate": resume,
        "required": required,
        "matched": matched,
        "missing": [{"name": skill, "priority": priorities[skill]} for skill in missing],
        "additional": additional,
        "match_percent": round(len(matched) / len(required_set) * 100, 2) if required_set else 0.0,
    }
