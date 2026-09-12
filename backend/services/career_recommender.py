ROLE_SKILLS = {
    "AI/ML Engineer": ["Python", "Machine Learning", "Scikit-learn", "TensorFlow", "SQL", "Docker", "Model Deployment"],
    "Machine Learning Intern": ["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "Statistics"],
    "Data Scientist": ["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "Statistics", "Scikit-learn"],
    "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Pandas", "Statistics"],
    "Data Science Intern": ["Python", "Pandas", "NumPy", "SQL", "Statistics", "Machine Learning"],
    "Python Developer": ["Python", "FastAPI", "Flask", "REST API", "Git", "SQL"],
    "NLP Engineer": ["Python", "NLP", "Natural Language Processing", "TF-IDF", "Machine Learning", "PyTorch"],
    "ML Engineer": ["Python", "Machine Learning", "Scikit-learn", "Docker", "Kubernetes", "Model Deployment"],
    "Business/Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Statistics", "A/B Testing"],
}


def recommend_careers(candidate_skills: list[str], limit: int = 6) -> list[dict]:
    have = set(candidate_skills)
    recommendations = []
    for role, required in ROLE_SKILLS.items():
        matching = sorted(have.intersection(required))
        missing = sorted(set(required) - have)
        score = round(len(matching) / len(required) * 100, 2)
        recommendations.append({
            "role": role,
            "match_percent": score,
            "matching_skills": matching,
            "missing_skills": missing,
            "recommended_skills": missing[:3],
        })
    return sorted(recommendations, key=lambda item: item["match_percent"], reverse=True)[:limit]
