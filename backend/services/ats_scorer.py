def score_ats(skill_match: float, keyword_match: float, similarity: float, section_completeness: float, project_relevance: float) -> dict:
    components = {
        "skill_match": _clip(skill_match),
        "keyword_match": _clip(keyword_match),
        "similarity": _clip(similarity),
        "section_completeness": _clip(section_completeness),
        "project_relevance": _clip(project_relevance),
    }
    score = (
        components["skill_match"] * 0.35
        + components["keyword_match"] * 0.25
        + components["similarity"] * 0.20
        + components["section_completeness"] * 0.10
        + components["project_relevance"] * 0.10
    )
    return {"score": round(score, 2), "components": {k: round(v, 2) for k, v in components.items()}}


def _clip(value: float) -> float:
    return max(0.0, min(100.0, float(value or 0)))
