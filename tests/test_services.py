from backend.ml.ann_predict import predict_ann
from backend.ml.predict import predict_suitability
from backend.services.ats_scorer import score_ats
from backend.services.career_recommender import recommend_careers
from backend.services.nlp_processor import preprocess_text
from backend.services.resume_parser import parse_resume
from backend.services.similarity import calculate_similarity
from backend.services.skill_extractor import analyze_skill_match


def test_resume_parser_extracts_contact_and_sections():
    parsed = parse_resume(
        "Asha Sharma\nasha@example.com | +91 9999999999\n"
        "Education\nB.Tech Computer Science\nSkills\nPython SQL\nProjects\nML dashboard"
    )
    assert parsed["name"] == "Asha Sharma"
    assert parsed["email"] == "asha@example.com"
    assert "education" in parsed["sections_detected"]


def test_nlp_pipeline_normalizes_text():
    result = preprocess_text("Python, Python systems are running quickly.")
    assert result["token_count"] > 0
    assert "python" in result["processed_text"]
    assert "the" not in result["processed_text"]


def test_skill_matching_returns_real_sets():
    result = analyze_skill_match("Python SQL and Machine Learning", "Python SQL TensorFlow")
    assert "Python" in result["matched"]
    assert any(item["name"] == "TensorFlow" for item in result["missing"])
    assert result["match_percent"] > 0


def test_tfidf_similarity_is_bounded():
    result = calculate_similarity("Python SQL machine learning", "Python SQL role")
    assert 0 <= result["score"] <= 100
    assert result["resume_vector_size"] > 0


def test_ats_formula_is_transparent():
    result = score_ats(100, 80, 60, 100, 50)
    assert result["score"] == 82.0
    assert set(result["components"]) == {"skill_match", "keyword_match", "similarity", "section_completeness", "project_relevance"}


def test_ml_and_ann_predictions_return_labels():
    features = {
        "skill_match": 80, "keyword_match": 70, "resume_jd_similarity": 75,
        "experience_score": 80, "education_score": 80, "project_relevance": 70,
        "certification_score": 60, "ats_score": 75,
    }
    assert 0 <= predict_suitability(features)["probability"] <= 100
    assert 0 <= predict_ann(features)["probability"] <= 100


def test_career_recommendations_are_ranked():
    careers = recommend_careers(["Python", "SQL", "Pandas", "NumPy", "Machine Learning"])
    assert careers
    assert careers[0]["match_percent"] >= careers[-1]["match_percent"]
