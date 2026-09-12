from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_text: str, jd_text: str) -> dict:
    texts = [resume_text or "", jd_text or ""]
    if not any(t.strip() for t in texts):
        return {"score": 0.0, "terms": [], "resume_vector_size": 0}
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    try:
        matrix = vectorizer.fit_transform(texts)
        score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)
        terms = vectorizer.get_feature_names_out()
        return {
            "score": round(max(0.0, min(100.0, score)), 2),
            "terms": terms.tolist()[:40],
            "resume_vector_size": int(matrix.shape[1]),
        }
    except ValueError:
        return {"score": 0.0, "terms": [], "resume_vector_size": 0}
