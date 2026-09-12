import re
from collections import Counter

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "with",
    "this", "will", "we", "you", "your", "our", "their", "they", "was", "were",
    "can", "should", "into", "using", "use", "about", "who", "which", "not",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def preprocess_text(text: str) -> dict:
    raw = text or ""
    lowered = raw.lower()
    cleaned = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", lowered)
    cleaned = normalize_whitespace(cleaned)
    tokens = re.findall(r"[a-z][a-z0-9+#.\-/]*", cleaned)
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    lemmas = [_simple_lemma(t) for t in filtered]
    processed = " ".join(lemmas)
    return {
        "raw_length": len(raw),
        "cleaned_text": cleaned,
        "tokens": tokens[:500],
        "stopwords_removed": [t for t in tokens if t not in STOPWORDS][:500],
        "lemmas": lemmas[:500],
        "processed_text": processed,
        "token_count": len(tokens),
        "keyword_frequency": dict(Counter(lemmas).most_common(20)),
    }


def _simple_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token
