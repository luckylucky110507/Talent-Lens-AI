# TalentLens AI

**AI-Powered Resume Intelligence & Career Matching Platform**

TalentLens AI is an academic/prototype recruitment intelligence system for a final-year B.Tech CSE project. It combines document parsing, classic NLP, TF-IDF/cosine similarity, transparent ATS scoring, a Random Forest classifier, an ANN-compatible model, skill-gap analysis, recruiter ranking, career recommendations, and rule-based interview practice in one Replit-ready application.

> TalentLens AI is designed for demonstration and career assistance only. Its scores and predictions must not be used as the sole basis for real-world employment decisions.

## What is implemented

- PDF and DOCX resume upload with drag-and-drop validation
- Resilient extraction of name, email, phone, summary, education, experience, skills, projects, and certifications
- NLP pipeline: lowercasing, cleaning, punctuation removal, tokenization, stopword removal, lemmatization, and whitespace normalization
- Structured skill extraction from resumes and job descriptions
- Real `TfidfVectorizer` and `cosine_similarity` comparison
- ATS score using the transparent formula:
  - 35% skill match
  - 25% keyword match
  - 20% resume/JD similarity
  - 10% section completeness
  - 10% project/certification relevance
- Synthetic 1,200-row training dataset generated at first startup
- Random Forest training, held-out evaluation, feature importance, and suitability prediction
- ANN-shaped suitability prototype using an 8 → 32 → 16 → 8 → 1 MLP architecture. TensorFlow/Keras can replace the backend where a compatible wheel is available; the default Python 3.13 environment uses scikit-learn's neural-network implementation for portability.
- Candidate overall score combining skill match, similarity, ATS, project relevance, and ANN output
- Recruiter batch ranking and comparison
- Skill-overlap career recommendations
- Local interview question bank and rule-based concept coverage evaluation
- SQLite/SQLAlchemy persistence for candidates, resumes, jobs, analyses, results, and interview attempts
- React/Vite dashboard with responsive analytics views and Recharts

## Project structure

```text
.
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── services/
│   │   ├── resume_parser.py
│   │   ├── nlp_processor.py
│   │   ├── skill_extractor.py
│   │   ├── similarity.py
│   │   ├── ats_scorer.py
│   │   └── career_recommender.py
│   └── ml/
│       ├── train_model.py
│       ├── random_forest_model.py
│       ├── predict.py
│       ├── ann_model.py
│       └── ann_predict.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
├── data/
│   └── interview_questions.json
├── models/                 # generated at startup
├── uploads/
├── tests/
├── requirements.txt
├── package.json
└── vite.config.js
```

## Run locally or on Replit

### Install

```bash
pip install -r requirements.txt
npm install
```

### Start the complete app

```bash
npm run build
python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-5000}
```

The app is served at `http://localhost:5000` locally. Replit uses the configured `Start application` workflow and its proxied preview URL. The backend and built React frontend share the same port.

The first startup creates:

- `data/candidate_training_data.csv`
- `models/random_forest_model.joblib`
- `models/random_forest_metrics.json`
- `models/ann_model.joblib`
- `models/ann_metrics.json`
- `talentlens.db`

### Run tests

```bash
python tests/run_tests.py
```

The tests cover resume parsing, NLP processing, skill matching, TF-IDF similarity, ATS scoring, Random Forest and ANN predictions, career recommendations, health, interview questions, and model metrics.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Service health |
| POST | `/api/resume/upload` | Parse and store a resume |
| POST | `/api/analyze` | Analyze one resume against a JD |
| POST | `/api/candidates/analyze` | Analyze and rank multiple resumes |
| POST | `/api/candidates/rank` | Batch ranking alias |
| POST | `/api/candidates/compare` | Compare stored analysis IDs |
| GET | `/api/analysis/{analysis_id}` | Retrieve persisted analysis |
| GET | `/api/career/recommendations/{analysis_id}` | Retrieve role recommendations |
| GET | `/api/interview/questions` | Filter local question bank |
| POST | `/api/interview/evaluate` | Rule-based answer evaluation |
| POST | `/api/model/train` | Retrain the demo models |
| GET | `/api/model/metrics` | Model metrics and ANN metadata |
| GET | `/api/model/feature-importance` | Random Forest feature importance |

## Methodology

### Feature engineering

The suitability models receive: `skill_match`, `keyword_match`, `resume_jd_similarity`, `experience_score`, `education_score`, `project_relevance`, `certification_score`, and `ats_score`.

The overall candidate score intentionally does not use the model output alone:

```text
40% Skill Match
25% Resume-JD Similarity
15% ATS Score
10% Project Relevance
10% ANN Suitability
```

### NLP and similarity

The same preprocessing pipeline is applied to resume and job description text. TF-IDF creates document vectors and cosine similarity estimates text alignment. It is not semantic LLM reasoning.

### Interview scoring

Interview answers are evaluated locally using expected concept keywords and minimum answer length. The interface explicitly labels this as a **rule-based prototype evaluation**, not an LLM-powered semantic assessment.

## Known limitations

- Resume formatting can make section extraction imperfect; undetected fields are shown as `Not detected`.
- The training dataset is synthetic and cannot represent real hiring outcomes.
- The ANN uses a scikit-learn MLP fallback in the default Python 3.13 environment when TensorFlow is unavailable.
- Skill matching is catalog-based and should be extended for a production domain vocabulary.
- No authentication, live job-market integrations, LLMs, RAG, vector databases, or multi-agent functionality are included by design.

## Future enhancements

Potential future work includes an LLM-based interview assistant, RAG-based career assistant, vector database retrieval, multi-agent recruitment workflows, real job-market integration, richer evaluation datasets, and authenticated workspaces.