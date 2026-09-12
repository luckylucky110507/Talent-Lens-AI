import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import AnalysisResult, Candidate, InterviewAttempt, JobDescription, Resume, ResumeAnalysis
from .schemas import CompareRequest, InterviewEvaluationRequest
from .ml.ann_model import ensure_ann
from .ml.ann_predict import predict_ann
from .ml.predict import predict_suitability
from .ml.random_forest_model import FEATURE_NAMES
from .ml.train_model import ensure_model, train_model
from .services.ats_scorer import score_ats
from .services.career_recommender import recommend_careers
from .services.nlp_processor import preprocess_text
from .services.resume_parser import extract_text_from_bytes, parse_resume
from .services.similarity import calculate_similarity
from .services.skill_extractor import analyze_skill_match, extract_skills

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

Base.metadata.create_all(bind=engine)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TalentLens AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # Training is deterministic and only happens the first time, so the first request is fully usable.
    ensure_model()
    ensure_ann()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TalentLens AI", "version": "1.0.0"}


@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename, content = await _read_upload(file)
    try:
        text = extract_text_from_bytes(filename, content)
        parsed = parse_resume(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    candidate = Candidate(name=parsed.get("name", "Candidate"), email=parsed.get("email"), phone=parsed.get("phone"))
    resume = Resume(candidate=candidate, filename=filename, text=text, parsed_data=parsed)
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return {"resume_id": resume.id, "filename": filename, "size": len(content), "parsed": parsed}


@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    candidate_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    if len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide a job description with at least 20 characters.")
    filename, content = await _read_upload(file)
    try:
        resume_text = extract_text_from_bytes(filename, content)
        parsed = parse_resume(resume_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = _analyze_text(resume_text, parsed, job_description)
    if candidate_name and candidate_name.strip():
        parsed["name"] = candidate_name.strip()
    candidate = Candidate(name=parsed["name"], email=parsed["email"], phone=parsed["phone"])
    resume = Resume(candidate=candidate, filename=filename, text=resume_text, parsed_data=parsed)
    jd = JobDescription(title=_extract_role_title(job_description), text=job_description.strip())
    analysis = ResumeAnalysis(resume=resume, job_description=jd)
    db.add(analysis)
    db.flush()
    result["analysis_id"] = analysis.id
    db.add(AnalysisResult(
        analysis_id=analysis.id,
        ats_score=result["scores"]["ats_score"],
        skill_match=result["scores"]["skill_match"],
        similarity=result["scores"]["similarity"],
        ann_probability=result["scores"]["ann_suitability"],
        overall_score=result["scores"]["overall_score"],
        result_json=result,
    ))
    db.commit()
    return result


@app.post("/api/candidates/analyze")
async def analyze_candidates(
    files: list[UploadFile] = File(...),
    job_description: str = Form(...),
):
    if not files or len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload between 1 and 20 resumes.")
    if len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Please provide a job description with at least 20 characters.")
    candidates = []
    for file in files:
        filename, content = await _read_upload(file)
        try:
            text = extract_text_from_bytes(filename, content)
            parsed = parse_resume(text)
        except ValueError as exc:
            candidates.append({"filename": filename, "error": str(exc)})
            continue
        result = _analyze_text(text, parsed, job_description)
        result["filename"] = filename
        candidates.append(result)
    ranked = sorted([c for c in candidates if "error" not in c], key=lambda item: item["scores"]["overall_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {"candidates": ranked, "errors": [c for c in candidates if "error" in c], "count": len(ranked)}


@app.post("/api/candidates/rank")
async def rank_candidates(files: list[UploadFile] = File(...), job_description: str = Form(...)):
    return await analyze_candidates(files, job_description)


@app.post("/api/candidates/compare")
def compare_candidates(request: CompareRequest, db: Session = Depends(get_db)):
    rows = []
    for analysis_id in request.analysis_ids:
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if analysis and analysis.result:
            rows.append(analysis.result.result_json)
    return {"candidates": rows}


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
    if not analysis or not analysis.result:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return analysis.result.result_json


@app.get("/api/career/recommendations/{analysis_id}")
def get_career_recommendations(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
    if not analysis or not analysis.result:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"recommendations": analysis.result.result_json.get("career_recommendations", [])}


@app.get("/api/interview/questions")
def interview_questions(category: str | None = None, difficulty: str | None = None, skills: str | None = None):
    questions = _load_questions()
    detected = {s.strip().lower() for s in (skills or "").split(",") if s.strip()}
    filtered = [
        q for q in questions
        if (not category or q["category"].lower() == category.lower())
        and (not difficulty or q["difficulty"].lower() == difficulty.lower())
        and (not detected or _question_relevant(q, detected))
    ]
    return {"questions": filtered[:30], "count": len(filtered)}


@app.post("/api/interview/evaluate")
def evaluate_interview(request: InterviewEvaluationRequest, db: Session = Depends(get_db)):
    questions = {q["id"]: q for q in _load_questions()}
    question = questions.get(request.question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Interview question not found.")
    answer = request.answer.strip().lower()
    points = request.expected_points or question.get("points", [])
    covered = [point for point in points if any(word in answer for word in point.lower().split() if len(word) > 3)]
    minimum_bonus = 20 if len(answer.split()) >= 25 else 0
    concept_score = (len(covered) / len(points) * 65) if points else 0
    score = round(min(100, concept_score + minimum_bonus + (15 if len(answer.split()) >= 10 else 0)), 2)
    missing = [point for point in points if point not in covered]
    evaluation = {
        "score": score,
        "covered_concepts": covered,
        "missing_concepts": missing,
        "suggestions": _interview_suggestions(score, missing),
        "label": "Rule-based prototype evaluation",
    }
    db.add(InterviewAttempt(question_id=request.question_id, answer=request.answer, score=score, evaluation_json=evaluation))
    db.commit()
    return evaluation


@app.post("/api/model/train")
def retrain_models():
    return {"random_forest": train_model(), "ann": ensure_ann()}


@app.get("/api/model/metrics")
def model_metrics():
    return {"random_forest": ensure_model(), "ann": ensure_ann()}


@app.get("/api/model/feature-importance")
def feature_importance():
    model = __import__("joblib").load(ROOT / "models" / "random_forest_model.joblib")
    return {"features": [{"name": name, "importance": round(float(value), 4)} for name, value in zip(FEATURE_NAMES, model.feature_importances_)]}


def _analyze_text(resume_text: str, parsed: dict[str, Any], jd_text: str) -> dict:
    skills = analyze_skill_match(resume_text, jd_text)
    similarity = calculate_similarity(resume_text, jd_text)
    resume_nlp = preprocess_text(resume_text)
    jd_nlp = preprocess_text(jd_text)
    required_terms = set(jd_nlp["keyword_frequency"])
    resume_terms = set(resume_nlp["keyword_frequency"])
    keyword_match = len(required_terms & resume_terms) / len(required_terms) * 100 if required_terms else 0
    detected_sections = len(parsed.get("sections_detected", []))
    section_completeness = min(100, detected_sections / 6 * 100)
    project_text = f"{parsed.get('projects', '')} {parsed.get('certifications', '')}".lower()
    project_relevance = min(100, calculate_similarity(project_text, jd_text)["score"] + (15 if parsed.get("projects") != "Not detected" else 0))
    ats = score_ats(skills["match_percent"], keyword_match, similarity["score"], section_completeness, project_relevance)
    features = {
        "skill_match": skills["match_percent"],
        "keyword_match": keyword_match,
        "resume_jd_similarity": similarity["score"],
        "experience_score": 85 if parsed.get("experience") != "Not detected" else 20,
        "education_score": 85 if parsed.get("education") != "Not detected" else 20,
        "project_relevance": project_relevance,
        "certification_score": 85 if parsed.get("certifications") != "Not detected" else 20,
        "ats_score": ats["score"],
    }
    rf = predict_suitability(features)
    ann = predict_ann(features)
    overall = round(
        skills["match_percent"] * .40 + similarity["score"] * .25 + ats["score"] * .15
        + project_relevance * .10 + ann["probability"] * .10, 2
    )
    career = recommend_careers(skills["candidate"]["flat"])
    recommendations = _ats_recommendations(skills, parsed, ats["components"], project_relevance)
    questions = _select_questions(skills["candidate"]["flat"], jd_text)
    return {
        "candidate": parsed,
        "scores": {
            "ats_score": ats["score"], "skill_match": skills["match_percent"],
            "similarity": similarity["score"], "ann_suitability": ann["probability"],
            "rf_suitability": rf["probability"], "overall_score": overall,
        },
        "skills": {
            "candidate": skills["candidate"]["flat"], "required": skills["required"]["flat"],
            "matched": skills["matched"], "missing": skills["missing"], "additional": skills["additional"],
        },
        "ats_breakdown": ats["components"],
        "ml_prediction": rf,
        "ann_prediction": ann,
        "feature_values": features,
        "career_recommendations": career,
        "interview_questions": questions,
        "nlp": {
            "resume": {"processed_text": resume_nlp["processed_text"], "token_count": resume_nlp["token_count"], "keyword_frequency": resume_nlp["keyword_frequency"]},
            "job_description": {"processed_text": jd_nlp["processed_text"], "token_count": jd_nlp["token_count"], "keyword_frequency": jd_nlp["keyword_frequency"]},
            "steps": ["Lowercasing", "Text cleaning", "Punctuation removal", "Tokenization", "Stopword removal", "Lemmatization", "Whitespace normalization"],
        },
        "similarity_detail": similarity,
        "recommendations": recommendations,
    }


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "resume")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF or DOCX resume.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Resume must be smaller than 5 MB.")
    return filename, content


def _extract_role_title(text: str) -> str:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "Untitled role")
    return first[:180]


def _load_questions() -> list[dict]:
    return json.loads((DATA_DIR / "interview_questions.json").read_text())


def _question_relevant(question: dict, detected: set[str]) -> bool:
    haystack = f"{question['category']} {question['question']}".lower()
    return any(skill in haystack or skill in question["category"].lower() for skill in detected)


def _select_questions(skills: list[str], jd_text: str) -> list[dict]:
    detected = set(s.lower() for s in skills)
    questions = _load_questions()
    relevant = [q for q in questions if _question_relevant(q, detected) or q["category"].lower() in jd_text.lower()]
    return (relevant or questions)[:6]


def _ats_recommendations(skills: dict, parsed: dict, components: dict, project_relevance: float) -> list[str]:
    output = []
    if skills["missing"]:
        output.append("Add the missing technical keywords that genuinely match your experience.")
    if skills["match_percent"] >= 70:
        output.append("Your resume has strong skill alignment with this role.")
    if parsed.get("projects") == "Not detected" or project_relevance < 45:
        output.append("Add measurable outcomes and role-relevant projects.")
    if parsed.get("certifications") == "Not detected":
        output.append("Consider adding relevant certifications or coursework.")
    if components["section_completeness"] < 70:
        output.append("Add clear section headings for experience, education, projects, and skills.")
    return output[:5] or ["Your resume has a solid baseline; tailor the summary to this role."]


def _interview_suggestions(score: float, missing: list[str]) -> list[str]:
    suggestions = []
    if missing:
        suggestions.append("Cover these concepts: " + ", ".join(missing[:3]) + ".")
    if score < 60:
        suggestions.append("Use a short definition, a concrete example, and the trade-off or limitation.")
    if not suggestions:
        suggestions.append("Good coverage. Add a project example to make the answer more convincing.")
    return suggestions


@app.exception_handler(Exception)
async def safe_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": "The analysis service encountered an unexpected error. Please try again."})


DIST_DIR = ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        requested = (DIST_DIR / path).resolve()
        if requested.is_file() and DIST_DIR.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(DIST_DIR / "index.html")
else:
    @app.get("/")
    def no_build():
        return {"message": "TalentLens AI API is running. Build the frontend with npm run build."}