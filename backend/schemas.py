from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobDescriptionRequest(BaseModel):
    job_description: str = Field(min_length=20, max_length=30000)


class CompareRequest(BaseModel):
    analysis_ids: List[int] = Field(min_length=2, max_length=5)


class InterviewEvaluationRequest(BaseModel):
    question_id: str
    answer: str = Field(min_length=1, max_length=10000)
    expected_points: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    id: int
    candidate: Dict[str, Any]
    scores: Dict[str, float]
    skills: Dict[str, Any]
    ats_breakdown: Dict[str, float]
    ml_prediction: Dict[str, Any]
    ann_prediction: Dict[str, Any]
    career_recommendations: List[Dict[str, Any]]
    interview_questions: List[Dict[str, Any]]
    nlp: Dict[str, Any]
    recommendations: List[str]
