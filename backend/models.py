from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from .database import Base


class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String(180), default="Candidate")
    email = Column(String(180), nullable=True)
    phone = Column(String(80), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resumes = relationship("Resume", back_populates="candidate")


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    filename = Column(String(255))
    text = Column(Text)
    parsed_data = Column(JSON, default=dict)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    candidate = relationship("Candidate", back_populates="resumes")
    analyses = relationship("ResumeAnalysis", back_populates="resume")


class JobDescription(Base):
    __tablename__ = "job_descriptions"
    id = Column(Integer, primary_key=True)
    title = Column(String(180), default="Untitled role")
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    analyses = relationship("ResumeAnalysis", back_populates="job_description")


class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), unique=True)
    category = Column(String(80))


class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    resume = relationship("Resume", back_populates="analyses")
    job_description = relationship("JobDescription", back_populates="analyses")
    result = relationship("AnalysisResult", back_populates="analysis", uselist=False, cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("resume_analyses.id"), unique=True)
    ats_score = Column(Float)
    skill_match = Column(Float)
    similarity = Column(Float)
    ann_probability = Column(Float)
    overall_score = Column(Float)
    result_json = Column(JSON, default=dict)
    analysis = relationship("ResumeAnalysis", back_populates="result")


class InterviewAttempt(Base):
    __tablename__ = "interview_attempts"
    id = Column(Integer, primary_key=True)
    question_id = Column(String(120))
    answer = Column(Text)
    score = Column(Float)
    evaluation_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
