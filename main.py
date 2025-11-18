import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from database import db, create_document, get_documents

app = FastAPI(title="SoulSync AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# Models (Pydantic)
# ==========================
class Participant(BaseModel):
    name: str
    role: Optional[str] = Field(None, description="e.g., you, partner, teammate")

class SessionStartRequest(BaseModel):
    title: Optional[str] = "Conversation Session"
    participants: List[Participant]
    baseline_connection: int = Field(60, ge=0, le=100)

class AnalyzeRequest(BaseModel):
    session_id: str
    speaker: str
    text: str = Field(..., description="Transcript snippet to analyze")
    features: Optional[Dict[str, Any]] = Field(None, description="Optional acoustic features: energy, pauses, pace")

class Insight(BaseModel):
    emotions: List[str]
    sentiment: float
    connection_delta: float
    connection_score: float
    empathy_prompt: str
    suggestions: List[str]

class AnalyzeResponse(BaseModel):
    session_id: str
    speaker: str
    text: str
    insight: Insight
    created_at: datetime

class InsightsListResponse(BaseModel):
    session_id: str
    title: Optional[str]
    connection_score: float
    events: List[AnalyzeResponse]
    summary: Dict[str, Any]


# ==========================
# Utility: simple text emotion/sentiment analysis (rule-based demo)
# ==========================
positive_words = {
    "love", "great", "good", "amazing", "glad", "happy", "excited", "appreciate", "thanks", "grateful",
}
negative_words = {
    "angry", "upset", "sad", "worried", "anxious", "frustrated", "annoyed", "tired", "hate", "confused",
}
connection_builders = {"listen", "understand", "hear", "feel", "together", "we", "us", "thank"}
conflict_markers = {"but", "however", "always", "never", "you", "blame"}


def analyze_text(text: str, baseline: float, acoustic: Optional[Dict[str, Any]] = None) -> Insight:
    t = text.lower()
    pos = sum(1 for w in positive_words if w in t)
    neg = sum(1 for w in negative_words if w in t)
    builders = sum(1 for w in connection_builders if w in t)
    conflicts = sum(1 for w in conflict_markers if w in t)

    # sentiment score 0..1
    raw = (pos - neg) * 0.15 + builders * 0.1 - conflicts * 0.1
    sentiment = max(0.0, min(1.0, 0.5 + raw))

    # acoustic nudges (demo): energy and pauses influence
    energy = None
    pauses = None
    pace = None
    if acoustic:
        energy = acoustic.get("energy")  # 0..1 expected
        pauses = acoustic.get("pauses")  # per 10s
        pace = acoustic.get("pace")      # words per minute
        if isinstance(energy, (int, float)):
            sentiment += (energy - 0.5) * 0.1
        if isinstance(pauses, (int, float)):
            sentiment -= max(0, (pauses - 3)) * 0.02
        if isinstance(pace, (int, float)) and pace:
            # too fast can reduce connection
            if pace > 170:
                sentiment -= 0.05
            elif pace < 90:
                sentiment -= 0.02
    sentiment = max(0.0, min(1.0, sentiment))

    # Connection score shift
    delta = (sentiment - 0.5) * 12 + builders * 2 - conflicts * 2
    connection_score = max(0.0, min(100.0, baseline + delta))

    # Emotions (heuristic)
    emotions: List[str] = []
    if neg > pos and "frustrated" in t or "annoyed" in t:
        emotions.append("frustration")
    if "worried" in t or "anxious" in t:
        emotions.append("anxiety")
    if "sad" in t or "tired" in t:
        emotions.append("sadness")
    if pos >= neg and ("love" in t or "grateful" in t or "appreciate" in t):
        emotions.append("gratitude")
    if not emotions:
        emotions.append("neutral")

    # Prompts and suggestions
    if neg > pos or conflicts > 0:
        empathy_prompt = "Try reflecting: 'It sounds like you're feeling X because Y. Did I get that right?'"
        suggestions = [
            "Slow down and paraphrase what you heard.",
            "Acknowledge the feeling before solving the problem.",
            "Use 'we' language to reduce defensiveness.",
        ]
    else:
        empathy_prompt = "Invite depth: 'I'd love to hear more about how that felt for you.'"
        suggestions = [
            "Validate the positive emotion and ask a follow-up.",
            "Share a small vulnerability to deepen connection.",
            "Stay curious and avoid jumping to advice.",
        ]

    return Insight(
        emotions=emotions,
        sentiment=round(sentiment, 3),
        connection_delta=round(delta, 2),
        connection_score=round(connection_score, 1),
        empathy_prompt=empathy_prompt,
        suggestions=suggestions,
    )


# ==========================
# Routes
# ==========================
@app.get("/")
def read_root():
    return {"message": "SoulSync AI Backend Running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from SoulSync backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


@app.post("/api/session/start")
def start_session(payload: SessionStartRequest) -> Dict[str, Any]:
    session_doc = {
        "title": payload.title,
        "participants": [p.model_dump() for p in payload.participants],
        "baseline_connection": payload.baseline_connection,
        "connection_score": payload.baseline_connection,
        "status": "active",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    session_id = create_document("conversationsession", session_doc)
    return {"session_id": session_id, "connection_score": payload.baseline_connection}


@app.post("/api/session/analyze", response_model=AnalyzeResponse)
def analyze_utterance(payload: AnalyzeRequest):
    # Fetch last event to get most recent connection score
    events = get_documents("insightevent", {"session_id": payload.session_id}, limit=1)
    baseline = events[0].get("insight", {}).get("connection_score", 60) if events else 60

    insight = analyze_text(payload.text, baseline, payload.features)

    event_doc = {
        "session_id": payload.session_id,
        "speaker": payload.speaker,
        "text": payload.text,
        "insight": insight.model_dump(),
        "created_at": datetime.now(timezone.utc),
    }
    _id = create_document("insightevent", event_doc)

    return AnalyzeResponse(
        session_id=payload.session_id,
        speaker=payload.speaker,
        text=payload.text,
        insight=insight,
        created_at=event_doc["created_at"],
    )


@app.get("/api/session/{session_id}/insights", response_model=InsightsListResponse)
def get_insights(session_id: str, limit: int = 20):
    # latest events
    events = get_documents("insightevent", {"session_id": session_id}, limit=limit)
    # sort by created_at ascending for readability
    events_sorted = sorted(events, key=lambda x: x.get("created_at", datetime.now(timezone.utc)))

    responses: List[AnalyzeResponse] = []
    scores: List[float] = []
    for e in events_sorted:
        ins = e.get("insight", {})
        scores.append(ins.get("connection_score", 0))
        responses.append(
            AnalyzeResponse(
                session_id=session_id,
                speaker=e.get("speaker", ""),
                text=e.get("text", ""),
                insight=Insight(
                    emotions=ins.get("emotions", []),
                    sentiment=ins.get("sentiment", 0),
                    connection_delta=ins.get("connection_delta", 0),
                    connection_score=ins.get("connection_score", 0),
                    empathy_prompt=ins.get("empathy_prompt", ""),
                    suggestions=ins.get("suggestions", []),
                ),
                created_at=e.get("created_at", datetime.now(timezone.utc)),
            )
        )

    connection_score = scores[-1] if scores else 60

    # get session title if exists
    sess = get_documents("conversationsession", {"_id": None, "id": None}, limit=0)  # placeholder not used
    title = "Conversation Session"

    summary = {
        "total_events": len(responses),
        "avg_connection": round(sum(scores) / len(scores), 1) if scores else connection_score,
        "trend": ("up" if len(scores) > 1 and scores[-1] > scores[0] else "flat") if scores else "flat",
        "last_updated": datetime.now(timezone.utc)
    }

    return InsightsListResponse(
        session_id=session_id,
        title=title,
        connection_score=connection_score,
        events=responses,
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
