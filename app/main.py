"""FastAPI app: serves the chat UI and a /chat endpoint.

Flow per message: retrieve relevant docs from the knowledge base (BM25) ->
build a grounded prompt -> call an OpenRouter free model -> return the answer
plus the sources used.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load a local .env file if present (no-op in production where env vars are
# set by the host, e.g. Render's Environment tab).
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import DEFAULT_MODELS, LLMError, chat
from .rag import kb

app = FastAPI(title="DarGlobal + Wasalt Chatbot")

_STATIC_DIR = Path(__file__).parent.parent / "static"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class Source(BaseModel):
    source: str
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(kb.docs)}


@app.get("/diag")
def diag():
    """Diagnostic: is the key present, and does a live LLM call work?

    Open /diag in the browser. If 'llm_ok' is false, 'error' shows the exact
    reason from OpenRouter (401 = bad/missing key, 429 = rate limited,
    404 = model or data-policy issue -> enable free endpoints in OpenRouter
    privacy settings).
    """
    key = os.getenv("OPENROUTER_API_KEY")
    info = {
        "api_key_present": bool(key),
        "api_key_prefix": (key[:10] + "…") if key else None,
        "models": os.getenv("OPENROUTER_MODEL", DEFAULT_MODELS),
    }
    try:
        answer = chat("Say hello in one short sentence.", "DarGlobal is a developer.", [])
        info["llm_ok"] = True
        info["sample"] = answer[:160]
    except LLMError as e:
        info["llm_ok"] = False
        info["error"] = str(e)[:800]
    return info


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    context, hits = kb.build_context(req.message, k=4)
    sources = [
        Source(source=d["source"], title=d["title"], url=d["url"]) for d in hits
    ]
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        answer = chat(req.message, context, history)
    except LLMError as e:
        # Graceful fallback: if the LLM is unavailable (e.g. no API key or
        # rate limited), still return the retrieved knowledge so the app works.
        if hits:
            answer = (
                "The AI model is unavailable right now, but here is the most "
                "relevant information I found:\n\n"
                + "\n\n".join(f"- {d['title']}: {d['text']}" for d in hits)
            )
        else:
            answer = (
                "Sorry, I couldn't reach the AI model and found no matching "
                f"information. ({e})"
            )
    return ChatResponse(answer=answer, sources=sources)


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


# Serve any other static assets (css/js) if added later.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
