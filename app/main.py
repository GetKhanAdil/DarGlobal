"""FastAPI app: serves the chat UI and a /chat endpoint.

Flow per message: retrieve relevant docs from the knowledge base (BM25) ->
build a grounded prompt -> call an OpenRouter free model -> return the answer
plus the sources used.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import LLMError, chat
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
    key = os.getenv("OPENROUTER_API_KEY")
    info = {
        "api_key_present": bool(key),
        "api_key_prefix": (key[:10] + "…") if key else None,
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    }
    try:
        answer = chat("Say hello in one short sentence.", "DarGlobal is a developer.", [])
        info["llm_ok"] = True
        info["sample"] = answer[:160]
    except Exception as e:
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
