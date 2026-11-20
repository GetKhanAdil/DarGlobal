"""OpenRouter chat client.

Uses any free model available through OpenRouter (OpenAI-compatible API).
The model is configurable via the OPENROUTER_MODEL env var so you can swap a
free model if one is rate-limited or retired.
"""
import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models on OpenRouter rotate often and single ":free" models can hit
# upstream rate limits. We therefore try several in order and use the first
# that responds. "openrouter/free" is OpenRouter's auto free-router, which
# picks an available free model for you — the most resilient first choice.
# Override with OPENROUTER_MODEL (single slug or comma-separated list).
DEFAULT_MODELS = (
    "openrouter/free,"
    "openai/gpt-oss-120b:free,"
    "meta-llama/llama-3.3-70b-instruct:free"
)
# Backwards-compatible alias (first model) for anything importing DEFAULT_MODEL.
DEFAULT_MODEL = DEFAULT_MODELS.split(",")[0]

SYSTEM_PROMPT = (
    "You are DarGlobal Assistant, a helpful real-estate assistant for DarGlobal "
    "(a London-listed international luxury real estate developer) and Wasalt "
    "(a Saudi real estate platform). Answer ONLY using the information in the "
    "CONTEXT provided with each question. If the answer is not in the context, "
    "say you don't have that information and suggest contacting DarGlobal or "
    "visiting the website. Be concise, accurate and friendly. Do not invent "
    "prices, dates or project details that are not in the context."
)


class LLMError(Exception):
    pass


def build_messages(question: str, context: str, history: list[dict] | None = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # history is a list of {"role": "user"|"assistant", "content": str}
        messages.extend(history[-6:])  # keep the last few turns only
    user_content = (
        f"CONTEXT:\n{context if context else '(no relevant context found)'}\n\n"
        f"QUESTION: {question}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def _models_from_env() -> list[str]:
    raw = os.getenv("OPENROUTER_MODEL", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def chat(question: str, context: str, history: list[dict] | None = None) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENROUTER_API_KEY is not set. Add it in your host's environment "
            "variables and redeploy."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional OpenRouter attribution headers.
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": "DarGlobal Wasalt Chatbot",
    }
    messages = build_messages(question, context, history)

    errors = []
    for model in _models_from_env():
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        except httpx.HTTPError as e:
            errors.append(f"{model}: request failed ({e})")
            continue

        if resp.status_code != 200:
            # 429 (rate limit) or 404 (model/data-policy) -> try the next model.
            errors.append(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")
            continue

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            errors.append(f"{model}: unexpected response {str(data)[:200]}")
            continue

    raise LLMError("All models failed. " + " | ".join(errors))
