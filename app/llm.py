"""OpenRouter chat client.

Uses any free model available through OpenRouter (OpenAI-compatible API).
The model is configurable via the OPENROUTER_MODEL env var so you can swap a
free model if one is rate-limited or retired.
"""
import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# A free model on OpenRouter. Swap via env if this one is busy; other free
# options: "deepseek/deepseek-chat-v3-0324:free",
# "google/gemini-2.0-flash-exp:free", "qwen/qwen-2.5-72b-instruct:free".
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

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


def chat(question: str, context: str, history: list[dict] | None = None) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set.")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional OpenRouter attribution headers.
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": "DarGlobal Wasalt Chatbot",
    }
    payload = {
        "model": model,
        "messages": build_messages(question, context, history),
        "temperature": 0.2,
        "max_tokens": 600,
    }

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
    except httpx.HTTPError as e:
        raise LLMError(f"Request to OpenRouter failed: {e}") from e

    if resp.status_code != 200:
        raise LLMError(f"OpenRouter returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected OpenRouter response: {data}") from e
