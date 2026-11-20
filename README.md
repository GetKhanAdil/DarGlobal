# DarGlobal + Wasalt AI Chatbot

An AI chatbot that answers questions about **DarGlobal** (London-listed luxury
real estate developer) and **Wasalt** (Saudi real estate platform), grounded in
publicly available data scraped from their websites.

Built for the DarGlobal "Junior AI Engineer – Applied AI" technical assignment.

- **Scrape** → public data from `darglobal.co.uk` and `wasalt.sa`
- **RAG** → BM25 retrieval over a curated knowledge base
- **LLM** → any free model via **OpenRouter** (OpenAI-compatible API)
- **API/UI** → FastAPI backend + a simple chat web UI
- **Containerised** → single `Dockerfile`
- **Deployed** → runs on any Docker host; one-click config for Render

---

## Architecture

```
          ┌──────────────┐     scrape (requests + BeautifulSoup / Firecrawl)
          │  darglobal.co.uk │ ───────────────────────────┐
          │  wasalt.sa       │                             ▼
          └──────────────┘                      app/data/knowledge_base.json
                                                           │
 user ──▶ static/index.html ──▶ POST /chat ──▶ BM25 retrieve (rank_bm25)
                                                 │
                                                 ▼
                                   build grounded prompt (context + question)
                                                 │
                                                 ▼
                                   OpenRouter free model  ──▶ answer + sources
```

The chatbot only answers from the retrieved context, so it does not hallucinate
prices or projects. If the model is unavailable, it falls back to returning the
retrieved knowledge directly.

## Project structure

```
darglobal-chatbot/
├── app/
│   ├── main.py                 # FastAPI app: /, /chat, /health
│   ├── rag.py                  # BM25 retrieval over the knowledge base
│   ├── llm.py                  # OpenRouter client + grounded prompt
│   └── data/knowledge_base.json# scraped + curated documents
├── scraper/scrape.py           # regenerates the knowledge base (no API key)
├── static/index.html           # chat UI
├── requirements.txt
├── Dockerfile
├── render.yaml                 # one-click Render blueprint
├── .env.example
└── README.md
```

---

## 1. Run locally (Python)

```bash
cd darglobal-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your OpenRouter key into .env
export $(grep -v '^#' .env | xargs)   # load env vars (Linux/macOS)

uvicorn app.main:app --reload
```

Open http://localhost:8000

Get a **free** OpenRouter key at https://openrouter.ai/keys (no credit card
needed for free models).

## 2. Run locally (Docker)

```bash
docker build -t darglobal-chatbot .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-v1-xxxx darglobal-chatbot
```

Open http://localhost:8000

## 3. Refresh the scraped data (optional)

```bash
python scraper/scrape.py        # rewrites app/data/knowledge_base.json
```

---

## Deploy to get a public URL

### Option A — Render (recommended, free, Docker-native)

1. Push this folder to a **GitHub** repo.
2. Go to https://dashboard.render.com → **New** → **Web Service** → connect the repo.
   (Render auto-detects the `Dockerfile`, or use **Blueprint** to read `render.yaml`.)
3. Settings:
   - Runtime: **Docker**
   - Health check path: **/health**
   - Instance type: **Free**
4. Add an environment variable:
   - `OPENROUTER_API_KEY` = your key (mark as secret)
   - *(optional)* `OPENROUTER_MODEL` = `meta-llama/llama-3.3-70b-instruct:free`
5. Click **Create Web Service**. After the build, Render gives you a public URL like
   `https://darglobal-wasalt-chatbot.onrender.com` — that is the working URL to submit.

> Free Render services sleep after inactivity and take ~30–60s to wake on the
> first request. Open the URL once before sharing it.

### Option B — Railway

1. https://railway.app → **New Project** → **Deploy from GitHub repo**.
2. Railway detects the `Dockerfile`. Add `OPENROUTER_API_KEY` under **Variables**.
3. Under **Settings → Networking**, click **Generate Domain** for a public URL.

### Option C — Fly.io

```bash
fly launch --no-deploy          # accept the detected Dockerfile
fly secrets set OPENROUTER_API_KEY=sk-or-v1-xxxx
fly deploy
```
Your URL: `https://<app-name>.fly.dev`

---

## API

`POST /chat`

```json
{ "message": "What projects does DarGlobal have in Oman?", "history": [] }
```

Response:

```json
{
  "answer": "DarGlobal's flagship Oman development is AIDA near Muscat...",
  "sources": [
    { "source": "DarGlobal", "title": "DarGlobal AIDA masterplan and projects in Oman",
      "url": "https://darglobal.co.uk/landing-page/aida" }
  ]
}
```

`GET /health` → `{ "status": "ok", "documents": 20 }`

---

## Notes & honest limitations

- The knowledge base is a curated snapshot of **publicly available** marketing
  content (project names, locations, company profile, services, FAQs). It does
  not include gated data or live per-unit pricing.
- Retrieval is BM25 (keyword) rather than vector embeddings — deliberately, to
  keep the image small and the build fast for a small corpus. Swapping in an
  embedding model + vector store is straightforward if the corpus grows.
- Free OpenRouter models are rate-limited and occasionally retired; set
  `OPENROUTER_MODEL` to another free model if needed.
