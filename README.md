# 🎥 YouTube RAG Assistant — Production System

A hardened, containerized, and verifiable YouTube transcript Retrieval-Augmented Generation (RAG) system with timestamped citations, Redis-backed sliding-window rate limiting, entity knowledge graph cross-referencing, an authenticated FastAPI backend, and an interactive Streamlit frontend.

---

## 1. Architecture

```text
┌─────────────────────────────────┐
│ Streamlit Frontend (app.py)     │  (Port 8501)
│ Session-aware, user-facing UI   │
└────────────────┬────────────────┘
                 │
                 │ HTTP (X-API-Key + X-Session-ID)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Backend (api/main.py)                      (Port 8000)
│                                                             │
│  ├── Security: Constant-time Auth, Prompt Injection Filter, │
│  │             Output Guardrails, CORS Middleware           │
│  ├── Health: Liveness (/health) & Readiness (/ready)        │
│  ├── Rate Limiter: Redis Sliding Window (Atomic Lua Script) │
│  ├── Knowledge Graph: NetworkX + RapidFuzz Resolution       │
│  └── Ingestion: YouTube Transcript API with Jittered Retry  │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────┐  ┌────────────────────────────┐
│ Vector Store (ChromaDB)     │  │ LLM Inference              │
│ Embeddings:                 │  │ Groq (openai/gpt-oss-20b)  │
│ gemini-embedding-001        │  │ Grounded Prompting         │
└─────────────────────────────┘  └────────────────────────────┘
```

---

## 2. Core Features

- **Timestamped Evidence Citations**: Every generated claim links directly to the exact second in the source YouTube video.
- **Service-to-Service API Key Auth**: All operational endpoints are gated by constant-time `hmac.compare_digest` against `RAG_API_KEY`.
- **Session-Aware Rate Limiting**: The Streamlit client transmits an isolated `X-Session-ID`, ensuring per-user quotas (10 requests/min) are enforced alongside global system limits (100 requests/min).
- **Prompt Injection Defense**: High-recall heuristic regex and token pattern defenses reject malicious system-prompt override attempts.
- **Atomic Rate Limiter with Circuit Breaker**: Powered by Redis sliding-window Lua scripts. Gracefully raises structured `503 Service Unavailable` if Redis goes offline rather than leaking raw connection exceptions.
- **Reviewed Knowledge Graph**: NetworkX entity store with RapidFuzz similarity ranking. Flags ambiguous transcript spellings for manual human review without blindly autocorrecting source text.
- **Containerized & Reproducible**: Fully orchestrated via Docker and Docker Compose with health checks and non-root execution.
- **Zero-Cost Deployment Target**: Configured for ₹0 / $0 monthly hosting across free-tier providers.

---

## 3. Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Frontend** | Streamlit 1.66+ | Web interface for ingestion, queries, and graph exploration |
| **Backend API** | FastAPI + Uvicorn | High-throughput asynchronous REST API |
| **Embeddings** | Google Generative AI | `gemini-embedding-001` (3072-dimensional vectors) |
| **Vector Database** | ChromaDB 1.5+ | Local persistent vector index partitioned by video ID |
| **LLM Provider** | Groq Cloud | `openai/gpt-oss-20b` (low-latency grounded generation) |
| **Rate Limiter** | Redis 7 + Lua | Distributed atomic sliding-window rate limiting |
| **Graph Store** | NetworkX + RapidFuzz | Entity-relationship graph with alias cross-referencing |
| **Testing** | Pytest + FakeRedis | 65 automated tests with offline mocking & live CI verification |

---

## 4. Repository Structure

```text
├── api/
│   ├── main.py                   # FastAPI REST service & route handlers
│   └── __init__.py
├── config.py                     # Centralized settings and environment validation
├── exceptions.py                 # Structured domain exception hierarchy
├── logging_config.py             # Formatted application logger
├── rag.py                        # RAG pipeline, prompt assembly, and inference
├── app.py                        # Streamlit interactive UI
├── ingestion/
│   ├── ingestion.py              # Chunking, metadata preservation, and Chroma indexing
│   └── yt_client.py              # YouTube transcript extraction with retry backoff
├── knowledge_graph/
│   ├── graph_store.py            # NetworkX graph persistence and alias fuzzy matching
│   └── entity_verification.py    # Verification heuristics and review pipelines
├── infrastructure/
│   └── rate_limiter.py           # Atomic Redis Lua sliding window with health check
├── security/
│   ├── guardrails.py             # Response length and format validators
│   ├── input_validation.py       # Query length and input sanitizer
│   └── prompt_security.py        # Prompt injection pattern rejection
├── tests/                        # 65 automated test suites
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Multi-service local environment (API + Redis + Frontend)
├── render.yaml                   # Free-tier deployment blueprint
├── requirements.txt              # Standard production dependencies
└── .env.example                  # Environment variable template
```

---

## 5. Local Setup & Quickstart

### Option A: Running with Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lemniscate-2525/YT_ChatBot.git
   cd YT_ChatBot
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Add your `GEMINI_API_KEY` and `GROQ_API_KEY`.

3. **Start all services**:
   ```bash
   docker compose up --build
   ```

4. **Access the application**:
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)
   - FastAPI Health Check: [http://localhost:8000/health](http://localhost:8000/health)
   - FastAPI Readiness Check: [http://localhost:8000/ready](http://localhost:8000/ready)

---

### Option B: Running Locally (Native Python)

Requires Python 3.11 or 3.12.

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Copy `.env.example` to `.env` and populate your API keys.

4. **Start the API**:
   ```powershell
   uvicorn api.main:app --reload --port 8000
   ```

5. **Start the Frontend**:
   ```powershell
   streamlit run app.py
   ```

---

## 6. Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google AI Studio key for `gemini-embedding-001` |
| `GROQ_API_KEY` | **Yes** | — | Groq API key for LLM generation |
| `RAG_API_KEY` | **Yes** | — | Shared secret for backend API authentication |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_TIMEOUT` | No | `2.0` | Socket connect and read timeout in seconds |
| `GLOBAL_LIMIT` | No | `100` | Maximum system requests per sliding window |
| `USER_LIMIT` | No | `10` | Maximum requests per session/user per window |
| `RATE_LIMIT_WINDOW`| No | `60` | Window size in seconds |
| `CHROMA_DIR` | No | `chroma_db` | Persistent vector store path |
| `KNOWLEDGE_GRAPH_PATH` | No | `data/knowledge_graph.json` | JSON path for entity graph |
| `ENABLE_API_DOCS` | No | `false` | Enables `/docs` OpenAPI UI (set `true` for local dev) |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated) |
| `RAG_API_URL` | No | `http://localhost:8000` | Streamlit backend URL target |

---

## 7. API Reference

All endpoints except `GET /health` and `GET /ready` require the `X-API-Key` header. Clients can optionally pass `X-Session-ID` to track rate limits per user session.

### Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Public | Liveness probe (returns 200 OK) |
| `GET` | `/ready` | Public | Readiness probe (validates Redis, storage, auth config) |
| `POST` | `/ingest` | Required | Fetches transcript, chunks, embeds, and indexes video |
| `POST` | `/ask` | Required | Executes query validation, retrieval, and grounded inference |
| `POST` | `/entities/register` | Required | Registers reviewed canonical entity and aliases |
| `POST` | `/entities/resolve` | Required | Performs fuzzy candidate matching against known entities |
| `GET` | `/graph/summary` | Required | Returns graph entity and relation statistics |

---

## 8. Free-First Cloud Deployment (₹0 / $0 Cost)

The recommended production deployment runs entirely within permanent free allowances:

```text
┌─────────────────────────────────┐
│ Streamlit Community Cloud       │  100% Free
│ Hosts app.py                    │  (Free public HTTPS, secrets management)
└────────────────┬────────────────┘
                 │
                 │ Public HTTPS + X-API-Key
                 ▼
┌─────────────────────────────────┐
│ Render Web Service (Free Tier)  │  100% Free
│ Hosts api.main:app              │  (Free 750 hrs/month, automatic HTTPS)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Upstash Serverless Redis        │  100% Free
│ Distributed rate limiting       │  (10,000 commands/day free allowance)
└─────────────────────────────────┘
```

### Step 1: Deploy Redis (Upstash)
1. Sign up at [upstash.com](https://upstash.com) (free tier, no credit card required).
2. Create a serverless Redis database.
3. Copy the `rediss://...` connection string.

### Step 2: Deploy Backend (Render)
1. Push this repository to GitHub.
2. In Render, select **New → Blueprint** and link this repository (`render.yaml`).
3. Set your environment variables in Render:
   - `GEMINI_API_KEY`: Your Google AI Studio API key.
   - `GROQ_API_KEY`: Your Groq API key.
   - `REDIS_URL`: The Upstash connection string.
   - `RAG_API_KEY`: Auto-generated or configured strong secret.
4. Deploy the service. Once active, verify `https://YOUR-APP.onrender.com/health`.

### Step 3: Deploy Frontend (Streamlit Community Cloud)
1. Sign up at [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub repository and select `app.py` as the main entry point.
3. In **App settings → Secrets**, paste:
   ```toml
   RAG_API_URL = "https://YOUR-APP.onrender.com"
   RAG_API_KEY = "the-same-secret-used-in-Render"
   ```
4. Deploy! Your app is now publicly accessible at zero infrastructure cost.

---

## 9. Testing & Quality Assurance

The test suite contains 65 automated tests covering authentication, contracts, ingestion, vector store integration, knowledge graph resolution, rate limiting, and prompt security.

Run all tests:
```powershell
python -m pytest tests/ -v
```

GitHub Actions automatically executes the full test suite and verifies the Docker container build on every push and pull request.
