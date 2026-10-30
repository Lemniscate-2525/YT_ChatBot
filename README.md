# YouTube RAG Assistant

YouTube transcript RAG with timestamped sources, Redis rate limiting, a local knowledge graph for entity cross-checks, a FastAPI service, and a Streamlit frontend.

## Architecture

\`\`\`text
Streamlit (app.py)
        |
        | HTTP
        v
FastAPI (api/main.py)
    |            |
    |            +--> Knowledge graph (knowledge_graph/)
    |
    +--> RAG inference (rag.py) --> Chroma (chroma_db/)
    |
    +--> Ingestion (ingestion/) --> YouTube transcript --> chunk/embed/store

Redis provides global and per-user request limits.
\`\`\`

## Setup

Use Python 3.11 or 3.12.

\`\`\`powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r req.txt
\`\`\`

Create a local \`.env\` file (never commit it):

\`\`\`env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
REDIS_URL=redis://localhost:6379/0
\`\`\`

Start Redis (Docker required):

\`\`\`powershell
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
\`\`\`

If the container already exists, use \`docker start rag-redis\`.

## Run the services

Terminal 1 — FastAPI:

\`\`\`powershell
uvicorn api.main:app --reload
\`\`\`

API docs: http://127.0.0.1:8000/docs

Terminal 2 — Streamlit:

\`\`\`powershell
streamlit run app.py
\`\`\`

The frontend calls FastAPI at \`http://localhost:8000\`. Override with \`RAG_API_URL\` if needed.

## Main endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | \`/health\` | Liveness check (no model initialization) |
| POST | \`/ingest\` | Fetch transcript, chunk, embed, and store a video |
| POST | \`/ask\` | Retrieve evidence and generate an answer |
| POST | \`/entities/register\` | Register a reviewed canonical entity and aliases |
| POST | \`/entities/resolve\` | Resolve a name against graph aliases |
| GET | \`/graph/summary\` | Graph counts |

## Knowledge graph workflow

The graph is a local NetworkX \`MultiDiGraph\`, persisted to \`data/knowledge_graph.json\` (ignored by Git).

1. Open the Streamlit UI.
2. Under **Knowledge graph → Register a reviewed entity**, add a canonical name such as \`Alex Hormozi\`.
3. Add aliases only when independently supported (for example, a spelling confirmed by a human review or source metadata).
4. Include the evidence and optionally the YouTube video ID.
5. Search a transcript spelling in the resolver. Strong aliases produce a candidate match; weaker matches are marked \`needs_review\`; unrelated text returns \`not_found\`.
6. During RAG inference, name phrases in retrieved chunks are checked against the graph. Reviewed matches are supplied to the LLM separately from transcript text.

**Important:** Fuzzy similarity is not proof of identity. The graph never silently rewrites transcript text. It preserves the source wording and asks the answer model to distinguish it from reviewed identity evidence. The graph must be seeded with reviewed entities; an empty graph cannot infer who a speaker is. This baseline uses heuristic name-phrase extraction, not a full named-entity recognition model.

## Local tests

\`\`\`powershell
python -m pytest tests/ -v
\`\`\`

The graph tests are local and deterministic. API tests require the dependencies in \`req.txt\`. Full ingestion and inference additionally require valid provider keys, a running Redis instance, and a populated Chroma collection.

## Current deployment caveats

- API and Streamlit are separate processes; run both.
- The Streamlit user ID and API request user ID are development placeholders. Production must derive identity from authentication.
- Protect write endpoints and add authentication/authorization before exposing this API publicly.
- Redis rate limiting is only effective when all requests pass through the API.
- The local JSON graph is suitable for this project MVP. A multi-worker or multi-instance deployment should use a shared database/graph service with concurrency control.
- Entity matching proposes candidates; it does not independently verify a person's identity or the truth of a statement.
