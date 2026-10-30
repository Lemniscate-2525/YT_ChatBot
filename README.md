# YouTube RAG Assistant

YouTube transcript RAG with timestamped citations, Redis-backed rate limiting, a reviewed knowledge graph, an authenticated FastAPI backend, and a Streamlit frontend.

## Architecture

\`\`\`text
Streamlit Cloud (app.py) -- X-API-Key --> Render FastAPI (api/main.py)
                                             |-- RAG inference --> Chroma
                                             |-- ingestion --> YouTube transcripts
                                             |-- entity resolution --> knowledge graph
                                             |-- global/per-client rate limiting --> Redis
\`\`\`

## Local setup

Use Python 3.11 or 3.12.

\`\`\`powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r req.txt
\`\`\`

Copy \`.env.example\` to \`.env\` and fill in your provider keys. Start Redis locally:

\`\`\`powershell
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
\`\`\`

If the container already exists, use \`docker start rag-redis\`.

Terminal 1:

\`\`\`powershell
uvicorn api.main:app --reload
\`\`\`

Terminal 2:

\`\`\`powershell
streamlit run app.py
\`\`\`

For local calls, set \`RAG_API_URL=http://localhost:8000\` and \`RAG_API_KEY\` to the same strong random value in your local environment.

## Deploy the API to Render

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from this repository and select \`render.yaml\`.
3. Render generates \`RAG_API_KEY\`. Keep it private.
4. Set \`GEMINI_API_KEY\` and \`GROQ_API_KEY\` in Render environment variables.
5. Create a Redis database on [Upstash](https://console.upstash.com/) using its free tier. In the database console, copy the TLS connection URL (usually starts with \`rediss://\`) and add it to the Render API service as the \`REDIS_URL\` environment variable. Keep the URL secret. Do not use \`localhost\` for hosted Redis. Upstash is external to Render, so a public TLS connection URL is expected; do not use an internal Render URL.
6. The blueprint mounts a persistent disk at \`/var/data\` for Chroma and the knowledge graph.
7. After deployment, check \`https://YOUR-RENDER-SERVICE.onrender.com/health\`. This liveness endpoint is intentionally public.
8. Do not expose the service URL with API credentials in a public repository or logs.

The disk begins empty. Ingest your videos through the authenticated frontend after deployment; local \`chroma_db\` files are ignored by Git and are not automatically uploaded. Keep the disk attached to the same service and back it up before destructive changes.

## Deploy the frontend to Streamlit Community Cloud

1. Create a Streamlit Community Cloud app from this GitHub repository.
2. Set the main file path to \`app.py\`.
3. In the app's **Settings → Secrets**, add:

\`\`\`toml
RAG_API_URL = "https://YOUR-RENDER-SERVICE.onrender.com"
RAG_API_KEY = "paste-the-same-secret-generated-by-Render"
\`\`\`

4. Deploy/reboot the Streamlit app. The API key is read server-side from Streamlit secrets and sent in the \`X-API-Key\` header; it is not shown in the UI.

The repository includes both \`req.txt\` (Render backend) and \`requirements.txt\` (Streamlit Cloud's conventional dependency filename).

## API endpoints

Only \`GET /health\` is public. All other endpoints require the \`X-API-Key\` header.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | \`/health\` | Public liveness check |
| POST | \`/ingest\` | Fetch transcript, chunk, embed, and store a video |
| POST | \`/ask\` | Retrieve evidence and generate an answer |
| POST | \`/entities/register\` | Register a reviewed canonical entity and aliases |
| POST | \`/entities/resolve\` | Resolve a name against graph aliases |
| GET | \`/graph/summary\` | Graph statistics |

Interactive API docs are disabled by default. Set \`ENABLE_API_DOCS=true\` only for controlled debugging.

## Security model and production caveats

- This is **service-to-service API-key authentication**, not individual end-user accounts or OAuth. It protects the backend from unauthenticated callers and is appropriate for a single trusted Streamlit frontend MVP.
- Keep the API key in Render environment variables and Streamlit Cloud secrets only. Never commit it, paste it into client-side code, or log it.
- Rotate the key if exposed. Use separate keys for staging and production.
- The current rate limiter treats requests from the Streamlit service as one client identity. Real per-user quotas require user authentication and a trusted user ID, not a user-editable text field.
- Redis should be private to the backend service. If Redis is unavailable, requests should fail rather than silently bypass rate limiting.
- The local JSON knowledge graph uses the Render persistent disk. This is suitable for a single service instance; multi-instance deployments need shared storage and concurrency control.
- The app requires valid provider keys. YouTube transcript retrieval may fail for videos without accessible captions or when YouTube blocks the service IP.
- Fuzzy matching suggests candidate entities; it does not prove identity. Register aliases only with independent evidence.
- Add individual accounts, OAuth, role-based permissions, audit trails, and stronger abuse controls before treating this as a multi-user public product.

## Tests

\`\`\`powershell
python -m pytest tests/ -v
\`\`\`

GitHub Actions runs the test suite with Redis provisioned as a service.
