from config import settings
from exceptions import (
    InvalidQueryError,
    PromptInjectionError,
    GuardrailViolationError,
    RateLimitExceededError,
)
from security.input_validation import validate_query
from security.prompt_security import detect_prompt_injection
from security.guardrails import validate_answer

from infrastructure.rate_limiter import check_rate_limit

import os
import re
from urllib.parse import parse_qs, urlparse

import yt_dlp

from dotenv import load_dotenv
from retrieval.vec_store import get_vector_store

from langchain_google_genai import (
    ChatGoogleGenerativeAI ,
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

import logging

from logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


load_dotenv() # Load environment variables from .env file

if not os.getenv("GEMINI_API_KEY"):   # Configuration Validation.
    raise ValueError("Set GEMINI_API_KEY in your .env file.")

if not os.getenv("GROQ_API_KEY"):   # Configuration Validation.
    raise ValueError("Set GROQ_API_KEY in your .env file.")


vector_store = get_vector_store()

if vector_store._collection.count() == 0:  # Explicitly checks if coll is empty so user doesn't query an emoty collection.
    raise RuntimeError(
        "The Chroma collection is empty. "
        "Ingest at least one YouTube video first."
    ) 

# 2. RETRIEVAL : 

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5},
)

# 3. AUGMENTATION : 

prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant for YouTube video transcripts.

Your task is to answer the user's question accurately using ONLY
the supplied transcript context.

Follow these rules carefully:

1. EVIDENCE
   - Base factual claims on the supplied transcript.
   - Do not use outside knowledge to fill gaps in the transcript.
   - A keyword appearing in the transcript does not automatically
     mean the transcript contains the answer.

2. MISSING INFORMATION
   - If the transcript does not provide enough evidence to answer,
     clearly say that the information is not available in the
     provided transcript.
   - If the transcript discusses a topic without establishing
     the specific fact being asked for, explain this distinction.
   - Do not guess, infer unsupported personal details, or fabricate
     facts to produce a complete-sounding answer.

3. ANSWER QUALITY
   - Answer the specific question directly.
   - Include the essential supporting details from the context.
   - For questions asking for multiple points, address each point.
   - Avoid irrelevant information, unnecessary repetition, and
     claims that do not help answer the question.

4. SOURCE CITATIONS
   - Cite claims using the source URLs supplied with the context.
   - Use the provided timestamp links when available.
   - Only cite sources supplied in the context.
   - Do not fabricate URLs, timestamps, or citations.

5. UNCERTAINTY
   - Distinguish explicitly stated facts from reasonable
     interpretations of the transcript.
   - If the evidence is ambiguous or contradictory, explain
     the limitation instead of presenting an uncertain claim
     as established fact.

Retrieved transcript context:
{context}

User question:
{question}

Answer:
""")

# 4. GENERATION : 

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

def ask_question(query, user_id="local-dev-user"):
    # 1. Validate input
    query = validate_query(query)

    # 2. Detect known prompt-injection patterns
    detect_prompt_injection(query)

    # 3. Enforce global and per-user rate limits
    check_rate_limit(user_id)

    # 4. Retrieve relevant transcript chunks
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return "No relevant transcript chunks were found.", []

    # 5. Build context with source links and timestamps
    context_parts = []

    for doc in retrieved_docs:
        video_url = doc.metadata.get("video_url", "")
        start_seconds = doc.metadata.get("start_seconds")

        source = video_url

        if start_seconds is not None:
            separator = "&" if "?" in video_url else "?"
            source = f"{video_url}{separator}t={int(start_seconds)}s"

        context_parts.append(
            f"Source: {source}\n"
            f"Transcript:\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # 6. Augment the prompt
    messages = prompt.invoke({
        "context": context,
        "question": query,
    })

    # 7. Generate the answer
    response = llm.invoke(messages)

    # 8. Validate the generated output
    answer = validate_answer(response.content)

    logger.info(
        "RAG request completed | user_id=%s | retrieved_chunks=%d",
        user_id,
        len(retrieved_docs),
    )

    return answer, retrieved_docs

if __name__ == "__main__":

    while True:
        query = input(
            "\nAsk about the videos (or type 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        try:
            answer, sources = ask_question(query)

            print("\nAnswer:\n", answer)

            print("\nSources:")

            for doc in sources:
                video_url = doc.metadata.get("video_url", "")
                start_seconds = doc.metadata.get("start_seconds")

                if start_seconds is not None:
                    separator = "&" if "?" in video_url else "?"
                    source = (
                        f"{video_url}{separator}"
                        f"t={int(start_seconds)}s"
                    )
                else:
                    source = video_url

                print(source)

        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")

