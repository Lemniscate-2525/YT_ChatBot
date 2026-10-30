import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langchain_groq import ChatGroq

load_dotenv()

# Configuration : 
DATASET_NAME = "youtube-rag-baseline-v1"
OUTPUT_FILE = Path("ground_truth.json")
MODEL_NAME = "openai/gpt-oss-20b"

# Initialize LangSmith and Groq : 
client = Client()

judge_llm = ChatGroq(
    model = MODEL_NAME,
    temperature = 0,
)

def load_chunks():
    """Load all chunks from the existing Chroma index."""

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model = "gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name = "youtube_rag",
        persist_directory = "./chroma_db",
        embedding_function = embeddings,
    )

    records = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = records["documents"]
    metadatas = records["metadatas"]
    ids = records["ids"]

    chunks = []

    for index, (chunk_id, content, metadata) in enumerate(
        zip(ids, documents, metadatas),
        start=1,
    ):
        chunks.append(
            {
                "label": f"C{index}",
                "chunk_id": chunk_id,
                "content": content,
                "metadata": metadata or {},
            }
        )

    if not chunks:
        raise ValueError(
            "No chunks found. Check the Chroma path and collection name."
        )

    print(f"Loaded {len(chunks)} chunks from Chroma.")
    return chunks


def parse_json_response(response_text):
    """Parse JSON even if the model wraps it in Markdown fences."""

    response_text = response_text.strip()

    # Remove Markdown code fences if present.
    if response_text.startswith("```"):
        response_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            response_text,
            count=1,
        )
        response_text = re.sub(
            r"\s*```$",
            "",
            response_text,
            count=1,
        )

    # Extract the JSON object if extra text surrounds it.
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"Model did not return a JSON object: {response_text}"
        )

    return json.loads(response_text[start:end + 1])


def label_question(question, chunks):
    """Ask Groq to identify relevant chunks and map labels to Chroma IDs."""

    chunk_text = "\n\n".join(
        f"CHUNK LABEL: {chunk['label']}\n"
        f"CONTENT:\n{chunk['content']}"
        for chunk in chunks
    )

    prompt = f"""
You are helping evaluate a retrieval system.

QUESTION:
{question}

TRANSCRIPT CHUNKS:
{chunk_text}

TASK:
Identify every chunk containing information useful for answering
the question.

RULES:
- Judge relevance using the actual content, not keyword overlap alone.
- Include chunks that provide a useful part of the answer.
- Do not include chunks that are only loosely related.
- Use only the labels C1, C2, C3, etc. shown above.
- Do not return Chroma UUIDs.
- Do not invent labels.
- If no chunk is relevant, return an empty list.
- Return valid JSON only, without any text before or after it.

Use this exact JSON structure:
{{
    "relevant_chunk_ids": ["C1", "C3"],
    "reasoning": "Briefly explain why the selected chunks are relevant."
}}

The example labels above are illustrative. Select labels based on the
actual question and transcript content.
"""

    response = judge_llm.invoke(prompt)
    response_text = response.content

    # Handle responses that return a list of content blocks : 
    if isinstance(response_text, list):
        response_text = "".join(
            block.get("text", "")
            for block in response_text
            if isinstance(block, dict)
        )

    result = parse_json_response(response_text)

    selected_labels = result.get("relevant_chunk_ids")

    if not isinstance(selected_labels, list):
        raise ValueError(
            "The model response must contain a list named "
            "'relevant_chunk_ids'."
        )

    label_to_id = {
        chunk["label"]: chunk["chunk_id"]
        for chunk in chunks
    }

    # Remove duplicate labels while preserving their order.
    selected_labels = list(dict.fromkeys(selected_labels))

    invalid_labels = set(selected_labels) - set(label_to_id)

    if invalid_labels:
        raise ValueError(
            f"Groq returned unknown chunk labels: {invalid_labels}"
        )

    # Convert short labels back to the actual Chroma UUIDs.
    relevant_ids = [
        label_to_id[label]
        for label in selected_labels
    ]

    return {
        "relevant_chunk_ids": relevant_ids,
        "reasoning": result.get("reasoning", ""),
    }


def load_existing_ground_truth():
    """Load previous results so successful questions aren't repeated."""

    if not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0:
        return {}

    try:
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{OUTPUT_FILE} contains invalid JSON. "
            "Back it up and repair it before continuing."
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            f"{OUTPUT_FILE} must contain a JSON object."
        )

    return data


def save_ground_truth(ground_truth):
    """Save progress after every successfully labelled question."""

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            ground_truth,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    chunks = load_chunks()

    examples = list(
        client.list_examples(
            dataset_name=DATASET_NAME
        )
    )

    if not examples:
        raise ValueError(
            f"No examples found in LangSmith dataset '{DATASET_NAME}'."
        )

    print(f"Found {len(examples)} questions in LangSmith.")

    ground_truth = load_existing_ground_truth()

    for index, example in enumerate(examples, start=1):
        question = example.inputs.get("new0")

        if not question:
            print(
                f"[{index}/{len(examples)}] Skipping example "
                f"{example.id}: input key 'new0' was not found."
            )
            continue

        example_id = str(example.id)

        # Reuse successfully saved labels.
        if example_id in ground_truth:
            print(
                f"[{index}/{len(examples)}] "
                "Labels already saved. Skipping."
            )
            continue

        print(f"\n[{index}/{len(examples)}] Question: {question}")

        try:
            labels = label_question(question, chunks)

            ground_truth[example_id] = {
                "question": question,
                "relevant_chunk_ids": labels["relevant_chunk_ids"],
                "reasoning": labels["reasoning"],
                "reviewed": False,
            }

            save_ground_truth(ground_truth)

            print(
                "Proposed relevant chunks:",
                labels["relevant_chunk_ids"],
            )

        except Exception as error:
            print(f"Error processing question: {error}")
            print("Previously saved labels have been preserved.")
            print("Fix the issue and rerun to resume.")
            break

    print(f"\nGround-truth file: {OUTPUT_FILE.resolve()}")
    print("Review the proposed labels before calculating metrics.")


if __name__ == "__main__":
    main()
