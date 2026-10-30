import json
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

json_path = Path(
    "data/eval_datasets/MzAIP_WJ-jE_20261028_155257.json"
)

with json_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

examples = payload["examples"]

client = Client()

dataset_name = "youtube-rag-MzAIP_WJ-jE-reviewed-v1"

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description=(
        "Reviewed, evidence-validated evaluation dataset "
        "for YouTube video MzAIP_WJ-jE."
    ),
)

client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"new0": example["question"]}
        for example in examples
    ],
    outputs=[
        {
            "reference_answer": example["reference_answer"],
            "evidence": example["evidence"],
        }
        for example in examples
    ],
)

print(f"Uploaded {len(examples)} examples.")
print(f"Dataset name: {dataset_name}")
