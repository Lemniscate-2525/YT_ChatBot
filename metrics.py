"""
Retrieval metrics for the YouTube RAG evaluation project.

This file contains pure Python calculations only. It does not call any APIs
or run an experiment.

Expected inputs:
- retrieved_ids: chunk IDs returned by the retriever, in ranked order.
- relevant_ids: ALL chunk IDs judged relevant to the question in the reference
  corpus, not only the chunks returned by the retriever.
"""

from collections.abc import Iterable  # Type hint; ie fn can have both list/tuple as input. 

def precision_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """Fraction of the top-k retrieved chunks that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    retrieved = list(retrieved_ids)[:k]  # Top k in a list. 
    relevant = set(relevant_ids)   # Set of the same. 

    if not retrieved:  # if ret dooesn't ret anything, no point in div by 0. 
        return 0.0

    relevant_retrieved = sum(1 for chunk_id in retrieved if chunk_id in relevant) # Total cnt of rel ret chunks. 
    return relevant_retrieved / len(retrieved)   # precision formula. 


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """Fraction of all known relevant chunks found in the top-k results."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    retrieved = list(retrieved_ids)[:k]
    relevant = set(relevant_ids)

    if not relevant:  # if rel is empty we can't calc recall, here it's not even ab divsn with 0; it's more about that rec without rel chunks is undefined. 
        raise ValueError(
            "Recall is undefined because relevant_ids is empty. "
            "Label at least one relevant chunk, or exclude this example."
        )

    relevant_retrieved = sum(1 for chunk_id in retrieved if chunk_id in relevant)
    return relevant_retrieved / len(relevant)


def reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    """Reciprocal rank of the first relevant result within the top-k; 0 if none."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(list(retrieved_ids)[:k], start=1):  # Give seach chunk_id a rank starting from 1 instead of 0
        if chunk_id in relevant:
            return 1.0 / rank  # as soon as 1st rel chunk is disc, reciprocal of it's rank is ret. 

    return 0.0


def evaluate_retrieval(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate the three retrieval metrics for one question."""
    retrieved_ids = list(retrieved_ids)
    relevant_ids = list(relevant_ids)

    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"reciprocal_rank@{k}": reciprocal_rank(retrieved_ids, relevant_ids, k),
    }


def mean_reciprocal_rank(
    retrieved_by_question: Iterable[Iterable[str]], # Iterable twice as we're in list of list.
    relevant_by_question: Iterable[Iterable[str]],
    k: int = 5,
) -> float:
    """Mean reciprocal rank across questions with relevance labels."""
    retrieved_lists = list(retrieved_by_question)
    relevant_lists = list(relevant_by_question)

    if len(retrieved_lists) != len(relevant_lists): # Ensures ground truth labels for every ret result. 
        raise ValueError("Provide one relevant-ID list for every retrieved-ID list.")

    if not retrieved_lists:
        return 0.0

    ranks = [
        reciprocal_rank(retrieved, relevant, k)
        for retrieved, relevant in zip(retrieved_lists, relevant_lists)
    ]
    return sum(ranks) / len(ranks)
