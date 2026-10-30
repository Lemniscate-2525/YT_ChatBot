from exceptions import InvalidQueryError

MAX_QUERY_LENGTH = 200

def validate_query(query: str) -> str:
    """
    Validate and normalize a user query before RAG processing.
    """

    if not isinstance(query, str):
        raise InvalidQueryError("Query must be a string.")

    query = query.strip()

    if not query:
        raise InvalidQueryError("Query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise InvalidQueryError(
            f"Query exceeds the {MAX_QUERY_LENGTH}-character limit."
        )

    return query