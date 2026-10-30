from exceptions import GuardrailViolationError


MAX_ANSWER_LENGTH = 3000



def validate_answer(answer: str) -> str:
    """
    Validate and normalize an LLM-generated answer
    before returning it to the user.
    """

    if not isinstance(answer, str):
        raise GuardrailViolationError(
            "Generated answer must be a string."
        )

    answer = answer.strip()

    if not answer:
        raise GuardrailViolationError(
            "The model generated an empty answer."
        )

    if len(answer) > MAX_ANSWER_LENGTH:
        raise GuardrailViolationError(
            f"Generated answer exceeds the "
            f"{MAX_ANSWER_LENGTH}-character limit."
        )

    return answer