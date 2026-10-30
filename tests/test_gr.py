from security.guardrails import (
    validate_answer,
    GuardrailViolationError,
)


def test_valid_answer():
    answer = validate_answer("  The speaker discusses wealth.  ")
    assert answer == "The speaker discusses wealth."


def test_empty_answer():
    try:
        validate_answer("   ")
    except GuardrailViolationError:
        return

    raise AssertionError("Expected GuardrailViolationError")


def test_non_string_answer():
    try:
        validate_answer(None)
    except GuardrailViolationError:
        return

    raise AssertionError("Expected GuardrailViolationError")


def test_answer_too_long():
    try:
        validate_answer("A" * 3001)
    except GuardrailViolationError:
        return

    raise AssertionError("Expected GuardrailViolationError")


if __name__ == "__main__":
    test_valid_answer()
    test_empty_answer()
    test_non_string_answer()
    test_answer_too_long()

    print("All guardrail tests passed.")
