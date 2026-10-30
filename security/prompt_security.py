import re
from exceptions import PromptInjectionError

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(your\s+)?system\s+prompt",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"bypass\s+(all\s+)?(safety\s+)?(rules|restrictions|guardrails)",
]




def detect_prompt_injection(query: str) -> None:
    """
    Reject queries matching known prompt-injection patterns.

    This is a basic detection layer, not a complete security solution.
    """

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            raise PromptInjectionError(
                "Query contains a potentially malicious instruction."
            )
