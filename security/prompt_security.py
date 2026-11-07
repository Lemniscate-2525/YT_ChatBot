import re
from exceptions import PromptInjectionError

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(your\s+)?(system\s+)?prompt",
    r"repeat\s+(the\s+|your\s+)?(system\s+)?prompt",
    r"what\s+are\s+your\s+(initial\s+)?instructions",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"bypass\s+(all\s+)?(safety\s+)?(rules|restrictions|guardrails)",
    r"override\s+(all\s+)?(instructions|rules|safety|guardrails)",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"do\s+anything\s+now",
    r"(?:^|\n)\s*(?:system|assistant)\s*:\s*",
    r"\[/?INST\]",
    r"<\|?(?:system|im_start|im_end)\|?>",
]


def detect_prompt_injection(query: str) -> None:
    """
    Reject queries matching known prompt-injection patterns.

    Protects system prompt and instructions from untrusted user tampering.
    """
    if not isinstance(query, str):
        return

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            raise PromptInjectionError(
                "Query contains a potentially malicious instruction."
            )
