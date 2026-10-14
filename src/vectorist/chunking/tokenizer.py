def estimate_tokens(text: str) -> int:
    """Whitespace/word approximation used until a model tokenizer is wired."""
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())
