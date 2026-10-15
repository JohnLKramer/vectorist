from pathlib import Path

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


def detect_code_language(
    info_string: str | None, source_uri: str = ""
) -> tuple[str, str]:
    """Returns (language, detection_source)."""
    if info_string and info_string.strip():
        language = info_string.strip().split()[0]
        return language, "fence"

    extension = Path(source_uri).suffix.lower()
    if extension in _EXTENSION_TO_LANGUAGE:
        return _EXTENSION_TO_LANGUAGE[extension], "extension"

    return "unknown", "unknown"
