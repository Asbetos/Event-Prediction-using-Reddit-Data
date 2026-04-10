"""Text cleaning and tokenization helpers."""

import re
import string


def clean_reddit_text(text: str) -> str:
    """Clean a Reddit comment/submission body for NLP processing."""
    if not text or text in ("[deleted]", "[removed]"):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove Reddit-specific markup
    text = re.sub(r"/r/\w+", "", text)
    text = re.sub(r"/u/\w+", "", text)
    # Remove markdown
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_~`#>]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_text(text: str, min_length: int = 10) -> bool:
    """Check if text is usable for NLP (not deleted, not too short)."""
    if not text:
        return False
    if text in ("[deleted]", "[removed]", ""):
        return False
    return len(text) >= min_length
