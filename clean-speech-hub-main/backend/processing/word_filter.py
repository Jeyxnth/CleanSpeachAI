import re
from typing import Dict, List

FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically", "literally", "right", "so", "actually"
]

FLAGGED_WORDS = set([
    # Add default flagged words here
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "crap", "asshole", "dick", "piss"
])

def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _build_word_pattern(words: List[str]) -> str:
    ordered = sorted((w for w in words if w), key=len, reverse=True)
    escaped = [re.escape(word).replace(r"\ ", r"\s+") for word in ordered]
    return r"\b(" + "|".join(escaped) + r")\b"


def filter_words(text: str, flagged_words: set = None) -> Dict:
    if flagged_words is None:
        flagged_words = FLAGGED_WORDS
    removed_words = []
    filler_count = 0
    flagged_count = 0
    # Remove filler words
    def filler_repl(match):
        nonlocal filler_count
        filler_count += 1
        removed_words.append(match.group(0))
        return ""
    pattern = _build_word_pattern(FILLER_WORDS)
    cleaned = re.sub(pattern, filler_repl, text, flags=re.IGNORECASE)
    # Replace flagged words
    def flagged_repl(match):
        nonlocal flagged_count
        flagged_count += 1
        removed_words.append(match.group(0))
        return "[removed]"
    if flagged_words:
        pattern_flagged = _build_word_pattern(list(flagged_words))
        cleaned = re.sub(pattern_flagged, flagged_repl, cleaned, flags=re.IGNORECASE)

    cleaned = _normalize_spacing(cleaned)
    return {
        "cleaned_transcript": cleaned,
        "removed_words": removed_words,
        "filler_count": filler_count,
        "flagged_count": flagged_count
    }

def get_filler_words() -> List[str]:
    return FILLER_WORDS

def get_flagged_words() -> List[str]:
    return list(FLAGGED_WORDS)

def add_flagged_word(word: str):
    FLAGGED_WORDS.add(word.lower())
