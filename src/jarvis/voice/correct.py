"""Fixes small mishearings by comparing what was heard with words JARVIS knows.

Whisper writes "power be eye" for "Power BI" or "jupiter" for "Jupyter". This compares
every one-, two- and three-word stretch of the sentence with the names you use (apps,
folders, contacts, websites) and swaps a stretch for the name only when they sound and
look nearly the same. Short words and exact matches are never touched.
"""

from __future__ import annotations

import difflib
import re

_WORD = re.compile(r"[A-Za-z0-9']+")
MIN_LENGTH = 4          # words shorter than this are never replaced
MAX_WORDS = 3
CUTOFF = 0.8            # 1.0 = identical


def _squash(text: str) -> str:
    return "".join(_WORD.findall(text.lower()))


class VocabularyCorrector:
    def __init__(self, terms: list[str], cutoff: float = CUTOFF) -> None:
        self._cutoff = cutoff
        self._terms: dict[str, str] = {}
        for term in terms:
            key = _squash(term)
            if len(key) >= MIN_LENGTH:
                self._terms.setdefault(key, term.strip())

    def __bool__(self) -> bool:
        return bool(self._terms)

    def correct(self, text: str) -> str:
        if not self._terms or not text:
            return text
        words = list(_WORD.finditer(text))
        out: list[str] = []
        cursor = 0
        i = 0
        while i < len(words):
            replaced = False
            for size in range(min(MAX_WORDS, len(words) - i), 0, -1):
                first, last = words[i], words[i + size - 1]
                chunk = text[first.start():last.end()]
                key = _squash(chunk)
                if len(key) < MIN_LENGTH:
                    continue
                if key in self._terms:
                    # Same letters, maybe different spacing or case ("data bricks"): use the proper spelling.
                    proper = self._terms[key]
                    if chunk != proper and chunk.lower() != proper.lower():
                        out.append(text[cursor:first.start()])
                        out.append(proper)
                        cursor = last.end()
                    i += size
                    replaced = True
                    break
                match = [
                    m
                    for m in difflib.get_close_matches(key, self._terms, n=3, cutoff=self._cutoff)
                    if abs(len(m) - len(key)) <= max(2, len(m) // 5)  # similar length: no swallowing of extra words
                ]
                if match:
                    out.append(text[cursor:first.start()])
                    out.append(self._terms[match[0]])
                    cursor = last.end()
                    i += size
                    replaced = True
                    break
            if not replaced:
                i += 1
        out.append(text[cursor:])
        return "".join(out)
