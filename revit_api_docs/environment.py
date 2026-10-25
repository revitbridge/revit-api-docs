"""Development-environment notes: a small curated corpus shipped with the package.

`environment.yaml` holds a few dozen hand-maintained English facts about Revit
add-in development environments (target framework, SDK, manifest, reference
packages, runtime checks), each with the Revit releases it applies to and the
URL it was checked against. Matching is lexical, with the keyword search's
token rules (stop words dropped, verbs stemmed); nothing is embedded and
nothing needs an API key or the downloaded data set, so the notes are
available even while the server is still installing its data.

    match("target framework for Revit 2027 add-in")  -> notes for 2027, best first
    select(revit=(2026, 5))                          -> every note for 2026.5
    format_section(notes)                            -> text block for tool output
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

CORPUS = Path(__file__).with_name("environment.yaml")
MAX_NOTES = 5

# A Revit release as (year, update): 2026 -> (2026, 0), 2026.5 -> (2026, 5).
Version = tuple[int, int]

_VERSION_RE = re.compile(r"^(20\d{2})(?:\.(\d))?$")
_QUERY_VERSION_RE = re.compile(r"(?<![\d.])(20[1-3]\d)(?:\.(\d))?(?!\d)")
_BOUND_RE = re.compile(r"^(>=|<=|>|<|==|=)?(20\d{2}(?:\.\d)?)$")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._+#-]*")
_STRIP = "?!,;:\"'()[]{}<>`*"

# Query words that mean the same as a tag or a corpus word.
_SYNONYMS = {
    "net": "dotnet",
    "plugin": "addin",
    "plug-in": "addin",
    "tfm": "framework",
    "targetframework": "framework",
    "nupkg": "nuget",
    "globaljson": "global.json",
    "lts": "lifecycle",
    "eol": "end-of-support",
}

# A tag hit counts 2, a word that is only in the text 1; a note needs 2.
_TAG_WEIGHT = 2
_TEXT_WEIGHT = 1
_MIN_SCORE = 2


def parse_version(text: str) -> Version:
    """"2027" -> (2027, 0), "2026.5" -> (2026, 5); ValueError otherwise."""
    m = _VERSION_RE.match(str(text).strip())
    if not m:
        raise ValueError(f"not a Revit release: {text!r} (use e.g. 2027 or 2026.5)")
    return int(m.group(1)), int(m.group(2) or 0)


def format_version(v: Version) -> str:
    return f"{v[0]}" if v[1] == 0 else f"{v[0]}.{v[1]}"


def query_version(query: str) -> Version | None:
    """The first Revit release named in a query ("... Revit 2026.5 ..."), if any."""
    m = _QUERY_VERSION_RE.search(query)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


@dataclass(frozen=True)
class RevitRange:
    """Releases a note applies to: ">=2025 <2028", ">=2025.5 <2026 || >=2026.5 <2027", "*"."""

    spec: str
    alternatives: tuple[tuple[tuple[str, Version], ...], ...]

    @classmethod
    def parse(cls, spec: str) -> "RevitRange":
        spec = str(spec).strip()
        if spec in ("", "*"):
            return cls("*", ((),))
        alternatives = []
        for alt in spec.split("||"):
            bounds = []
            for part in alt.split():
                m = _BOUND_RE.match(part)
                if not m:
                    raise ValueError(f"bad Revit range {spec!r}: {part!r}")
                op = m.group(1) or "=="
                bounds.append(("==" if op == "=" else op, parse_version(m.group(2))))
            if not bounds:
                raise ValueError(f"bad Revit range {spec!r}: empty alternative")
            alternatives.append(tuple(bounds))
        return cls(spec, tuple(alternatives))

    def contains(self, v: Version) -> bool:
        return any(all(_holds(v, op, bound) for op, bound in alt) for alt in self.alternatives)


def _holds(v: Version, op: str, bound: Version) -> bool:
    if op == ">=":
        return v >= bound
    if op == ">":
        return v > bound
    if op == "<=":
        return v <= bound
    if op == "<":
        return v < bound
    return v == bound


def _variants(word: str) -> set[str]:
    """The word, without hyphens, without a plural s, and its synonym."""
    out = {word, word.replace("-", "")}
    for w in list(out):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            out.add(w[:-1])
    for w in list(out):
        if w in _SYNONYMS:
            out.add(_SYNONYMS[w])
    out.discard("")
    return out


def _vocabulary(text: str) -> frozenset[str]:
    """Words of a note's text, plus the parts of compound words (net10.0-windows)."""
    words: set[str] = set()
    for raw in _WORD_RE.findall(text.lower()):
        raw = raw.rstrip(".-_")
        if not raw:
            continue
        words |= _variants(raw)
        for part in re.split(r"[-./_]", raw):
            if len(part) >= 2:
                words |= _variants(part)
    return frozenset(words)


@dataclass(frozen=True)
class EnvNote:
    id: str
    text: str
    revit: RevitRange
    tags: tuple[str, ...]
    source: str
    tag_words: frozenset[str]
    words: frozenset[str]

    def applies_to(self, v: Version | None) -> bool:
        return v is None or self.revit.contains(v)


def _note(raw: dict, seen: set[str]) -> EnvNote:
    missing = [k for k in ("id", "text", "revit", "tags", "source") if not raw.get(k)]
    if missing:
        raise ValueError(f"environment note {raw.get('id', '?')!r} lacks {', '.join(missing)}")
    note_id = str(raw["id"]).strip()
    if note_id in seen:
        raise ValueError(f"duplicate environment note id {note_id!r}")
    seen.add(note_id)
    source = str(raw["source"]).strip()
    if not source.startswith("https://"):
        raise ValueError(f"environment note {note_id!r}: source must be an https URL")
    tags = tuple(str(t).strip().lower() for t in raw["tags"] if str(t).strip())
    text = " ".join(str(raw["text"]).split())
    tag_words: set[str] = set()
    for t in tags:
        tag_words |= _variants(t)
    return EnvNote(
        id=note_id,
        text=text,
        revit=RevitRange.parse(raw["revit"]),
        tags=tags,
        source=source,
        tag_words=frozenset(tag_words),
        words=_vocabulary(text + " " + note_id.replace("-", " ")),
    )


def load_notes(path: str | Path | None = None) -> tuple[EnvNote, ...]:
    """Parse and validate the corpus (the packaged file unless `path` is given)."""
    if path is None:
        return _default_notes()
    return _load(Path(path))


@lru_cache(maxsize=1)
def _default_notes() -> tuple[EnvNote, ...]:
    return _load(CORPUS)


def _load(path: Path) -> tuple[EnvNote, ...]:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    seen: set[str] = set()
    return tuple(_note(raw, seen) for raw in doc.get("notes", []))


def _query_words(query: str) -> list[set[str]]:
    """Query tokens (keyword-search rules) without release numbers, as variant sets."""
    from .retriever import normalize_search_tokens

    out = []
    for token, _ in normalize_search_tokens(query):
        token = token.strip(_STRIP)
        if token.startswith(".") and len(token) > 1:   # ".net" -> "net"
            token = token[1:]
        token = token.rstrip(".")
        if len(token) < 2 or _VERSION_RE.match(token):
            continue
        out.append(_variants(token))
    return out


def match(
    query: str,
    *,
    revit: Version | None = None,
    limit: int | None = MAX_NOTES,
    notes: Iterable[EnvNote] | None = None,
) -> list[EnvNote]:
    """Notes the query hits, best first. `revit` defaults to a release named in the query."""
    version = revit if revit is not None else query_version(query)
    words = _query_words(query)
    if not words:
        return []
    scored: list[tuple[int, int, EnvNote]] = []
    for pos, note in enumerate(notes if notes is not None else load_notes()):
        if not note.applies_to(version):
            continue
        score = 0
        for variants in words:
            if variants & note.tag_words:
                score += _TAG_WEIGHT
            elif variants & note.words:
                score += _TEXT_WEIGHT
        if score >= _MIN_SCORE:
            scored.append((score, pos, note))
    scored.sort(key=lambda x: (-x[0], x[1]))
    found = [n for _, _, n in scored]
    return found if limit is None else found[:limit]


def select(revit: Version | None = None, notes: Iterable[EnvNote] | None = None) -> list[EnvNote]:
    """Every note that applies to `revit` (all notes when None), in corpus order."""
    return [n for n in (notes if notes is not None else load_notes()) if n.applies_to(revit)]


def describe_range(note: EnvNote) -> str:
    return "any Revit release" if note.revit.spec == "*" else f"Revit {note.revit.spec}"


def format_section(notes: list[EnvNote]) -> str:
    """The block search_revit_api appends; empty string when there is nothing to show."""
    if not notes:
        return ""
    lines = ["Environment notes (curated development-environment facts, not from the API reference):"]
    for n in notes:
        lines.append(f"- {n.text} [{describe_range(n)}; source: {n.source}]")
    return "\n".join(lines)


def notes_for_query(query: str) -> str:
    """format_section(match(query)), guarded: a broken corpus never breaks a tool call."""
    try:
        return format_section(match(query))
    except Exception as e:  # pragma: no cover - the shipped corpus is tested
        import logging

        logging.getLogger("revit_api_docs.environment").warning(f"environment notes unavailable: {e}")
        return ""
