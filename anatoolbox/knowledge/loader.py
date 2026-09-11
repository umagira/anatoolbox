"""Load markdown knowledge packs and select them from research scope.

A pack matches a project when:

- ``always: true``, or
- every declared scope facet matches (empty selected = unconstrained), or
- the query/question hits a ``keywords`` entry.

Omitted facets on the call (``None``) do not satisfy a pack constraint —
chat without a brief relies on keywords instead of assuming the whole journey.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class KnowledgePack:
    name: str
    project: str | None
    always: bool
    touchpoints: tuple[str, ...]
    classes: tuple[str, ...]
    keywords: tuple[str, ...]
    retrieve_terms: tuple[str, ...]
    omit_from_query: tuple[str, ...]
    body: str
    path: Path


def _as_str_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_str_list(item))
        return out
    return []


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, Any] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key is not None:
            meta.setdefault(current_key, [])
            if not isinstance(meta[current_key], list):
                meta[current_key] = [meta[current_key]]
            meta[current_key].append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        current_key = key
        if value == "":
            meta[key] = []
        elif value.lower() in {"true", "yes"}:
            meta[key] = True
        elif value.lower() in {"false", "no"}:
            meta[key] = False
        else:
            meta[key] = value
    return meta, body


def _pack_from_file(path: Path) -> KnowledgePack | None:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    body = body.strip()
    if not body:
        return None
    name = str(meta.get("name") or path.stem).strip()
    project = meta.get("project")
    project_s = str(project).strip() if isinstance(project, str) and project.strip() else None
    return KnowledgePack(
        name=name,
        project=project_s.casefold() if project_s else None,
        always=bool(meta.get("always")),
        touchpoints=tuple(_as_str_list(meta.get("touchpoints"))),
        classes=tuple(_as_str_list(meta.get("classes"))),
        keywords=tuple(_as_str_list(meta.get("keywords"))),
        retrieve_terms=tuple(_as_str_list(meta.get("retrieve_terms"))),
        omit_from_query=tuple(_as_str_list(meta.get("omit_from_query"))),
        body=body,
        path=path,
    )


def load_knowledge_packs(*, root: Path | None = None) -> list[KnowledgePack]:
    directory = root or _DEFAULT_ROOT
    packs: list[KnowledgePack] = []
    if not directory.is_dir():
        return packs
    for path in sorted(directory.rglob("*.md")):
        if path.name.startswith("."):
            continue
        pack = _pack_from_file(path)
        if pack is not None:
            packs.append(pack)
    return packs


def _facet_matches(required: Sequence[str], selected: Sequence[str] | None) -> bool:
    """Empty selected = unconstrained. None = unknown (does not match)."""
    if not required:
        return True
    if selected is None:
        return False
    if len(selected) == 0:
        return True
    wanted = {item.casefold() for item in required}
    have = {str(item).casefold() for item in selected}
    return bool(wanted & have)


def _keyword_hit(keywords: Sequence[str], query: str | None) -> bool:
    if not keywords:
        return False
    if not isinstance(query, str) or not query.strip():
        return False
    blob = query.casefold()
    return any(str(kw).strip().casefold() in blob for kw in keywords if str(kw).strip())


def _strip_phrases(text: str, phrases: Sequence[str]) -> str:
    out = text or ""
    for phrase in sorted(
        (p for p in phrases if str(p).strip()), key=lambda p: len(str(p)), reverse=True
    ):
        pattern = r"(?<![a-z0-9])" + re.escape(str(phrase).strip()) + r"(?![a-z0-9])"
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    return " ".join(out.split())


def apply_retrieve_hints(
    semantic: dict[str, Any],
    *,
    project: str | None,
    query: str | None = None,
    research_question: str | None = None,
    root: Path | None = None,
    packs: Iterable[KnowledgePack] | None = None,
) -> dict[str, Any]:
    """Blend pack retrieve_terms / omit_from_query into a semantic-query payload.

    Only packs whose **keywords** hit the original ask are applied — not every
    Cabin-scoped pack. That way a seats/IFEC cabin brief is unchanged.
    """
    out = dict(semantic)
    blob = " ".join(
        part.strip()
        for part in (query, research_question)
        if isinstance(part, str) and part.strip()
    )
    if not blob:
        return out
    catalog = list(packs) if packs is not None else load_knowledge_packs(root=root)
    retrieve: list[str] = []
    omit: list[str] = []
    seen_retrieve: set[str] = set()
    for pack in catalog:
        if pack.project and (project or "").casefold() != pack.project:
            continue
        if not _keyword_hit(pack.keywords, blob):
            continue
        for term in pack.retrieve_terms:
            key = term.casefold()
            if key not in seen_retrieve:
                seen_retrieve.add(key)
                retrieve.append(term)
        omit.extend(pack.omit_from_query)
    if not retrieve and not omit:
        return out
    raw_query = out.get("query") if isinstance(out.get("query"), str) else ""
    cleaned = _strip_phrases(raw_query, omit) if omit else raw_query.strip()
    if retrieve:
        missing = [term for term in retrieve if term.casefold() not in cleaned.casefold()]
        if missing or not cleaned:
            cleaned = " ".join([*retrieve, cleaned]).strip() if cleaned else " ".join(retrieve)
        exact = list(out.get("exact_terms") or [])
        have = {str(item).casefold() for item in exact if isinstance(item, str)}
        for term in retrieve:
            if term.casefold() not in have:
                exact.append(term)
                have.add(term.casefold())
        out["exact_terms"] = exact
    out["query"] = cleaned or raw_query.strip() or blob
    alts = out.get("alternate_queries")
    if omit and isinstance(alts, list):
        out["alternate_queries"] = [
            _strip_phrases(alt, omit) if isinstance(alt, str) else alt for alt in alts
        ]
    return out


def pack_matches(
    pack: KnowledgePack,
    *,
    project: str | None,
    touchpoints: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    query: str | None = None,
) -> bool:
    if pack.project and (project or "").casefold() != pack.project:
        return False
    if pack.always:
        return True
    facets_ok = _facet_matches(pack.touchpoints, touchpoints) and _facet_matches(
        pack.classes, classes
    )
    if pack.touchpoints or pack.classes:
        if facets_ok:
            return True
        return _keyword_hit(pack.keywords, query)
    if pack.keywords:
        return _keyword_hit(pack.keywords, query)
    # No selectors → available for this project on every call.
    return True


def select_knowledge_packs(
    *,
    project: str | None,
    touchpoints: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    query: str | None = None,
    root: Path | None = None,
    packs: Iterable[KnowledgePack] | None = None,
) -> list[KnowledgePack]:
    catalog = list(packs) if packs is not None else load_knowledge_packs(root=root)
    return [
        pack
        for pack in catalog
        if pack_matches(
            pack,
            project=project,
            touchpoints=touchpoints,
            classes=classes,
            query=query,
        )
    ]


def format_knowledge_section(
    *,
    project: str | None,
    touchpoints: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
    query: str | None = None,
    root: Path | None = None,
    packs: Iterable[KnowledgePack] | None = None,
) -> str:
    """Markdown block for a system prompt, or empty when nothing matches."""
    selected = select_knowledge_packs(
        project=project,
        touchpoints=touchpoints,
        classes=classes,
        query=query,
        root=root,
        packs=packs,
    )
    if not selected:
        return ""
    bodies = [pack.body.strip() for pack in selected if pack.body.strip()]
    if not bodies:
        return ""
    return "--- Domain knowledge ---\n\n" + "\n\n".join(bodies)


def _selected_list(block: Any) -> list[str] | None:
    """None if the selector is absent; a list (maybe empty) if present."""
    if block is None:
        return None
    if isinstance(block, dict):
        selected = block.get("selected")
        if selected is None and "available_values" not in block:
            return None
        if isinstance(selected, list):
            return [str(item) for item in selected if isinstance(item, str)]
        return []
    if isinstance(block, list):
        return [str(item) for item in block if isinstance(item, str)]
    return None


def scope_from_brief(brief: dict[str, Any] | None) -> dict[str, Any]:
    """Touchpoints/classes/query for knowledge selection from an approved brief."""
    if not isinstance(brief, dict):
        return {"touchpoints": None, "classes": None, "query": None}
    question = brief.get("research_question")
    query = question.strip() if isinstance(question, str) else None
    return {
        "touchpoints": _selected_list(brief.get("touchpoints")),
        "classes": _selected_list(brief.get("classes")),
        "query": query,
    }
