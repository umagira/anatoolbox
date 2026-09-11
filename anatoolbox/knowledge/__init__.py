"""Project domain knowledge packs, loaded when scope matches.

Not stage SKILL.md files (those are research-stage orchestration) and not
model-selected Cursor skills. The server picks packs from YAML frontmatter
and injects them into writing prompts.
"""

from anatoolbox.knowledge.loader import (
    KnowledgePack,
    apply_retrieve_hints,
    format_knowledge_section,
    load_knowledge_packs,
    scope_from_brief,
    select_knowledge_packs,
)

__all__ = [
    "KnowledgePack",
    "apply_retrieve_hints",
    "format_knowledge_section",
    "load_knowledge_packs",
    "select_knowledge_packs",
    "scope_from_brief",
]
