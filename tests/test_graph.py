"""Knowledge graphs from Stage 1: loading, lookups, and ingest_knowledge_graph."""

import json

import pytest

from anatoolbox import ToolContext
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.ingest.ingest_knowledge_graph import IngestKnowledgeGraphTool
from anatoolbox.graph import (
    KnowledgeGraph,
    clear_graphs,
    describe_fact,
    fact_source_ids,
    get_graph,
)

FACTS = [
    {
        "subject": "Nvidia",
        "relation": "develops",
        "object": "Blackwell",
        "subject_type": "Company",
        "object_type": "Chip",
        "source_ids": ["a1", "a2"],
        "date": "2025-03-18",
    },
    {
        "subject": "Blackwell",
        "relation": "used_by",
        "object": "OpenAI",
        "object_type": "Company",
        "source_id": "a3",
    },
    {"subject": "OpenAI", "relation": "develops", "object": "GPT-4o", "source_ids": "a4;a5"},
    {"subject": "OpenAI", "relation": "develops", "object": "GPT-4", "source_ids": "['a6']"},
    {"subject": "AMD", "relation": "competes_with", "object": "nvidia", "weight": "7"},
]


@pytest.fixture(autouse=True)
def no_graphs():
    clear_graphs()
    yield
    clear_graphs()


@pytest.fixture
def graph():
    return KnowledgeGraph.from_records(FACTS, name="hardware")


class TestLoading:
    def test_source_ids_in_every_accepted_shape(self, graph):
        assert [f["source_ids"] for f in graph.facts] == [
            ["a1", "a2"],
            ["a3"],
            ["a4", "a5"],
            ["a6"],
            [],
        ]
        assert graph.facts[4]["weight"] == "7"  # extra fields are kept

    def test_mapped_field_names(self):
        g = KnowledgeGraph.from_records(
            [{"head": "A", "type": "rel", "tail": "B"}],
            name="g",
            subject_field="head",
            relation_field="type",
            object_field="tail",
        )
        assert g.facts[0]["subject"] == "A" and g.facts[0]["relation"] == "rel"

    def test_a_fact_without_its_core_fields_is_refused(self):
        with pytest.raises(ValueError, match=r"no \['relation'\]"):
            KnowledgeGraph.from_records([{"subject": "A", "object": "B"}], name="g")

    def test_from_csv(self, tmp_path):
        path = tmp_path / "edges.csv"
        path.write_text("subject,relation,object,source_ids\nA,rel,B,1|2\n", encoding="utf-8")
        g = KnowledgeGraph.from_file(path)
        assert g.name == "edges" and g.facts[0]["source_ids"] == ["1", "2"]

    def test_from_node_link_json(self, tmp_path):
        data = {
            "directed": True,
            "nodes": [{"id": "Nvidia", "type": "Company"}, {"id": "H100", "type": "Chip"}],
            "links": [
                {"source": "Nvidia", "target": "H100", "relation": "develops", "source_ids": ["a1"]}
            ],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        g = KnowledgeGraph.from_file(path)
        assert g.facts == [
            {
                "subject": "Nvidia",
                "relation": "develops",
                "object": "H100",
                "source_ids": ["a1"],
                "subject_type": "Company",
                "object_type": "Chip",
            }
        ]

    def test_from_networkx_like_graph(self):
        class Graph:  # the two methods from_networkx uses
            def nodes(self, data=False):
                return [("Nvidia", {"type": "Company"}), ("AMD", {})]

            def edges(self, data=False):
                return [("Nvidia", "AMD", {"weight": 12, "source_ids": ["a1"]})]

        g = KnowledgeGraph.from_networkx(Graph(), name="co")
        assert g.facts[0]["relation"] == "related_to"  # co-mention edges have no label
        assert g.facts[0]["subject_type"] == "Company" and g.facts[0]["weight"] == 12

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "g.graphml"
        path.write_text("<graphml/>", encoding="utf-8")
        with pytest.raises(ValueError, match="from_networkx"):
            KnowledgeGraph.from_file(path)


class TestLookups:
    def test_entities_are_matched_ignoring_case_and_spacing(self, graph):
        assert graph.has_entity("NVIDIA") and graph.has_entity("  gpt-4o ")
        assert graph.entity_type("nvidia") == "Company"
        assert len(graph.facts_about("Nvidia")) == 2  # develops Blackwell; AMD competes_with nvidia

    def test_neighbors_and_relation_filter(self, graph):
        assert graph.neighbors("OpenAI") == ["Blackwell", "GPT-4o", "GPT-4"]
        assert graph.neighbors("OpenAI", relations=["develops"]) == ["GPT-4o", "GPT-4"]

    def test_subgraph_grows_by_hops(self, graph):
        one = graph.subgraph(["Nvidia"], hops=1)
        two = graph.subgraph(["Nvidia"], hops=2)
        assert {f["object"] for f in one} == {"Blackwell", "nvidia"}
        assert {describe_fact(f) for f in one} < {describe_fact(f) for f in two}
        assert any(f["object"] == "OpenAI" for f in two)
        assert len(graph.subgraph(["Nvidia"], hops=3, max_facts=2)) == 2
        assert graph.subgraph(["Intel"]) == []

    def test_find_entities_prefers_the_longest_whole_name(self, graph):
        found = graph.find_entities("Did OpenAI's GPT-4o run on  blackwell, unlike GPT-4?")
        assert found == ["OpenAI", "GPT-4o", "Blackwell", "GPT-4"]
        assert graph.find_entities("NvidiaX and AMDs") == []

    def test_describe_and_helpers(self, graph):
        described = graph.describe()
        assert described["facts"] == 5 and described["relations"]["develops"] == 3
        assert described["facts_with_sources"] == 4 and described["dated_facts"] == 1
        assert describe_fact(graph.facts[0]) == (
            "Nvidia [Company] --develops--> Blackwell [Chip] (2025-03-18)"
        )
        assert fact_source_ids(graph.facts) == ["a1", "a2", "a3", "a4", "a5", "a6"]


class TestIngestKnowledgeGraph:
    def test_registers_describes_and_keeps_the_schema(self, tmp_path):
        edges = tmp_path / "agentic_web.jsonl"
        edges.write_text("\n".join(json.dumps(f) for f in FACTS), encoding="utf-8")
        schema = tmp_path / "schema.md"
        schema.write_text("# Schema\n- develops: Company -> Product", encoding="utf-8")
        out = IngestKnowledgeGraphTool().run(
            {"path": str(edges), "schema_path": str(schema)}, context=ToolContext()
        )
        assert out["graph"] == "agentic_web" and out["facts"] == 5 and out["has_schema"]
        assert get_graph("agentic_web").schema.startswith("# Schema")
        settings = out["provenance"]["settings"]
        assert (
            settings["source_file"] == "agentic_web.jsonl"
            and settings["schema_file"] == "schema.md"
        )
        assert str(tmp_path) not in json.dumps(out)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ToolInputError, match="No such graph file"):
            IngestKnowledgeGraphTool().run(
                {"path": str(tmp_path / "nope.csv")}, context=ToolContext()
            )

    def test_prepare_hook_filters_facts(self, tmp_path):
        edges = tmp_path / "g.jsonl"
        edges.write_text("\n".join(json.dumps(f) for f in FACTS), encoding="utf-8")

        class IngestDevelopsOnly(IngestKnowledgeGraphTool):
            tool_name = "ingest_knowledge_graph_develops_only"

            def prepare(self, graph, settings):
                graph.facts = [f for f in graph.facts if f["relation"] == "develops"]
                graph.refresh()
                return graph

        out = IngestDevelopsOnly().run({"path": str(edges)}, context=ToolContext())
        assert out["facts"] == 3 and not get_graph("g").has_entity("AMD")
