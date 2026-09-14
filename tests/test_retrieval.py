import pytest

from anatoolbox.retrieval import (
    BM25Index,
    Hit,
    dense_rank,
    rank_records,
    reciprocal_rank_fusion,
    tokenize,
)

TEXTS = [
    "autonomous agents and web standards",
    "AI accelerators and data centers",
    "foundation model families and capabilities",
    "agents browsing the web without supervision",
]
RECORDS = [{"i": i, "topic": "agents" if i in (0, 3) else "infra"} for i in range(4)]


def test_tokenize_lowercases_and_splits_on_non_word():
    assert tokenize("Agentic-Web, 2026!") == ["agentic", "web", "2026"]


def test_tokenize_handles_empty():
    assert tokenize("") == [] and tokenize(None) == []


class TestBM25:
    def test_ranks_the_best_match_first(self):
        hits = BM25Index(TEXTS).rank("autonomous agents web standards")
        assert hits[0].index == 0

    def test_omits_non_matching_documents(self):
        hits = BM25Index(TEXTS).rank("accelerators")
        assert [h.index for h in hits] == [1]

    def test_unknown_term_scores_nothing(self):
        assert BM25Index(TEXTS).rank("quantum") == []

    def test_empty_query_returns_nothing(self):
        assert BM25Index(TEXTS).rank("   ") == []

    def test_idf_is_zero_for_unseen_terms(self):
        assert BM25Index(TEXTS).idf("nonexistent") == 0.0

    def test_rarer_terms_carry_more_weight(self):
        index = BM25Index(TEXTS)
        assert index.idf("accelerators") > index.idf("agents")

    def test_empty_corpus_does_not_divide_by_zero(self):
        assert BM25Index([]).rank("anything") == []

    def test_postings_give_the_same_ranking_as_scoring_every_document(self):
        """Candidate pruning is an optimisation; it must not change results."""
        index = BM25Index(TEXTS)
        query = "agents web accelerators model"
        brute = sorted(
            ((i, index.score(tokenize(query), i)) for i in range(index.doc_count)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        brute = [(i, sc) for i, sc in brute if sc > 0]
        assert [(h.index, h.score) for h in index.rank(query)] == brute

    def test_repeated_query_terms_still_count(self):
        index = BM25Index(TEXTS)
        once = index.rank("agents")[0].score
        twice = index.rank("agents agents")[0].score
        assert twice > once


class TestDense:
    def test_orders_by_cosine_similarity(self):
        docs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        assert [h.index for h in dense_rank([1.0, 0.0], docs)] == [0, 2]

    def test_drops_zero_similarity(self):
        """Padding results with irrelevant passages would poison a RAG answer."""
        assert all(h.score > 0 for h in dense_rank([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]))

    def test_accepts_a_query_matrix_or_a_bare_vector(self):
        docs = [[1.0, 0.0], [0.0, 1.0]]
        assert dense_rank([[1.0, 0.0]], docs)[0].index == dense_rank([1.0, 0.0], docs)[0].index

    def test_dimension_mismatch_is_explained(self):
        with pytest.raises(ValueError, match="different embedding models"):
            dense_rank([1.0, 0.0, 0.0], [[1.0, 0.0]])

    def test_zero_vectors_do_not_divide_by_zero(self):
        assert dense_rank([0.0, 0.0], [[0.0, 0.0]]) == []


class TestFusion:
    def test_rewards_agreement_between_rankings(self):
        a = [Hit(0, 9.0, "sparse"), Hit(1, 8.0, "sparse")]
        b = [Hit(1, 0.9, "dense"), Hit(0, 0.8, "dense")]
        fused = reciprocal_rank_fusion([a, b])
        assert {h.index for h in fused} == {0, 1}
        assert fused[0].strategy == "hybrid"

    def test_document_in_both_rankings_outranks_one_in_only_one(self):
        a = [Hit(0, 9.0, "sparse"), Hit(2, 1.0, "sparse")]
        b = [Hit(0, 0.9, "dense")]
        assert reciprocal_rank_fusion([a, b])[0].index == 0

    def test_ignores_score_scale(self):
        """Ranks, not scores — BM25 and cosine never share a scale."""
        a = [Hit(0, 1e6, "sparse")]
        b = [Hit(1, 0.001, "dense")]
        fused = reciprocal_rank_fusion([a, b])
        assert fused[0].score == fused[1].score


def _embedder(texts):
    """Deterministic bag-of-vocab embedder — no model download."""
    vocab = ["agents", "web", "accelerators", "model"]
    return [[float(t.lower().count(w)) for w in vocab] for t in texts]


class TestRankRecords:
    def test_sparse_needs_no_embedder(self):
        hits = rank_records(
            query="agents web", records=RECORDS, texts=TEXTS, strategy="sparse", size=2
        )
        assert len(hits) == 2

    def test_dense_requires_an_embedder(self):
        with pytest.raises(ValueError, match="needs an embedder"):
            rank_records(query="agents", records=RECORDS, texts=TEXTS, strategy="dense")

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            rank_records(query="x", records=RECORDS, texts=TEXTS, strategy="magic")

    @pytest.mark.parametrize("strategy", ["sparse", "dense", "hybrid"])
    def test_every_strategy_returns_at_most_size(self, strategy):
        hits = rank_records(
            query="agents web",
            records=RECORDS,
            texts=TEXTS,
            strategy=strategy,
            size=2,
            embedder=_embedder,
        )
        assert len(hits) <= 2

    def test_predicate_filters_while_walking_not_after_slicing(self):
        """The bug this guards: slicing to `size` then filtering under-fills."""
        hits = rank_records(
            query="agents web",
            records=RECORDS,
            texts=TEXTS,
            strategy="sparse",
            size=2,
            predicate=lambda r: r["topic"] == "agents",
        )
        assert len(hits) == 2
        assert all(RECORDS[h.index]["topic"] == "agents" for h in hits)

    def test_precomputed_matrix_is_reused(self):
        calls = []

        def counting(texts):
            calls.append(len(texts))
            return _embedder(texts)

        rank_records(
            query="agents",
            records=RECORDS,
            texts=TEXTS,
            strategy="dense",
            size=2,
            embedder=counting,
            document_matrix=_embedder(TEXTS),
        )
        assert calls == [1], "only the query should be embedded"


def test_boost_reorders_before_walking():
    """A boost must be able to bring a record into the top `size`, not just reorder it."""
    hits = rank_records(
        query="agents web",
        records=RECORDS,
        texts=TEXTS,
        strategy="sparse",
        size=1,
        boost=lambda record: 0.0 if record["i"] == 0 else 1.0,
    )
    assert [h.index for h in hits] == [3]


def test_boost_multiplies_scores():
    base = rank_records(query="agents web", records=RECORDS, texts=TEXTS, strategy="sparse", size=4)
    boosted = rank_records(
        query="agents web",
        records=RECORDS,
        texts=TEXTS,
        strategy="sparse",
        size=4,
        boost=lambda r: 2.0,
    )
    assert [round(h.score, 9) for h in boosted] == [round(2 * h.score, 9) for h in base]
