import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

fastapi = types.ModuleType("fastapi")
fastapi.HTTPException = Exception
dependency_modules = {"fastapi": fastapi}
for module_name, class_name in [
    ("services.embedding_service", "EmbeddingService"),
    ("services.reranker_service", "RerankerService"),
    ("services.tokenizer_service", "TokenizerService"),
]:
    module = types.ModuleType(module_name)
    setattr(module, class_name, type(class_name, (), {}))
    dependency_modules[module_name] = module

previous_modules = {
    name: sys.modules.get(name) for name in dependency_modules
}
sys.modules.update(dependency_modules)
from services import index_service as index_service_module
IndexService = index_service_module.IndexService
for name, previous_module in previous_modules.items():
    if previous_module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous_module


def chunk(chunk_id, num_tokens, relevant_chunks=None, position=0):
    return {
        "id": chunk_id,
        "title": chunk_id,
        "content": f"Content for {chunk_id}",
        "score": 1.0,
        "position": position,
        "num_tokens": num_tokens,
        "relevantChunks": relevant_chunks or [],
    }


def relevant_chunk(chunk_id, num_tokens, position=0):
    return {
        "id": chunk_id,
        "title": chunk_id,
        "content": f"Content for {chunk_id}",
        "position": position,
        "num_tokens": num_tokens,
    }


class ContextWindowPackingTest(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(IndexService)

    def build_sources(self, chunks, token_limit):
        with patch.object(index_service_module, "CONTEXT_WINDOW", token_limit), patch.object(
            index_service_module, "PROMPT_BUFFER", 0
        ):
            return self.service._build_sources_from_chunks(chunks, "test-request")

    def test_oversized_main_chunk_does_not_block_smaller_later_chunk(self):
        sources = self.build_sources(
            [
                chunk("first", 1000, position=1),
                chunk("oversized", 1800, position=2),
                chunk("later", 1200, position=3),
            ],
            token_limit=2500,
        )

        self.assertFalse(sources[0].skip)
        self.assertTrue(sources[1].skip)
        self.assertEqual(sources[1].skip_reason, "context_window")
        self.assertFalse(sources[2].skip)
        self.assertEqual(
            sum(source.num_tokens for source in sources if not source.skip),
            2200,
        )

    def test_oversized_related_chunk_does_not_block_later_chunks(self):
        sources = self.build_sources(
            [
                chunk(
                    "parent",
                    500,
                    relevant_chunks=[
                        relevant_chunk("large-related", 2200, position=2),
                        relevant_chunk("small-related", 700, position=3),
                    ],
                    position=1,
                ),
                chunk("later-main", 1200, position=4),
            ],
            token_limit=2500,
        )

        related = sources[0].relevantChunks
        self.assertTrue(related[0].skip)
        self.assertEqual(related[0].skip_reason, "context_window")
        self.assertFalse(related[1].skip)
        self.assertFalse(sources[1].skip)

    def test_children_of_oversized_parent_are_not_promoted(self):
        sources = self.build_sources(
            [
                chunk(
                    "large-parent",
                    1200,
                    relevant_chunks=[relevant_chunk("small-child", 200)],
                    position=1,
                ),
                chunk("later", 500, position=2),
            ],
            token_limit=1000,
        )

        self.assertTrue(sources[0].skip)
        self.assertEqual(sources[0].relevantChunks, [])
        self.assertFalse(sources[1].skip)

    def test_duplicates_use_no_tokens_and_exact_limit_is_accepted(self):
        sources = self.build_sources(
            [
                chunk("first", 600, position=1),
                chunk("first", 600, position=2),
                chunk("exact-fit", 400, position=3),
            ],
            token_limit=1000,
        )

        self.assertFalse(sources[0].skip)
        self.assertTrue(sources[1].skip)
        self.assertEqual(sources[1].skip_reason, "duplicate")
        self.assertFalse(sources[2].skip)
        self.assertEqual(
            sum(source.num_tokens for source in sources if not source.skip),
            1000,
        )

    def test_rerank_candidates_only_include_packed_chunks(self):
        sources = self.build_sources(
            [
                chunk("first", 1000, position=1),
                chunk("oversized", 1800, position=2),
                chunk("later", 1200, position=3),
            ],
            token_limit=2500,
        )

        candidates = self.service._build_rerank_candidates(sources)

        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            ["first", "later"],
        )


if __name__ == "__main__":
    unittest.main()

