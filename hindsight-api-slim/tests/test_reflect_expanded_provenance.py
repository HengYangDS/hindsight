"""Successful source expansion is retrieved evidence, not just requested IDs."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import DirectivePage, MemoryEngine, _count_retrieved_facts
from hindsight_api.engine.reflect import agent, tools
from hindsight_api.engine.reflect.models import ReflectAgentResult, ToolCall
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, ToolCallTrace
from hindsight_api.models import RequestContext

RECALLED_ID = "7f62514b-d718-4a26-a55c-66f0801a7bf7"
EXPANDED_ID = "4e6bcbdd-861a-48fc-9047-aeb887e01a92"
MISSING_ID = "3cdb5972-fb10-48e7-8b24-b43b405ea1b2"


def expansion_result():
    return {
        "results": [
            {
                "memory_id": EXPANDED_ID,
                "memory": {
                    "id": EXPANDED_ID,
                    "text": "An independently expanded source fact.",
                    "type": "experience",
                    "context": "Original source context.",
                },
                "chunk": {
                    "id": "second-chunk",
                    "text": "The complete independently expanded original.",
                    "index": 0,
                    "document_id": "second-original",
                },
            },
            {"memory_id": MISSING_ID, "error": "Memory not found."},
        ],
        "count": 2,
    }


class ExpandedEvidenceTests(unittest.TestCase):
    def test_expansion_returns_typed_raw_facts_without_failed_selectors(self):
        facts = tools.expanded_memory_facts(expansion_result())
        self.assertEqual([fact.id for fact in facts], [EXPANDED_ID])
        self.assertEqual(facts[0].fact_type, "experience")
        self.assertEqual(facts[0].context, "Original source context.")

    def test_visible_raw_fact_remains_citable_when_its_original_is_withheld(self):
        output = expansion_result()
        output["results"][0].pop("chunk")
        output["results"][0]["source_withheld"] = "Source is outside the current tag scope."
        facts = tools.expanded_memory_facts(output)
        self.assertEqual([fact.id for fact in facts], [EXPANDED_ID])
        self.assertIsNone(facts[0].document_id)

    def test_failed_expansion_never_becomes_retrieved_evidence(self):
        self.assertEqual(tools.expanded_memory_facts({"error": "Native scope refused."}), [])
        self.assertEqual(
            tools.expanded_memory_facts({"results": [{"memory_id": MISSING_ID, "error": "Missing."}]}),
            [],
        )

    def test_refresh_retrieval_counts_expanded_facts_once_by_native_type(self):
        trace = [
            ToolCallTrace(
                tool="recall",
                input={},
                output={"memories": [{"id": EXPANDED_ID, "fact_type": "experience"}]},
                duration_ms=0,
                iteration=1,
            ),
            ToolCallTrace(tool="expand", input={}, output=expansion_result(), duration_ms=0, iteration=2),
        ]
        self.assertEqual(_count_retrieved_facts(trace), {"experience": 1})
        self.assertEqual(_count_retrieved_facts(trace[1:]), {"experience": 1})


class ExpandedCitationTests(unittest.IsolatedAsyncioTestCase):
    async def test_done_preserves_successfully_expanded_ids_but_not_failed_selectors(self):
        unused = AsyncMock(return_value={})
        recall = AsyncMock(return_value={"memories": [{"id": RECALLED_ID, "text": "First original."}]})
        expand = AsyncMock(return_value=expansion_result())
        llm = SimpleNamespace(
            provider="synthetic",
            model="source-provenance-only",
            call=AsyncMock(),
            call_with_tools=AsyncMock(
                side_effect=[
                    LLMToolCallResult(
                        tool_calls=[LLMToolCall(id="read-first", name="recall", arguments={"query": "first"})],
                        finish_reason="tool_calls",
                    ),
                    LLMToolCallResult(
                        tool_calls=[
                            LLMToolCall(
                                id="read-second",
                                name="expand",
                                arguments={"memory_ids": [EXPANDED_ID, MISSING_ID], "depth": "chunk"},
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    LLMToolCallResult(
                        tool_calls=[
                            LLMToolCall(
                                id="write-document",
                                name="done",
                                arguments={
                                    "document": {
                                        "sections": [{"heading": "Sources", "level": 2, "blocks": ["Both originals."]}]
                                    },
                                    "memory_ids": [RECALLED_ID, EXPANDED_ID, MISSING_ID],
                                },
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                ]
            ),
        )
        result = await agent.run_reflect_agent(
            llm,
            "synthetic-expanded-provenance",
            "Use both original sources and retain their supporting IDs.",
            {"name": "Synthetic"},
            unused,
            unused,
            unused,
            recall,
            expand,
            max_iterations=3,
            max_tokens=1024,
            has_mental_models=False,
            include_observations=False,
            answer_as_document=True,
        )
        self.assertEqual(result.used_memory_ids, [RECALLED_ID, EXPANDED_ID])
        self.assertEqual(result.text, "## Sources\n\nBoth originals.")
        llm.call.assert_not_awaited()


class ExpandedBasedOnTests(unittest.IsolatedAsyncioTestCase):
    async def test_reflect_based_on_hydrates_used_expanded_facts_once(self):
        engine = MemoryEngine(
            memory_llm_provider="mock",
            memory_llm_model="synthetic",
            embeddings=MagicMock(),
            cross_encoder=MagicMock(),
            query_analyzer=MagicMock(),
            skip_llm_verification=True,
        )
        engine._authenticate_tenant = AsyncMock()
        engine._resolve_fuzzy_tag_groups = AsyncMock(return_value=None)
        engine.ensure_bank_profile = AsyncMock(return_value={"name": "Synthetic", "mission": ""})
        engine.get_bank_freshness = AsyncMock(return_value={})
        engine.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))
        engine._get_backend = AsyncMock(return_value=SimpleNamespace())
        engine._config_resolver = SimpleNamespace(
            resolve_full_config=AsyncMock(return_value=_get_raw_config()),
            get_bank_config=AsyncMock(return_value={}),
        )

        async def hydrate(bank_id, facts):
            return [fact.model_copy(update={"document_id": "second-original"}) for fact in facts]

        engine._evidence_as_stored = AsyncMock(side_effect=hydrate)
        trace = [
            ToolCall(
                tool="recall",
                input={},
                output={"memories": [{"id": RECALLED_ID, "text": "Unused first source."}]},
                duration_ms=0,
                iteration=1,
            ),
            ToolCall(tool="expand", input={}, output=expansion_result(), duration_ms=0, iteration=2),
            ToolCall(tool="expand", input={}, output=expansion_result(), duration_ms=0, iteration=3),
        ]
        native_result = ReflectAgentResult(
            text="The expanded source supports this answer.",
            used_memory_ids=[EXPANDED_ID],
            tool_trace=trace,
        )
        with patch("hindsight_api.engine.memory_engine.run_reflect_agent", AsyncMock(return_value=native_result)):
            result = await engine.reflect_async(
                bank_id="synthetic-expanded-source",
                query="Use the second source.",
                exclude_mental_models=True,
                request_context=RequestContext(),
            )

        self.assertEqual(result.based_on["world"], [])
        self.assertEqual([fact.id for fact in result.based_on["experience"]], [EXPANDED_ID])
        self.assertEqual(result.based_on["experience"][0].document_id, "second-original")
        engine._evidence_as_stored.assert_awaited_once()
        self.assertEqual([fact.id for fact in engine._evidence_as_stored.await_args.args[1]], [EXPANDED_ID])


if __name__ == "__main__":
    unittest.main()
