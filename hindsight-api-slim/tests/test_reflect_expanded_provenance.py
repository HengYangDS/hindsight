"""A fact reflect read through ``expand`` is evidence, like a recalled one (#5474).

``expand`` returns raw facts the agent reads in full, but only ``recall`` results used to
count: ``done`` dropped a citation of a fact that only ``expand`` had returned, ``based_on``
never carried it, and with no valid citation left reflect fell back to every recalled fact
as the answer's evidence. Only facts an expansion actually returned count; a requested id
that came back as an error does not.

The agent half (``done`` keeps the citation) is in ``test_reflect_agent.py``; the
client-visible story is ``hindsight-system-tests/tests/test_47_expanded_source_provenance.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import _count_retrieved_facts
from hindsight_api.engine.reflect.models import ReflectAgentResult, ToolCall
from hindsight_api.engine.reflect.tools import expanded_memory_facts
from hindsight_api.engine.response_models import ToolCallTrace
from hindsight_api.models import RequestContext

RECALLED_ID = "7f62514b-d718-4a26-a55c-66f0801a7bf7"
EXPANDED_ID = "4e6bcbdd-861a-48fc-9047-aeb887e01a92"
MISSING_ID = "3cdb5972-fb10-48e7-8b24-b43b405ea1b2"


def _expansion() -> dict:
    """An ``expand`` result in the shape tool_expand builds: one fact read, one id not found."""
    return {
        "results": [
            {
                "memory_id": EXPANDED_ID,
                "memory": {
                    "id": EXPANDED_ID,
                    "text": "The deploy runbook requires a source check.",
                    "type": "experience",
                    "context": "release review",
                },
                "chunk": {
                    "id": "chunk-1",
                    "text": "The full chunk the fact was extracted from.",
                    "index": 0,
                    "document_id": "runbook",
                },
            },
            {"memory_id": MISSING_ID, "error": f"Memory not found: {MISSING_ID}"},
        ],
        "count": 2,
    }


def test_only_the_facts_an_expansion_returned_are_evidence():
    facts = expanded_memory_facts(_expansion())

    assert [fact.id for fact in facts] == [EXPANDED_ID]
    assert facts[0].fact_type == "experience"
    assert facts[0].text == "The deploy runbook requires a source check."
    assert facts[0].context == "release review"


def test_a_fact_whose_source_is_withheld_is_still_evidence():
    output = _expansion()
    del output["results"][0]["chunk"]
    output["results"][0]["source_withheld"] = "The source of this memory is outside the current tag scope."

    facts = expanded_memory_facts(output)

    assert [fact.id for fact in facts] == [EXPANDED_ID]
    assert facts[0].document_id is None


@pytest.mark.parametrize(
    "output",
    [
        {"error": "No valid memory IDs provided", "details": {"x": "Invalid memory_id format: x"}},
        {"results": [{"memory_id": MISSING_ID, "error": f"Memory not found: {MISSING_ID}"}], "count": 1},
    ],
    ids=["whole-call-error", "every-id-missing"],
)
def test_a_failed_expansion_yields_no_evidence(output):
    assert expanded_memory_facts(output) == []


def test_refresh_counts_an_expanded_fact_as_retrieved_once():
    recalled_too = ToolCallTrace(
        tool="recall",
        input={},
        output={"memories": [{"id": EXPANDED_ID, "fact_type": "experience"}]},
        duration_ms=0,
        iteration=1,
    )
    expanded = ToolCallTrace(tool="expand", input={}, output=_expansion(), duration_ms=0, iteration=2)

    assert _count_retrieved_facts([expanded]) == {"experience": 1}
    assert _count_retrieved_facts([recalled_too, expanded]) == {"experience": 1}


@pytest.fixture
def engine():
    """A MemoryEngine with everything reflect_async touches around the agent stubbed."""
    pytest.importorskip("sentence_transformers", reason="MemoryEngine construction needs the local-ml extra")
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.memory_engine import DirectivePage

    eng = MemoryEngine(
        memory_llm_provider="mock",
        memory_llm_model="mock-model",
        skip_llm_verification=True,
    )
    eng._authenticate_tenant = AsyncMock()  # type: ignore[method-assign]
    eng.ensure_bank_profile = AsyncMock(return_value={"name": "Test", "mission": ""})  # type: ignore[method-assign]
    eng._bank_freshness = AsyncMock(  # type: ignore[method-assign]
        return_value={"last_consolidated_at": None, "pending_consolidation": 0, "last_memory_write_at": None}
    )
    eng.list_directives = AsyncMock(return_value=DirectivePage(items=[], total=0))  # type: ignore[method-assign]
    eng._get_backend = AsyncMock(return_value=SimpleNamespace())  # type: ignore[method-assign]
    eng._config_resolver = SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=_get_raw_config()),
        get_bank_config=AsyncMock(return_value={}),
    )
    return eng


async def test_based_on_carries_the_cited_expanded_fact_as_stored(engine, monkeypatch):
    """The cited fact reaches based_on once, with its stored provenance; the uncited recall does not."""
    trace = [
        ToolCall(
            tool="recall",
            input={"query": "q"},
            output={"memories": [{"id": RECALLED_ID, "text": "An unrelated recalled fact.", "fact_type": "world"}]},
            duration_ms=1,
        ),
        ToolCall(tool="expand", input={"memory_ids": [EXPANDED_ID, MISSING_ID]}, output=_expansion(), duration_ms=1),
        # Expanded again later in the same reflect: still one piece of evidence.
        ToolCall(tool="expand", input={"memory_ids": [EXPANDED_ID]}, output=_expansion(), duration_ms=1),
    ]

    async def fake_agent(**kwargs):
        return ReflectAgentResult(text="Check the source first.", tool_trace=trace, used_memory_ids=[EXPANDED_ID])

    async def as_stored(bank_id, facts):
        return [fact.model_copy(update={"document_id": "runbook"}) for fact in facts]

    monkeypatch.setattr("hindsight_api.engine.memory_engine.run_reflect_agent", fake_agent)
    engine._evidence_as_stored = AsyncMock(side_effect=as_stored)  # type: ignore[method-assign]

    result = await engine.reflect_async(
        bank_id="bank-1",
        query="What does the runbook require?",
        request_context=RequestContext(),
        exclude_mental_models=True,
    )

    assert result.based_on["world"] == []
    assert [fact.id for fact in result.based_on["experience"]] == [EXPANDED_ID]
    assert result.based_on["experience"][0].document_id == "runbook"
    engine._evidence_as_stored.assert_awaited_once()
    assert [fact.id for fact in engine._evidence_as_stored.await_args.args[1]] == [EXPANDED_ID]
