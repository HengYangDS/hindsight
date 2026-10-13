"""A fact reflect read through `expand` backs its answer like a recalled one (#5474).

`expand` reads a fact in full, with its chunk or document, and the model may cite what
it read in `done`. Only `recall` results used to be citable: a fact the model reached
through `expand` alone was dropped from the answer's citations, and with no valid
citation left reflect reported every recalled fact as the evidence instead -- so
`based_on` named a fact the answer never used.

Here the model learns the id from the request's context, as when a caller points
reflect at a source it already knows. Composed with it: the stored-provenance read
behind `based_on`, the source-text scope (test_81), and a document deleted before the
reflect, whose fact expand can no longer return.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest, ToolCall

pytestmark = pytest.mark.asyncio

FIRST = "Alice lives in Berlin."
SECOND = "The deployment recovery procedure requires an original-source check."
#: What the forced recall searches for: it matches FIRST, never SECOND.
QUERY = "Alice Berlin"


def _expand_then_cite(memory_ids: list[str], *, answer: str, cite: list[str]) -> Callable[[ChatRequest], ToolCall]:
    """The model's free turns: expand ``memory_ids`` once, then answer citing ``cite``."""

    def decide(request: ChatRequest) -> ToolCall:
        expanded = any(
            call.get("function", {}).get("name") == "expand"
            for message in request.messages
            for call in message.get("tool_calls", [])
        )
        if expanded:
            return ToolCall("done", {"answer": answer, "memory_ids": cite})
        return ToolCall("expand", {"memory_ids": memory_ids, "depth": "document"})

    return decide


async def test_a_fact_read_only_through_expand_is_the_answers_evidence(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains=FIRST).returns(extracted(fact(FIRST)))
    llm.on_step("extract_facts", contains=SECOND).returns(extracted(fact(SECOND, fact_type="assistant")))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=FIRST, document_id="first-original", tags=["first-source"])
    await client.aretain(bank_id=bank_id, content=SECOND, document_id="second-original", tags=["second-source"])
    await settled(bank_id)

    source = await client.arecall(bank_id=bank_id, query=SECOND, tags=["second-source"], tags_match="all_strict")
    assert len(source.results) == 1
    expanded_id = source.results[0].id
    # Requested and cited alongside the real fact; expand reports it not found, so it
    # never becomes evidence.
    missing_id = str(uuid.uuid4())

    llm.on_step("reflect", tool="expand").answers_with_tool_call(
        _expand_then_cite([expanded_id, missing_id], answer=SECOND, cite=[expanded_id, missing_id])
    )
    reflect_loop(llm, answer=SECOND, query=QUERY)
    response = await client.areflect(
        bank_id=bank_id,
        query=QUERY,
        context=f"Read source {expanded_id} through expand before answering.",
        include_facts=True,
        include_tool_calls=True,
        exclude_mental_models=True,
        # Recall sees world facts only, so the experience fact reaches the model through
        # expand and nothing else.
        fact_types=["world"],
    )

    assert response.trace is not None
    recalled = {
        memory["id"]
        for call in response.trace.tool_calls
        if call.tool == "recall"
        for memory in call.output.get("memories", [])
    }
    assert recalled, "the forced recall returned nothing, so there was no fact to fall back to"
    assert expanded_id not in recalled, "the story needs a fact that only expand returned"
    assert any(call.tool == "expand" for call in response.trace.tool_calls)
    assert response.text == SECOND
    assert response.based_on is not None
    assert [memory.id for memory in response.based_on.memories] == [expanded_id]
    memory = response.based_on.memories[0]
    assert memory.type == "experience"
    assert memory.document_id == "second-original"
    assert memory.tags == ["second-source"]


async def test_a_shared_fact_from_a_hidden_document_is_cited_without_its_source_text(client, llm, bank_id, settled):
    private = "The source also contains a private staffing discussion."
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "entity_labels": [
                    {
                        "key": "kind",
                        "type": "multi-values",
                        "tag": True,
                        "optional": True,
                        "values": [{"value": "rule", "description": "a shared release rule"}],
                    }
                ]
            }
        },
    )
    # The fact is shared through its `kind:rule` tag; the document it came from keeps
    # only its private `user:kate` tag, which the reader cannot see.
    llm.on_step("extract_facts").returns(extracted(fact(SECOND, fact_type="assistant", entities=["kind:rule"])))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(
        bank_id=bank_id, content=f"{SECOND} {private}", document_id="private-original", tags=["user:kate"]
    )
    await settled(bank_id)
    scope = {"tags": ["kind:rule"], "tags_match": "all_strict"}
    source = await client.arecall(bank_id=bank_id, query=SECOND, types=["experience"], **scope)
    assert len(source.results) == 1
    expected = source.results[0]

    llm.on_step("reflect", tool="expand").answers_with_tool_call(
        _expand_then_cite([expected.id], answer=SECOND, cite=[expected.id])
    )
    reflect_loop(llm, answer=SECOND, query=QUERY)
    response = await client.areflect(
        bank_id=bank_id,
        query=QUERY,
        include_facts=True,
        include_tool_calls=True,
        exclude_mental_models=True,
        fact_types=["world"],
        **scope,
    )

    assert response.text == SECOND
    assert response.based_on is not None
    assert [memory.id for memory in response.based_on.memories] == [expected.id]
    # The citation carries the fact as recall shows it to this reader...
    memory = response.based_on.memories[0]
    assert memory.document_id == expected.document_id
    assert memory.chunk_id == expected.chunk_id
    assert memory.tags == expected.tags
    assert memory.metadata == expected.metadata
    assert memory.attachments == expected.attachments
    # ...and the hidden document's text reaches the caller nowhere.
    assert response.trace is not None
    expansion = next(call for call in response.trace.tool_calls if call.tool == "expand")
    item = expansion.output["results"][0]
    assert item["source_withheld"]
    assert "chunk" not in item and "document" not in item
    assert private not in response.model_dump_json()


async def test_an_expanded_id_whose_document_was_deleted_is_not_cited(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains=FIRST).returns(extracted(fact(FIRST)))
    llm.on_step("extract_facts", contains=SECOND).returns(extracted(fact(SECOND, fact_type="assistant")))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=FIRST, document_id="first-original")
    await client.aretain(bank_id=bank_id, content=SECOND, document_id="second-original")
    await settled(bank_id)
    source = await client.arecall(bank_id=bank_id, query=SECOND, types=["experience"])
    assert len(source.results) == 1
    removed_id = source.results[0].id
    await client.documents.delete_document(bank_id, "second-original")
    await settled(bank_id)
    assert not (await client.arecall(bank_id=bank_id, query=SECOND, types=["experience"])).results
    documents = await client.documents.list_documents(bank_id)
    assert [document.id for document in documents.items] == ["first-original"]

    llm.on_step("reflect", tool="expand").answers_with_tool_call(
        _expand_then_cite([removed_id], answer=FIRST, cite=[removed_id])
    )
    reflect_loop(llm, answer=FIRST, query=QUERY)
    response = await client.areflect(
        bank_id=bank_id,
        query=QUERY,
        include_facts=True,
        include_tool_calls=True,
        exclude_mental_models=True,
        fact_types=["world"],
    )

    assert response.text == FIRST
    assert response.based_on is not None
    assert removed_id not in [memory.id for memory in response.based_on.memories]
    assert response.trace is not None
    expansion = next(call for call in response.trace.tool_calls if call.tool == "expand")
    item = expansion.output["results"][0]
    assert item["memory_id"] == removed_id and item["error"]
    assert "memory" not in item and "chunk" not in item and "document" not in item
