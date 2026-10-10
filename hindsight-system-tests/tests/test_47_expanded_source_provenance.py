"""A successfully expanded raw source is citable without a prior recall hit.

The client supplies an authorized source ID in context. The native agent reads
that source through expand and names it in done. Evidence must survive the real
HTTP response, stored-provenance hydration, and the published SDK decoder.
"""

from __future__ import annotations

import uuid

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest, ToolCall

pytestmark = pytest.mark.asyncio

FIRST = "Alice lives in Berlin."
SECOND = "The deployment recovery procedure requires an original-source check."
QUERY = "Alice Berlin"


async def test_expanded_source_reaches_public_based_on_without_a_recall_hit(client, llm, bank_id, settled):
    llm.on_step("extract_facts", contains=FIRST).returns(extracted(fact(FIRST)))
    llm.on_step("extract_facts", contains=SECOND).returns(extracted(fact(SECOND, fact_type="assistant")))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=FIRST, document_id="first-original", tags=["first-source"])
    await client.aretain(bank_id=bank_id, content=SECOND, document_id="second-original", tags=["second-source"])
    await settled(bank_id)

    source = await client.arecall(bank_id=bank_id, query=SECOND, tags=["second-source"], tags_match="all_strict")
    assert len(source.results) == 1
    expanded_id = source.results[0].id
    missing_id = str(uuid.uuid4())

    def expand_then_done(request: ChatRequest) -> ToolCall:
        if any(
            call.get("function", {}).get("name") == "expand"
            for message in request.messages
            for call in message.get("tool_calls", [])
        ):
            return ToolCall("done", {"answer": SECOND, "memory_ids": [expanded_id, missing_id]})
        return ToolCall("expand", {"memory_ids": [expanded_id, missing_id], "depth": "document"})

    llm.on_step("reflect", tool="expand").answers_with_tool_call(expand_then_done)
    reflect_loop(llm, answer=SECOND, query=QUERY)
    response = await client.areflect(
        bank_id=bank_id,
        query=QUERY,
        context=f"Read authorized source {expanded_id} through expand before answering.",
        include_facts=True,
        include_tool_calls=True,
        exclude_mental_models=True,
        fact_types=["world"],
    )

    assert response.trace is not None
    recalled = {
        memory["id"]
        for call in response.trace.tool_calls
        if call.tool == "recall"
        for memory in call.output.get("memories", [])
    }
    assert recalled, "The native retrieval ladder was not exercised."
    assert expanded_id not in recalled, "The fixture must exercise expansion without prior recall registration."
    assert response.text == SECOND
    assert response.based_on is not None
    assert [memory.id for memory in response.based_on.memories] == [expanded_id]
    memory = response.based_on.memories[0]
    assert memory.type == "experience"
    assert memory.document_id == "second-original"
    assert memory.tags == ["second-source"]
    assert any(call.tool == "expand" for call in response.trace.tool_calls)


async def test_visible_expanded_fact_keeps_native_provenance_without_opening_hidden_source(
    client, llm, bank_id, settled
):
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
    # The fact's shared tag does not grant access to the original's private tag.
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

    def expand_then_done(request: ChatRequest) -> ToolCall:
        if any(
            call.get("function", {}).get("name") == "expand"
            for message in request.messages
            for call in message.get("tool_calls", [])
        ):
            return ToolCall("done", {"answer": SECOND, "memory_ids": [expected.id]})
        return ToolCall("expand", {"memory_ids": [expected.id], "depth": "document"})

    llm.on_step("reflect", tool="expand").answers_with_tool_call(expand_then_done)
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
    memory = response.based_on.memories[0]
    assert memory.document_id == expected.document_id
    assert memory.chunk_id == expected.chunk_id
    assert memory.tags == expected.tags
    assert memory.metadata == expected.metadata
    assert memory.attachments == expected.attachments
    expansion = next(call for call in response.trace.tool_calls if call.tool == "expand")
    item = expansion.output["results"][0]
    assert item["source_withheld"]
    assert "chunk" not in item and "document" not in item
    assert private not in response.model_dump_json()


async def test_deleted_expanded_source_is_not_recited_from_a_stale_selector(client, llm, bank_id, settled):
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

    def expand_then_done(request: ChatRequest) -> ToolCall:
        if any(
            call.get("function", {}).get("name") == "expand"
            for message in request.messages
            for call in message.get("tool_calls", [])
        ):
            return ToolCall("done", {"answer": FIRST, "memory_ids": [removed_id]})
        return ToolCall("expand", {"memory_ids": [removed_id], "depth": "document"})

    llm.on_step("reflect", tool="expand").answers_with_tool_call(expand_then_done)
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
    expansion = next(call for call in response.trace.tool_calls if call.tool == "expand")
    item = expansion.output["results"][0]
    assert item["memory_id"] == removed_id and item["error"]
    assert "memory" not in item and "chunk" not in item and "document" not in item
