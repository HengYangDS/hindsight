"""The published client never receives a read whose source changed mid-read.

Real retain, worker consolidation and page generation precede withdrawal.
The server's original operation extension then deletes the source model in
its completion callback; result admission rejects model, page and list reads
without rewriting their bodies or changing the complete independent peer.
"""

from __future__ import annotations

import json
import uuid

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import HindsightServer, LLMStub, reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("read", ["model", "page", "list"])
async def test_completed_source_withdrawal_refuses_native_result_and_preserves_peer(
    read_result_policy_server: HindsightServer, llm: LLMStub, read: str
) -> None:
    client = Hindsight(base_url=read_result_policy_server.url, timeout=30)
    source, peer = f"systest-{uuid.uuid4().hex[:12]}", f"systest-{uuid.uuid4().hex[:12]}"
    source_answer = "Alice lives in Berlin."
    peer_answer = "Bob lives in Paris."
    try:
        await client.acreate_bank(source)
        await client.acreate_bank(peer)
        llm.on_step("extract_facts").returns(extracted(fact(source_answer, who="Alice", entities=["Alice", "Berlin"])))
        llm.on_step("consolidate").returns(consolidation())
        reflect_loop(llm, answer=source_answer)
        await client.aretain(source, source_answer, document_id="source-note")
        await wait_until_settled(client, source)
        source_page = await client.knowledge_base.create_knowledge_page(
            source,
            {
                "name": "Alice housing",
                "source_query": "Where does Alice live?",
                # This story retains original facts. Observation-only is the
                # page default, so declare the native source scope explicitly.
                "trigger": {"fact_types": ["world", "experience"]},
            },
        )
        await wait_until_settled(client, source)

        llm.reset()
        llm.on_step("extract_facts").returns(extracted(fact(peer_answer, who="Bob", entities=["Bob", "Paris"])))
        llm.on_step("consolidate").returns(consolidation())
        reflect_loop(llm, answer=peer_answer)
        await client.aretain(peer, peer_answer, document_id="peer-note")
        await wait_until_settled(client, peer)
        peer_page = await client.knowledge_base.create_knowledge_page(
            peer,
            {
                "name": "Bob housing",
                "source_query": "Where does Bob live?",
                "trigger": {"fact_types": ["world", "experience"]},
            },
        )
        await wait_until_settled(client, peer)
        source_before = await client.mental_models.get_mental_model(source, source_page.mental_model_id, detail="full")
        peer_before = await client.mental_models.get_mental_model(peer, peer_page.mental_model_id, detail="full")
        assert source_before.content == source_answer + "\n"
        assert peer_before.content == peer_answer + "\n"

        # Further reads must not generate anything; every background operation
        # above is settled before its rulebook is released.
        llm.reset()
        await client.aupdate_bank_config(source, reflect_mission="read-withdraw-once")
        with pytest.raises(ApiException) as denied:
            if read == "model":
                await client.mental_models.get_mental_model(source, source_page.mental_model_id, detail="full")
            elif read == "page":
                await client.knowledge_base.get_knowledge_page(source, source_page.page_id)
            else:
                await client.mental_models.list_mental_models(source, detail="full")
        assert denied.value.status == 409
        assert json.loads(denied.value.body) == {"detail": "source changed during read"}
        config = await client.banks.get_bank_config(source)
        assert config.config["reflect_mission"] == "read-withdrawn"
        with pytest.raises(ApiException) as removed:
            await client.mental_models.get_mental_model(source, source_page.mental_model_id, detail="full")
        assert removed.value.status == 404
        peer_after = await client.mental_models.get_mental_model(peer, peer_page.mental_model_id, detail="full")
        assert peer_after.model_dump(mode="json") == peer_before.model_dump(mode="json")
    finally:
        await client.banks.delete_bank(source)
        await client.banks.delete_bank(peer)
        await client.aclose()
