"""Native PostgreSQL/HTTP refusal after a completed source withdrawal.

Uses the suite's original engine/database fixtures and only generated test
banks plus a test-owned tombstone table. No engine method or ASGI middleware
is replaced. Completion observation remains best-effort by design.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from hindsight_api import RequestContext
from hindsight_api.api.http import create_app
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memory_engine import MemoryEngine, get_current_schema
from hindsight_api.extensions import (
    BankReadContext,
    BankReadResultContext,
    MentalModelGetContext,
    MentalModelGetResult,
    OperationValidationError,
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)


@pytest.fixture
def cross_encoder() -> RRFPassthroughCrossEncoder:
    """Result admission does not rerank; use the shipped native passthrough."""
    return RRFPassthroughCrossEncoder()


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["model", "page", "list", "http"])
async def test_completed_withdrawal_rejects_native_response_and_preserves_complete_peer(
    memory_no_llm_verify: MemoryEngine, read: str
) -> None:
    engine = memory_no_llm_verify
    suffix = uuid.uuid4().hex[:12]
    source, peer = f"result-source-{suffix}", f"result-peer-{suffix}"
    model, peer_model = f"result-model-{suffix}", f"result-peer-model-{suffix}"
    schema = get_current_schema()
    table = f'"{schema}"."result_withdrawals_{suffix}"'
    request = RequestContext(internal=True)
    original_validator = engine._operation_validator
    original_get = engine.get_mental_model.__func__
    assert original_get is MemoryEngine.get_mental_model
    backend = await engine._get_backend()
    async with backend.acquire() as conn:
        await conn.execute(f"CREATE TABLE {table} (bank_id text PRIMARY KEY)")
        armed = False

        class Policy(OperationValidatorExtension):
            async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
                return ValidationResult.accept()

            async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
                return ValidationResult.accept()

            async def _pin(self, bank: str, context: RequestContext) -> None:
                count = await conn.fetchval(f"SELECT count(*) FROM {table} WHERE bank_id=$1", bank)
                epochs = getattr(context, "result_test_epochs", None)
                if epochs is None:
                    epochs = {}
                    context.result_test_epochs = epochs
                epochs.setdefault(bank, count)

            async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
                await self._pin(ctx.bank_id, ctx.request_context)
                return ValidationResult.accept()

            async def validate_mental_model_get(self, ctx: MentalModelGetContext) -> ValidationResult:
                await self._pin(ctx.bank_id, ctx.request_context)
                return ValidationResult.accept()

            async def on_mental_model_get_complete(self, result: MentalModelGetResult) -> None:
                nonlocal armed
                if not armed or result.bank_id != source:
                    return
                armed = False
                await conn.execute(f"INSERT INTO {table} VALUES ($1)", source)
                await engine.delete_mental_model(source, model, request_context=request)
                raise OperationValidationError("Synthetic completion refusal remains observational", 409)

            async def validate_bank_read_result(self, ctx: BankReadResultContext) -> ValidationResult:
                before = getattr(ctx.request_context, "result_test_epochs", {}).get(ctx.bank_id)
                current = await conn.fetchval(f"SELECT count(*) FROM {table} WHERE bank_id=$1", ctx.bank_id)
                if before is not None and before != current:
                    return ValidationResult.reject("Synthetic completed withdrawal refused", status_code=409)
                return ValidationResult.accept()

        try:
            engine._operation_validator = Policy({})
            page = await engine.create_knowledge_page(
                source,
                "Synthetic source page",
                "Synthetic source identity",
                "Complete original source.\n",
                mental_model_id=model,
                request_context=request,
            )
            await engine.create_knowledge_page(
                peer,
                "Synthetic independent page",
                "Independent source identity",
                "Complete independent source.\n",
                mental_model_id=peer_model,
                request_context=request,
            )
            original_body = await conn.fetchval(
                f'SELECT content FROM "{schema}".mental_models WHERE bank_id=$1 AND id=$2', source, model
            )
            # Public reads cannot prove complete physical-row preservation.
            peer_before = await conn.fetchval(
                f'SELECT to_jsonb(m)::text FROM "{schema}".mental_models m WHERE bank_id=$1 AND id=$2', peer, peer_model
            )
            assert original_body == "Complete original source.\n"
            armed = True
            context = RequestContext(internal=True)
            if read == "http":
                app = create_app(engine, initialize_memory=False)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.get(f"/v1/default/banks/{source}/mental-models/{model}?detail=content")
                assert response.status_code == 409
                assert response.json() == {"detail": "Synthetic completed withdrawal refused"}
                assert original_body not in response.text
            else:
                with pytest.raises(OperationValidationError) as denied:
                    if read == "model":
                        await engine.get_mental_model(source, model, detail="content", request_context=context)
                    elif read == "page":
                        await engine.get_knowledge_page(source, page["id"], request_context=context)
                    else:
                        await engine.list_mental_models(source, detail="content", request_context=context)
                assert denied.value.status_code == 409
            assert await conn.fetchval(f"SELECT count(*) FROM {table} WHERE bank_id=$1", source) == 1
            assert not await conn.fetchval(
                f'SELECT 1 FROM "{schema}".mental_models WHERE bank_id=$1 AND id=$2', source, model
            )
            assert (
                await conn.fetchval(
                    f'SELECT to_jsonb(m)::text FROM "{schema}".mental_models m WHERE bank_id=$1 AND id=$2',
                    peer,
                    peer_model,
                )
                == peer_before
            )
            delivered_peer = await engine.get_mental_model(
                peer, peer_model, detail="content", request_context=RequestContext(internal=True)
            )
            assert delivered_peer["content"] == "Complete independent source.\n"
            assert engine.get_mental_model.__func__ is original_get
        finally:
            engine._operation_validator = original_validator
            for bank in (source, peer):
                await engine.delete_bank(bank, request_context=RequestContext(internal=True))
            await conn.execute(f"DROP TABLE {table}")


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["direct", "http"])
async def test_original_document_result_uses_native_policy_after_database_read(
    memory_no_llm_verify: MemoryEngine, read: str
) -> None:
    engine = memory_no_llm_verify
    suffix = uuid.uuid4().hex[:12]
    bank, document = f"original-document-bank-{suffix}", f"original-document-{suffix}"
    request = RequestContext(internal=True)
    schema = get_current_schema()
    backend = await engine._get_backend()
    original_validator = engine._operation_validator
    original_method = engine.get_document.__func__
    assert original_method is MemoryEngine.get_document
    await engine.ensure_bank_profile(bank, request_context=request)
    async with backend.acquire() as connection:
        await connection.execute(
            f'INSERT INTO "{schema}".documents(id,bank_id,original_text,content_hash,retain_params) '
            "VALUES ($1,$2,$3,$4,$5::jsonb)",
            document,
            bank,
            "Complete original document.\n",
            "complete-original-hash",
            json.dumps({"metadata": {"actor": "original-source"}}),
        )
        # Pin the complete physical preimage, including native internal columns.
        original_row = await connection.fetchval(
            f'SELECT to_jsonb(d)::text FROM "{schema}".documents d WHERE bank_id=$1 AND id=$2', bank, document
        )

    class ResultPolicy(OperationValidatorExtension):
        def __init__(self) -> None:
            super().__init__({})
            self.reject = False
            self.results: list[BankReadResultContext] = []

        async def validate_retain(self, context: RetainContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_recall(self, context: RecallContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_reflect(self, context: ReflectContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_bank_read_result(self, context: BankReadResultContext) -> ValidationResult:
            self.results.append(context)
            if self.reject and context.bank_id == bank:
                return ValidationResult.reject("Synthetic source generation changed", status_code=409)
            return ValidationResult.accept()

    policy = ResultPolicy()
    engine._operation_validator = policy
    try:
        accepted = await engine.get_document(document, bank, request_context=request)
        assert accepted["original_text"] == "Complete original document.\n"
        assert accepted["document_metadata"] == {"actor": "original-source"}
        assert policy.results[-1].result is accepted
        policy.reject = True
        if read == "direct":
            with pytest.raises(OperationValidationError) as denied:
                await engine.get_document(document, bank, request_context=request)
            assert denied.value.status_code == 409
        else:
            app = create_app(engine, initialize_memory=False)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get(f"/v1/default/banks/{bank}/documents/{document}")
            assert response.status_code == 409
            assert response.json() == {"detail": "Synthetic source generation changed"}
            assert accepted["original_text"] not in response.text
        async with backend.acquire() as connection:
            assert (
                await connection.fetchval(
                    f'SELECT to_jsonb(d)::text FROM "{schema}".documents d WHERE bank_id=$1 AND id=$2', bank, document
                )
                == original_row
            )
        assert engine.get_document.__func__ is original_method
    finally:
        engine._operation_validator = original_validator
        await engine.delete_bank(bank, request_context=request)
