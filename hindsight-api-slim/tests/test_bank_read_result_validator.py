"""Result admission is distinct from preflight and best-effort completion.

The original engine read methods run with a test-owned backend and byte store.
No model, database, listener, installed runtime or production read method is replaced.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.memory_engine import MemoryEngine
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


@dataclass
class StoredModel:
    """Native SQL row fixture, converted to a driver mapping at the boundary."""

    id: str
    bank_id: str
    content: str
    name: str = "Synthetic model"
    source_query: str = "Original synthetic source"
    tags: list[str] = field(default_factory=list)
    last_refreshed_at: datetime | None = None
    last_memory_seen_at: datetime | None = None
    last_refresh_failed_at: datetime | None = None
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    reflect_response: Any = None
    max_tokens: int | None = None
    trigger: Any = None
    structured_content: Any = None


@dataclass
class StoredPage:
    bank_id: str
    name: str
    mental_model_id: str
    created_at: datetime
    updated_at: datetime
    mm_tags: list[str]
    mm_source_query: str
    mm_content: str
    id: str = "synthetic-page"
    parent_id: str | None = None
    kind: str = "page"
    sort_order: int = 0
    managed: bool = False
    mm_trigger: Any = None
    mm_last_refreshed_at: datetime | None = None
    mm_last_refresh_failed_at: datetime | None = None


@dataclass
class StoredAttachment:
    attachment_hash: str = "synthetic-hash"
    short_id: str = "image"
    media_type: str = "image/png"
    byte_size: int = 23
    storage_key: str = "banks/source/image"
    kind: str = "image"
    filename: str = "synthetic.png"


@dataclass
class HistoricalContent:
    previous_content: str = "Complete historical source.\n"


@dataclass
class StoredHistory:
    content: HistoricalContent = field(default_factory=HistoricalContent)
    changed_at: datetime | None = None


@dataclass
class StoredDocument:
    id: str
    bank_id: str
    original_text: str = "Complete original document.\n"
    content_hash: str = "original-document-hash"
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    updated_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    tags: list[str] = field(default_factory=list)
    retain_params: str = json.dumps({"metadata": {"actor": "original-source"}})
    unit_count: int = 1
    world_count: int = 1
    experience_count: int = 0
    observation_count: int = 0


def _model(bank: str, identifier: str) -> StoredModel:
    return StoredModel(id=identifier, bank_id=bank, content=f"Complete {identifier} source.\n")


class ReadPolicy(OperationValidatorExtension):
    def __init__(self, rows: dict[str, list[StoredModel]]) -> None:
        super().__init__({})
        self.rows = rows
        self.epochs = dict.fromkeys(rows, 0)
        self.withdraw_after: str | None = None
        self.unavailable = False
        self.results: list[BankReadResultContext] = []

    def _pin(self, bank: str, context: RequestContext) -> None:
        epochs = getattr(context, "synthetic_read_epochs", None)
        if epochs is None:
            epochs = {}
            context.synthetic_read_epochs = epochs
        epochs.setdefault(bank, self.epochs[bank])

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        self._pin(ctx.bank_id, ctx.request_context)
        return ValidationResult.accept()

    async def validate_mental_model_get(self, ctx: MentalModelGetContext) -> ValidationResult:
        self._pin(ctx.bank_id, ctx.request_context)
        return ValidationResult.accept()

    async def on_mental_model_get_complete(self, result: MentalModelGetResult) -> None:
        if result.mental_model_id == self.withdraw_after:
            self.rows[result.bank_id] = [row for row in self.rows[result.bank_id] if row.id != result.mental_model_id]
            self.epochs[result.bank_id] += 1
            raise OperationValidationError("Synthetic completion observer refusal", 409)

    async def validate_bank_read_result(self, ctx: BankReadResultContext) -> ValidationResult:
        self.results.append(ctx)
        if self.unavailable:
            raise RuntimeError("Synthetic result policy unavailable")
        before = getattr(ctx.request_context, "synthetic_read_epochs", {}).get(ctx.bank_id)
        if before is not None and before != self.epochs[ctx.bank_id]:
            return ValidationResult.reject("Synthetic source changed during read", status_code=409)
        return ValidationResult.accept()


class Backend:
    _wraps_backend = True

    def __init__(self, rows: dict[str, list[StoredModel]]) -> None:
        self.rows = rows
        self.after_fetch: Callable[[str], None] | None = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Backend]:
        yield self

    async def fetchrow(self, query: str, bank: str, identifier: str) -> dict[str, Any] | None:
        model = next((row for row in self.rows[bank] if row.id == identifier), None)
        if model is None:
            return None
        row = asdict(model)
        if "knowledge_pages" not in query:
            return row
        return asdict(
            StoredPage(
                bank_id=bank,
                name=model.name,
                mental_model_id=model.id,
                created_at=model.created_at,
                updated_at=model.created_at,
                mm_tags=model.tags,
                mm_source_query=model.source_query,
                mm_content=model.content,
            )
        )

    async def fetch(self, query: str, bank: str, *args: Any) -> list[dict[str, Any]]:
        if "attachments" in query:
            return [asdict(StoredAttachment())]
        if "mental_model_history" in query:
            result = [asdict(StoredHistory())]
            if self.after_fetch:
                # Native history SQL binds the model id before the bank id.
                self.after_fetch(args[0])
            return result
        return [asdict(row) for row in self.rows[bank]]

    async def fetchval(self, query: str, bank: str, *args: Any) -> int:
        return len(self.rows[bank])


class DocumentBackend(Backend):
    """Return original SQL rows and commit a test-owned change on release."""

    def __init__(self, policy: ReadPolicy) -> None:
        super().__init__(policy.rows)
        self.policy = policy
        self.ops = PostgreSQLOps()
        self.change_on_release = False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DocumentBackend]:
        yield self
        if self.change_on_release:
            self.policy.epochs["source"] += 1

    def parse_json(self, value: str) -> Any:
        return json.loads(value)

    async def fetchrow(self, query: str, document: str, bank: str) -> dict[str, Any]:
        assert "FROM" in query and "documents" in query
        return asdict(StoredDocument(id=document, bank_id=bank))


@pytest.mark.asyncio
async def test_original_document_is_admitted_after_its_native_connection_releases() -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    backend = DocumentBackend(policy)
    engine._backend = backend
    engine._get_backend = AsyncMock(return_value=backend)
    context = RequestContext(internal=True)
    original = await engine.get_document("original-document", "source", request_context=context)
    assert original["original_text"] == "Complete original document.\n"
    assert original["retain_params"]["metadata"] == {"actor": "original-source"}
    peer_before = deepcopy(rows["peer"])
    backend.change_on_release = True
    with pytest.raises(OperationValidationError) as denied:
        await engine.get_document("original-document", "source", request_context=RequestContext(internal=True))
    assert denied.value.status_code == 409
    assert rows["peer"] == peer_before
    assert policy.results[-1].operation == "get_document"


@dataclass
class ReadHarness:
    engine: MemoryEngine
    policy: ReadPolicy
    rows: dict[str, list[StoredModel]]


def _engine() -> ReadHarness:
    rows = {"source": [_model("source", "first"), _model("source", "second")], "peer": [_model("peer", "peer-model")]}
    policy = ReadPolicy(rows)
    backend = Backend(rows)
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = policy
    engine._authenticate_tenant = AsyncMock(return_value="public")
    engine._get_backend = AsyncMock(return_value=backend)
    engine._require_bank_exists = AsyncMock()
    engine._get_bank_profile_authenticated = AsyncMock(side_effect=lambda bank, **kwargs: {"bank_id": bank})
    engine.compute_mental_model_is_stale = AsyncMock(return_value=False)
    return ReadHarness(engine=engine, policy=policy, rows=rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["model", "page", "list"])
async def test_completed_withdrawal_refuses_whole_native_result_and_preserves_peer(route: str) -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    peer_before = deepcopy(rows["peer"])
    policy.withdraw_after = "second" if route == "list" else "first"
    context = RequestContext(internal=True)
    with pytest.raises(OperationValidationError) as denied:
        if route == "model":
            await engine.get_mental_model("source", "first", detail="content", request_context=context)
        elif route == "page":
            await engine.get_knowledge_page("source", "first", request_context=context)
        else:
            await engine.list_mental_models("source", detail="content", request_context=context)
    assert denied.value.status_code == 409
    assert rows["peer"] == peer_before
    assert all(row.id != policy.withdraw_after for row in rows["source"])
    assert policy.results[-1].request_context is context
    peer = await engine.get_mental_model(
        "peer", "peer-model", detail="content", request_context=RequestContext(internal=True)
    )
    assert peer["content"] == peer_before[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["metadata", "config", "content", "full"])
async def test_accepting_result_policy_preserves_native_projection_and_default_behavior(detail: str) -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    context = RequestContext(internal=True)
    expected = engine._row_to_mental_model(asdict(rows["source"][0]), detail=detail)
    if detail == "full":
        expected["is_stale"] = False
    result = await engine.get_mental_model("source", "first", detail=detail, request_context=context)
    assert result == expected
    assert policy.results[-1].result is result
    assert policy.results[-1].operation == "get_mental_model"
    assert policy.results[-1].phase == "result"


@pytest.mark.asyncio
async def test_result_policy_failure_is_not_swallowed_as_observational_telemetry() -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    before = deepcopy(rows)
    policy.unavailable = True
    with pytest.raises(RuntimeError, match="result policy unavailable"):
        await engine.get_mental_model(
            "source", "first", detail="content", request_context=RequestContext(internal=True)
        )
    assert rows == before


@pytest.mark.asyncio
async def test_default_optional_result_hook_preserves_complete_content() -> None:
    harness = _engine()
    engine, rows = harness.engine, harness.rows

    class DefaultPolicy(OperationValidatorExtension):
        async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
            return ValidationResult.accept()

        async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
            return ValidationResult.accept()

    engine._operation_validator = DefaultPolicy({})
    result = await engine.get_mental_model(
        "source", "first", detail="content", request_context=RequestContext(internal=True)
    )
    assert result["content"] == rows["source"][0].content


@pytest.mark.asyncio
async def test_model_history_refuses_changed_source_without_rewriting_prior_content() -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    backend = await engine._get_backend()
    backend.after_fetch = lambda bank: policy.epochs.__setitem__(bank, policy.epochs[bank] + 1)
    before = deepcopy(rows)
    with pytest.raises(OperationValidationError) as denied:
        await engine.get_mental_model_history("source", "first", request_context=RequestContext(internal=True))
    assert denied.value.status_code == 409
    assert rows == before


@pytest.mark.asyncio
async def test_native_attachment_result_refuses_change_during_byte_read() -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    context = RequestContext(internal=True)
    policy._pin("source", context)

    async def load(key: str) -> bytes:
        policy.epochs["source"] += 1
        return b"complete-original-image"

    storage = MagicMock()
    storage.retrieve = load
    engine._file_storage = storage
    before = deepcopy(rows)
    with pytest.raises(OperationValidationError) as denied:
        await engine.retrieve_bank_attachment("source", "image", context)
    assert denied.value.status_code == 409
    assert rows == before


class ByteSource:
    def __init__(self, policy: ReadPolicy, *, withdraw_at: int | None = None) -> None:
        self.policy = policy
        self.withdraw_at = withdraw_at
        self.index = 0
        self.close_count = 0

    def __aiter__(self) -> ByteSource:
        return self

    async def __anext__(self) -> bytes:
        if self.index == 2:
            raise StopAsyncIteration
        if self.index == self.withdraw_at:
            self.policy.epochs["source"] += 1
        chunk = (b"original-first", b"original-second")[self.index]
        self.index += 1
        return chunk

    async def aclose(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_native_file_stream_refuses_next_chunk_after_source_change_and_closes_owner() -> None:
    harness = _engine()
    engine, policy, rows = harness.engine, harness.policy, harness.rows
    source = ByteSource(policy, withdraw_at=1)
    storage = MagicMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=29)
    storage.retrieve_stream = lambda key: source
    engine._file_storage = storage
    result = await engine.retrieve_bank_file_stream("source", "banks/source/export.zip", RequestContext(internal=True))
    assert await anext(result.stream) == b"original-first"
    with pytest.raises(OperationValidationError) as denied:
        await anext(result.stream)
    assert denied.value.status_code == 409
    assert source.close_count == 1
    assert rows["peer"] == [_model("peer", "peer-model")]


@pytest.mark.asyncio
async def test_caller_closing_unstarted_native_file_stream_closes_underlying_owner() -> None:
    harness = _engine()
    engine, policy = harness.engine, harness.policy
    source = ByteSource(policy)
    storage = MagicMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=29)
    storage.retrieve_stream = lambda key: source
    engine._file_storage = storage
    result = await engine.retrieve_bank_file_stream("source", "banks/source/export.zip", RequestContext(internal=True))
    await result.stream.aclose()
    assert source.close_count == 1
    assert source.index == 0


@pytest.mark.asyncio
async def test_cancelled_native_file_stream_closes_its_original_storage_owner() -> None:
    harness = _engine()

    class CancelledSource(ByteSource):
        async def __anext__(self) -> bytes:
            raise asyncio.CancelledError

    source = CancelledSource(harness.policy)
    storage = MagicMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=29)
    storage.retrieve_stream = lambda key: source
    harness.engine._file_storage = storage
    result = await harness.engine.retrieve_bank_file_stream(
        "source", "banks/source/export.zip", RequestContext(internal=True)
    )
    with pytest.raises(asyncio.CancelledError):
        await anext(result.stream)
    await result.stream.aclose()
    assert source.close_count == 1
    assert source.index == 0


@pytest.mark.asyncio
async def test_file_result_policy_failure_closes_unstarted_native_storage_owner() -> None:
    harness = _engine()
    engine, policy = harness.engine, harness.policy
    source = ByteSource(policy)
    storage = MagicMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=29)
    storage.retrieve_stream = lambda key: source
    engine._file_storage = storage
    policy.unavailable = True
    with pytest.raises(RuntimeError, match="result policy unavailable"):
        await engine.retrieve_bank_file_stream("source", "banks/source/export.zip", RequestContext(internal=True))
    assert source.close_count == 1
    assert source.index == 0


@pytest.mark.asyncio
async def test_result_context_is_exported_and_identity_is_immutable() -> None:
    from dataclasses import FrozenInstanceError

    from hindsight_api.extensions import BankReadOperation, BankReadResultContext

    result = {"content": "Complete unchanged source.\n"}
    request = RequestContext(internal=True)
    context = BankReadResultContext("source", BankReadOperation.GET_MENTAL_MODEL, request, result)
    assert context.result is result
    assert context.request_context is request
    with pytest.raises(FrozenInstanceError):
        context.bank_id = "peer"
