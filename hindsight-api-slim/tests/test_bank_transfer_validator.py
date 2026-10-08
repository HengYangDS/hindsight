"""Transfer-specific admission must not widen a caller's ordinary bank scope.

These are deterministic engine guard tests. Database/ZIP effects are mocked
because a rejection must occur before any of them; native lifecycle tests cover
the actual storage, worker and HTTP boundaries separately.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine, bank_utils
from hindsight_api.engine.transfer import TransferScope
from hindsight_api.engine.transfer.schema import TransferDocument, TransferManifest
from hindsight_api.extensions import OperationValidationError, OperationValidatorExtension, ValidationResult

if TYPE_CHECKING:
    from hindsight_api.extensions import BankReadContext, BankTransferContext


class RecordingTransferValidator(OperationValidatorExtension):
    """Refuse one source's transfers, without refusing its normal bank reads."""

    def __init__(self, revoked_bank_id: str | None = "withdrawn-source") -> None:
        super().__init__({})
        self.revoked_bank_id = revoked_bank_id
        self.transfers: list[BankTransferContext] = []

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_bank_transfer(self, ctx: BankTransferContext) -> ValidationResult:
        self.transfers.append(ctx)
        if self.revoked_bank_id is not None and self.revoked_bank_id in (ctx.source_bank_id, ctx.target_bank_id):
            return ValidationResult.reject("source transfer withdrawn", status_code=409)
        return ValidationResult.accept()

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        if ctx.operation == "retrieve_bank_file_stream" and ctx.bank_id == self.revoked_bank_id:
            return ValidationResult.reject("source transfer withdrawn", status_code=409)
        return ValidationResult.accept()


@dataclass
class TransferHarness:
    engine: MemoryEngine
    validator: RecordingTransferValidator
    storage: MagicMock
    submit: AsyncMock


@pytest.fixture
def transfer_harness(monkeypatch: pytest.MonkeyPatch) -> TransferHarness:
    """Use the real public engine entrypoints, without a DB or model process."""
    validator = RecordingTransferValidator()
    engine = object.__new__(MemoryEngine)
    engine._operation_validator = validator
    engine._authenticate_tenant = AsyncMock(return_value="public")
    engine._refuse_bank_wide_if_scoped = AsyncMock()
    engine._backend = MagicMock()
    engine._get_backend = AsyncMock(return_value=engine._backend)
    submit = AsyncMock(return_value={"operation_id": "synthetic-operation"})
    engine._submit_async_operation = submit
    storage = MagicMock()
    storage.store = AsyncMock()
    storage.retrieve = AsyncMock()
    storage.store_stream = AsyncMock()
    storage.exists = AsyncMock(return_value=True)
    storage.get_size = AsyncMock(return_value=0)
    engine._file_storage = storage
    engine.get_bank_profile = AsyncMock(return_value={"bank_id": "withdrawn-source"})
    engine.ensure_bank_profile = AsyncMock()
    engine._ensure_bank = AsyncMock()
    engine._ensure_bank_exists = AsyncMock()
    monkeypatch.setattr(bank_utils, "get_bank_profile_if_exists", AsyncMock(return_value=None))
    return TransferHarness(engine=engine, validator=validator, storage=storage, submit=submit)


def _archive(*, source_bank_id: str, whole_bank: bool) -> bytes:
    """Build the real declared native manifest, including the old scope form."""
    manifest = TransferManifest(source_bank_id=source_bank_id, archive_type="bank" if whole_bank else "documents")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", manifest.model_dump_json())
        document = TransferDocument(id="source-note", original_text="Original source body")
        archive.writestr("documents/source-note.json", document.model_dump_json())
        if whole_bank:
            archive.writestr("banks.json", "[]")
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["clone_bank", "export_bank", "export_documents"])
async def test_transfer_rejected_before_operation_submission(transfer_harness: TransferHarness, operation: str) -> None:
    harness = transfer_harness
    context = RequestContext(tenant_id="synthetic-tenant", api_key_id="synthetic-key-id")
    with pytest.raises(OperationValidationError, match="source transfer withdrawn") as error:
        if operation == "clone_bank":
            await harness.engine.submit_bank_clone_async("withdrawn-source", "fresh-target", context)
        elif operation == "export_bank":
            await harness.engine.submit_bank_export_async("withdrawn-source", context)
        else:
            await harness.engine.submit_export_documents_async("withdrawn-source", context, ["doc-1"])

    assert error.value.status_code == 409
    harness.submit.assert_not_awaited()
    harness.storage.store.assert_not_awaited()
    harness.storage.store_stream.assert_not_awaited()
    assert len(harness.validator.transfers) == 1
    admission = harness.validator.transfers[0]
    assert admission.operation == operation
    assert admission.source_bank_id == "withdrawn-source"
    assert admission.target_bank_id == ("fresh-target" if operation == "clone_bank" else None)
    assert admission.request_context is context


@pytest.mark.asyncio
@pytest.mark.parametrize("whole_bank", [True, False])
async def test_import_preserves_original_archive_source_before_storage_or_queue(
    transfer_harness: TransferHarness, whole_bank: bool
) -> None:
    harness = transfer_harness
    archive = _archive(source_bank_id="withdrawn-source", whole_bank=whole_bank)
    context = RequestContext(tenant_id="synthetic-tenant", api_key_id="synthetic-key-id")
    with pytest.raises(OperationValidationError, match="source transfer withdrawn"):
        if whole_bank:
            await harness.engine.submit_bank_import_async(
                "request-bank", archive, context, target_bank_id="fresh-target"
            )
        else:
            await harness.engine.import_documents_async("fresh-target", archive, context)

    harness.storage.store.assert_not_awaited()
    harness.submit.assert_not_awaited()
    admission = harness.validator.transfers[0]
    assert admission.source_bank_id == "withdrawn-source"
    assert admission.target_bank_id == "fresh-target"
    assert admission.operation == ("import_bank" if whole_bank else "import_documents")
    if not whole_bank:
        assert admission.document_ids == ["source-note"]
    assert admission.request_context is context


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["clone_bank", "export_bank", "export_documents"])
async def test_queued_transfer_rechecks_current_admission_before_archive_effects(
    transfer_harness: TransferHarness, operation: str
) -> None:
    harness = transfer_harness
    harness.engine._get_backend.side_effect = AssertionError("archive backend reached before transfer rejection")
    task = {
        "bank_id": "withdrawn-source",
        "target_bank_id": "fresh-target",
        "_tenant_id": "synthetic-tenant",
        "_api_key_id": "synthetic-key-id",
        "_retry_count": 1,
    }
    with pytest.raises(OperationValidationError, match="source transfer withdrawn"):
        if operation == "clone_bank":
            await harness.engine._handle_clone_bank(task)
        elif operation == "export_bank":
            await harness.engine._handle_export_bank(task)
        else:
            await harness.engine._handle_export_documents(task)

    harness.engine._get_backend.assert_not_awaited()
    harness.storage.store_stream.assert_not_awaited()
    admission = harness.validator.transfers[0]
    assert admission.operation == operation
    assert admission.source_bank_id == "withdrawn-source"
    assert admission.request_context.internal is True
    assert admission.request_context.tenant_id == "synthetic-tenant"
    assert admission.request_context.api_key_id == "synthetic-key-id"


@pytest.mark.asyncio
@pytest.mark.parametrize("whole_bank", [True, False])
async def test_internal_import_rechecks_archive_source_before_restore_effects(
    transfer_harness: TransferHarness, whole_bank: bool
) -> None:
    harness = transfer_harness
    harness.engine._get_backend.side_effect = AssertionError("restore backend reached before transfer rejection")
    archive = _archive(source_bank_id="withdrawn-source", whole_bank=whole_bank)
    context = RequestContext(internal=True, tenant_id="synthetic-tenant", api_key_id="synthetic-key-id")
    with pytest.raises(OperationValidationError, match="source transfer withdrawn"):
        if whole_bank:
            await harness.engine.import_bank_async(archive, context, target_bank_id="fresh-target")
        else:
            await harness.engine._run_import_documents("fresh-target", archive, "replace", context)

    harness.engine._get_backend.assert_not_awaited()
    admission = harness.validator.transfers[0]
    assert admission.source_bank_id == "withdrawn-source"
    assert admission.target_bank_id == "fresh-target"
    assert admission.request_context is context
    if not whole_bank:
        assert admission.document_ids == ["source-note"]


@pytest.mark.asyncio
async def test_legitimate_clone_keeps_ordinary_config_and_exact_scope(transfer_harness: TransferHarness) -> None:
    harness = transfer_harness
    context = RequestContext()
    scope = TransferScope(data=False, bank_config=True, history=False)
    result = await harness.engine.submit_bank_clone_async("healthy-source", "healthy-target", context, scope=scope)

    assert result == {"operation_id": "synthetic-operation"}
    harness.submit.assert_awaited_once()
    assert len(harness.validator.transfers) == 1
    admission = harness.validator.transfers[0]
    assert admission.source_bank_id == "healthy-source"
    assert admission.target_bank_id == "healthy-target"
    assert admission.scope == scope
    harness.engine._refuse_bank_wide_if_scoped.assert_awaited_once()


@pytest.mark.asyncio
async def test_old_stored_archive_download_has_its_own_read_operation(transfer_harness: TransferHarness) -> None:
    harness = transfer_harness
    with pytest.raises(OperationValidationError, match="source transfer withdrawn"):
        await harness.engine.retrieve_bank_file_stream(
            "withdrawn-source", "banks/withdrawn-source/exports/old/transfer.zip", RequestContext()
        )

    harness.storage.exists.assert_not_awaited()
    harness.storage.retrieve_stream.assert_not_called()


@pytest.mark.asyncio
async def test_direct_document_export_rejects_before_archive_build(
    transfer_harness: TransferHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = transfer_harness
    exporter = AsyncMock(return_value=b"synthetic archive must not be read")
    monkeypatch.setattr("hindsight_api.engine.transfer.export_documents", exporter)
    with pytest.raises(OperationValidationError, match="source transfer withdrawn"):
        await harness.engine.export_documents_async("withdrawn-source", RequestContext())
    exporter.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_validator_keeps_existing_transfer_behavior(transfer_harness: TransferHarness) -> None:
    harness = transfer_harness
    harness.validator.revoked_bank_id = None
    result = await harness.engine.submit_bank_export_async("healthy-source", RequestContext())
    assert result == {"operation_id": "synthetic-operation"}
    assert len(harness.validator.transfers) == 1
