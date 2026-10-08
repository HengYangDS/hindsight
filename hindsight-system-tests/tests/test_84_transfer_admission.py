"""Native transfer policy rejects old sources without disabling normal config.

The published client drives a real server/worker. The archive is exported
before policy changes, then imports/clones/exports are refused before queuing.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio


async def test_transfer_admission_keeps_config_reads_and_refuses_old_archive_source(
    transfer_policy_server, llm
) -> None:
    client = Hindsight(base_url=transfer_policy_server.url, timeout=30)
    source = f"systest-{uuid.uuid4().hex[:12]}"
    target = f"systest-{uuid.uuid4().hex[:12]}"
    denied_target = f"systest-{uuid.uuid4().hex[:12]}"
    anchor = f"systest-{uuid.uuid4().hex[:12]}"
    try:
        await client.acreate_bank(source)
        await client.acreate_bank(anchor)
        llm.on_step("extract_facts").returns(extracted(fact("Mira uses precise source citations", who="Mira")))
        llm.on_step("consolidate").returns(consolidation())
        await client.aretain(source, "Mira uses precise source citations.", document_id="source-note")
        await wait_until_settled(client, source)
        archive = await client.aexport_bank(source, poll_interval=0.1, timeout=60)
        document_archive = await client.aexport_documents(source, poll_interval=0.1, timeout=60)
        completed = (await client.operations.list_operations(source)).operations
        export_operation = next(item.id for item in completed if item.operation_type == "export_bank")
        export_status = await client.operations.get_operation_status(source, export_operation)
        download_key = export_status.result_metadata["storage_key"]
        await client.aupdate_bank_config(source, reflect_mission="transfer-withdrawn")
        config = await client.banks.get_bank_config(source)
        assert config.config["reflect_mission"] == "transfer-withdrawn"
        assert source in {bank.bank_id for bank in (await client.banks.list_banks()).banks}
        before = (await client.operations.list_operations(source)).model_dump(mode="json")

        with pytest.raises(ApiException) as clone:
            await client.aclone_bank(source, denied_target)
        assert clone.value.status == 409
        with pytest.raises(ApiException) as exported:
            await client.aexport_bank(source, poll_interval=0)
        assert exported.value.status == 409
        with pytest.raises(ApiException) as imported:
            await client.aimport_bank(anchor, archive, target_bank_id=denied_target)
        assert imported.value.status == 409
        with pytest.raises(ApiException) as document_import:
            await client.document_transfer.import_documents(anchor, document_archive)
        assert document_import.value.status == 409
        with pytest.raises(ApiException) as download:
            await client.document_transfer.download_file(download_key)
        assert download.value.status == 409
        assert (await client.documents.list_documents(anchor)).total == 0
        assert (await client.operations.list_operations(source)).model_dump(mode="json") == before
        assert denied_target not in {bank.bank_id for bank in (await client.banks.list_banks()).banks}

        # A native config edit re-admits only this source. The old archive still
        # has its original source identity; no archive rewrite is needed.
        await client.aupdate_bank_config(source, reflect_mission="transfer-allowed")
        operation_id = await client.aclone_bank(source, target)
        deadline = asyncio.get_running_loop().time() + 60
        while True:
            status = await client.operations.get_operation_status(source, operation_id)
            if status.status in {"completed", "failed", "cancelled"}:
                break
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.1)
        assert status.status == "completed", status
        assert target in {bank.bank_id for bank in (await client.banks.list_banks()).banks}
        assert (await client.banks.get_bank_config(target)).config["reflect_mission"] == "transfer-allowed"
        assert [document.id for document in (await client.documents.list_documents(target)).items] == ["source-note"]
    finally:
        for bank in (target, denied_target, anchor, source):
            await client.banks.delete_bank(bank)
        await client.aclose()
