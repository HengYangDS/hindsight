"""Test-only transfer policy loaded through the server's native extension slot.

A bank mission is the policy input, so a caller can change it after an export
without rewriting the archive. Ordinary reads/configuration stay available.
"""

from hindsight_api.extensions import BankReadContext, BankTransferContext, OperationValidatorExtension, ValidationResult


class TransferPolicy(OperationValidatorExtension):
    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def _withdrawn(self, bank_id: str, request_context) -> bool:
        config = await self.context.get_memory_engine().get_bank_config(bank_id, request_context=request_context)
        return config.config.get("reflect_mission") == "transfer-withdrawn"

    async def validate_bank_transfer(self, ctx: BankTransferContext) -> ValidationResult:
        if await self._withdrawn(ctx.source_bank_id, ctx.request_context):
            return ValidationResult.reject("source transfer withdrawn", status_code=409)
        return ValidationResult.accept()

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        if ctx.operation == "retrieve_bank_file_stream" and await self._withdrawn(ctx.bank_id, ctx.request_context):
            return ValidationResult.reject("source transfer withdrawn", status_code=409)
        return ValidationResult.accept()
