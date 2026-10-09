"""Test-only completed withdrawal through the original extension contract.

A bank config change arms one native model read. Its completion callback
deletes that model, commits a new policy revision and deliberately raises an
observational error. The separate result hook must refuse the assembled read.
No engine method, HTTP stack or shared request state is replaced.
"""

from hindsight_api import RequestContext
from hindsight_api.extensions import (
    BankReadContext,
    BankReadOperation,
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


class ReadResultPolicy(OperationValidatorExtension):
    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def _mission(self, bank_id: str, context: RequestContext) -> str | None:
        config = await self.context.get_memory_engine().get_bank_config(bank_id, request_context=context)
        return config.config.get("reflect_mission")

    async def _pin(self, bank_id: str, context: RequestContext) -> None:
        missions = getattr(context, "read_result_policy_missions", None)
        if missions is None:
            missions = {}
            context.read_result_policy_missions = missions
        if bank_id not in missions:
            missions[bank_id] = await self._mission(bank_id, context)

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        # Configuration reads are the policy input, not a recursive policy read.
        if ctx.operation in {BankReadOperation.LIST_MENTAL_MODELS, BankReadOperation.GET_KNOWLEDGE_PAGE}:
            await self._pin(ctx.bank_id, ctx.request_context)
        return ValidationResult.accept()

    async def validate_mental_model_get(self, ctx: MentalModelGetContext) -> ValidationResult:
        await self._pin(ctx.bank_id, ctx.request_context)
        return ValidationResult.accept()

    async def on_mental_model_get_complete(self, result: MentalModelGetResult) -> None:
        if await self._mission(result.bank_id, result.request_context) != "read-withdraw-once":
            return
        engine = self.context.get_memory_engine()
        await engine.update_bank_config(
            result.bank_id, {"reflect_mission": "read-withdrawn"}, request_context=result.request_context
        )
        await engine.delete_mental_model(result.bank_id, result.mental_model_id, request_context=result.request_context)
        raise OperationValidationError("Synthetic completion refusal is observational", 409)

    async def validate_bank_read_result(self, ctx: BankReadResultContext) -> ValidationResult:
        missions = getattr(ctx.request_context, "read_result_policy_missions", {})
        if ctx.bank_id in missions and missions[ctx.bank_id] != await self._mission(ctx.bank_id, ctx.request_context):
            return ValidationResult.reject("source changed during read", status_code=409)
        return ValidationResult.accept()
