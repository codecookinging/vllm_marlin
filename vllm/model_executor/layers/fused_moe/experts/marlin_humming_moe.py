# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin/Humming hybrid MoE experts for w4a16 MXFP4."""

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingGroupedExperts,
    HummingIndexedExperts,
    get_humming_moe_gemm_type,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)


def _humming_experts_cls() -> type[mk.FusedMoEExpertsModular]:
    if get_humming_moe_gemm_type() == "indexed":
        return HummingIndexedExperts
    return HummingGroupedExperts


def prepare_moe_mxfp4_layer_for_marlin_or_humming(
    layer: "RoutedExperts",
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    quant_method: str = "mxfp4",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Prepare Marlin and Humming weight layouts for runtime dispatch.

    Marlin weights are stored on ``layer._marlin_*`` buffers. The layer is
    transformed in-place to Humming format; returned tensors are the Humming
    weights to assign as the layer's primary parameters.
    """
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        prepare_humming_moe_layer,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )

    (
        marlin_w13,
        marlin_w2,
        marlin_w13_scale,
        marlin_w2_scale,
        marlin_w13_bias,
        marlin_w2_bias,
    ) = prepare_moe_mxfp4_layer_for_marlin(
        layer,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w13_bias,
        w2_bias,
    )

    layer.register_buffer("_marlin_w13_weight", marlin_w13)
    layer.register_buffer("_marlin_w2_weight", marlin_w2)
    layer.register_buffer("_marlin_w13_weight_scale", marlin_w13_scale)
    layer.register_buffer("_marlin_w2_weight_scale", marlin_w2_scale)
    if marlin_w13_bias is not None:
        layer.register_buffer("_marlin_w13_bias", marlin_w13_bias)
    if marlin_w2_bias is not None:
        layer.register_buffer("_marlin_w2_bias", marlin_w2_bias)

    prepare_humming_moe_layer(layer, {"quant_method": quant_method})

    return (
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
    )


class MarlinOrHummingExperts(FallbackExperts):
    """Use Marlin for small batches and Humming for large batches (w4a16)."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        marlin_quant_config: FusedMoEQuantConfig,
        layer: "RoutedExperts",
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        self.layer = layer
        self._threshold = envs.VLLM_MARLIN_HUMMING_M_THRESHOLD
        humming_cls = _humming_experts_cls()
        marlin_experts = MarlinExperts(
            moe_config=moe_config,
            quant_config=marlin_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        humming_experts = humming_cls(
            layer=layer,
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        super().__init__(
            experts=marlin_experts,
            fallback_experts=humming_experts,
        )
        # Input quantization follows the Humming quant config (w4a16).
        self.quant_config = quant_config
        logger.info_once(
            "Using Marlin for M<=%d and Humming for M>%d MoE batches.",
            self._threshold,
            self._threshold,
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
    ]:
        return (MarlinExperts, _humming_experts_cls())

    @staticmethod
    def supports_lora() -> bool:
        return False

    def _should_use_marlin(self, hidden_states: torch.Tensor) -> bool:
        return hidden_states.shape[0] <= self._threshold

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if M <= self._threshold:
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        return self.fallback_experts.workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEExpertsModular:
        if self._should_use_marlin(hidden_states):
            return self.experts
        return self.fallback_experts

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        experts = self._select_experts_impl(hidden_states, w1, w2)
        if experts is self.experts:
            w1 = self.layer._marlin_w13_weight
            w2 = self.layer._marlin_w2_weight
        experts.apply(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        )
