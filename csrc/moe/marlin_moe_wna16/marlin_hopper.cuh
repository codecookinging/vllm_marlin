// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// H100 (SM90/Hopper) specific PTX helpers for marlin_template.h.
//
// All helpers are guarded by __CUDA_ARCH__ >= 900 so they compile safely on
// any target but only emit SM90 PTX when the device code path requires it.

#pragma once
#include <cuda.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

namespace marlin_hopper {

// ---------------------------------------------------------------------------
// cp.async variants with L2-friendly cache policy
// ---------------------------------------------------------------------------
//
// NOTE: cp.async supports .L2::64B/.L2::128B/.L2::256B prefetch-size hints,
// which are useful for contiguous reusable data such as weights and scales.
// Eviction-priority hints such as .L2::evict_first are not legal on cp.async
// and ptxas rejects them with:
//   Illegal modifier '.level::eviction_priority' for instruction 'cp.async'
//
// For scattered activation gathers we use plain .cg.  This bypasses L1 and
// avoids polluting it with one-shot MoE token reads, while leaving L2 eviction
// to hardware policy.

// For weight / scale tiles: request a 128-byte L2 prefetch around each 16-byte
// async copy.  The copy size remains 16 bytes; .L2::128B is only a prefetch
// hint for nearby data that subsequent lanes / iterations are likely to use.
__device__ __forceinline__ void cp_async4_l2_128(void* smem_ptr,
                                                 const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
               :
               : "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// For activation tiles: use .cg so scattered reads don't thrash L1.

// Predicated variant matches the original cp_async4_pred() signature.
__device__ __forceinline__ void cp_async4_pred_cg(void* smem_ptr,
                                                  const void* glob_ptr,
                                                  bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %0, 0;\n"
      "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n"
      :
      : "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

}  // namespace marlin_hopper

#endif  // __CUDA_ARCH__ >= 900
