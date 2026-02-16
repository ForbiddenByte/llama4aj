#include "ggml-impl.h"
#include "opt-step-sgd.cuh"

#include <cstdint>

static __global__ void opt_step_sgd_f32(
    float * __restrict__ x, const float * __restrict__ g,
    const float * __restrict__ pars, const int64_t k) {

    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

static void opt_step_sgd_f32_cuda(
    float * x, const float * g, const float * __restrict__ pars, const int64_t k, cudaStream_t stream) {

    const dim3 block_dims(CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((k + CUDA_OPT_STEP_SGD_BLOCK_SIZE - 1) / CUDA_OPT_STEP_SGD_BLOCK_SIZE, 1, 1);
    opt_step_sgd_f32<<<block_nums, block_dims, 0, stream>>>(x, g, pars, k);
}

void lm_ggml_cuda_opt_step_sgd(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0      = dst->src[0];
    const lm_ggml_tensor * src0_grad = dst->src[1];
    const lm_ggml_tensor * params    = dst->src[2];

    LM_GGML_ASSERT(src0->type      == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(src0_grad->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(params->type    == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0_grad));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(params));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src0_grad));
    LM_GGML_ASSERT(lm_ggml_nelements(params) == 2);

    float       * src0_d      = (float       *) src0->data;
    const float * src0_grad_d = (const float *) src0_grad->data;
    const float * params_d    = (const float *) params->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne = lm_ggml_nelements(src0);

    opt_step_sgd_f32_cuda(src0_d, src0_grad_d, params_d, ne, stream);
}
