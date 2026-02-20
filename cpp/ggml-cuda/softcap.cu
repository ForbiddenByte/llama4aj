#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = tanhf(scale * x[i]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_f32<<<num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream>>>(x, dst, scale, softcap, k);
}

// fused LM_GGML_OP_SCALE + LM_GGML_UNARY_OP_TANH + LM_GGML_OP_SCALE
void lm_ggml_cuda_op_softcap(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst, lm_ggml_tensor * src) {
    const lm_ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, lm_ggml_nelements(src0), stream);
}
