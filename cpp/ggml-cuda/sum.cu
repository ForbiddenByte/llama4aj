#include "sum.cuh"
#include "sumrows.cuh"

#ifdef LM_GGML_CUDA_USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif  // LM_GGML_CUDA_USE_CUB

#include <cstdint>

void sum_f32_cuda(lm_ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream) {
#ifdef LM_GGML_CUDA_USE_CUB
    size_t tmp_size = 0;
    DeviceReduce::Sum(nullptr,       tmp_size, x, dst, ne, stream);
    lm_ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
    DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
#else
    // Use (inefficient) sum_rows implementation as a fallback.
    // For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 -> C++14.
    sum_rows_f32_cuda(x, dst, ne, 1, stream);
    LM_GGML_UNUSED(pool);
#endif // LM_GGML_CUDA_USE_CUB
}

void lm_ggml_cuda_op_sum(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(lm_ggml_is_contiguously_allocated(src0));

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne = lm_ggml_nelements(src0);

    lm_ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
}
