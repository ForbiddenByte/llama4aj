#include "set.cuh"
#include "cpy.cuh"

void lm_ggml_cuda_op_set(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT((src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_I32));
    LM_GGML_ASSERT(src1->type == src0->type);
    LM_GGML_ASSERT(dst ->type == src0->type);

    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));

    const size_t nb1    = ((int32_t *) dst->op_params)[0];
    const size_t nb2    = ((int32_t *) dst->op_params)[1];
    const size_t nb3    = ((int32_t *) dst->op_params)[2];
    const size_t offset = ((int32_t *) dst->op_params)[3];
    const bool   inplace= (bool)     ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        lm_ggml_cuda_cpy(ctx, src0, dst);
    }

    lm_ggml_tensor dst_view = *dst;
    dst_view.data  = (void *)((char *)dst->data + offset);
    dst_view.ne[0] = src1->ne[0];
    dst_view.ne[1] = src1->ne[1];
    dst_view.ne[2] = src1->ne[2];
    dst_view.ne[3] = src1->ne[3];

    dst_view.nb[0] = lm_ggml_element_size(dst);
    dst_view.nb[1] = nb1;
    dst_view.nb[2] = nb2;
    dst_view.nb[3] = nb3;

    lm_ggml_cuda_cpy(ctx, src1, &dst_view);
}
