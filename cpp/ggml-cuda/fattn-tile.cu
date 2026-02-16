#include "common.cuh"
#include "fattn-tile.cuh"
#include "fattn-wmma-f16.cuh"

void lm_ggml_cuda_flash_attn_ext_tile(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * K = dst->src[1];
    const lm_ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ctx, dst);
        } break;
        case  64: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ctx, dst);
        } break;
        case  72: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ctx, dst);
        } break;
        case  80: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ctx, dst);
        } break;
        case  96: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ctx, dst);
        } break;
        case 112: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case<112, 112>(ctx, dst);
        } break;
        case 128: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case<128, 128>(ctx, dst);
        } break;
        case 256: {
            LM_GGML_ASSERT(V->ne[0] == K->ne[0]);
            lm_ggml_cuda_flash_attn_ext_tile_case<256, 256>(ctx, dst);
        } break;
        case 576: {
            LM_GGML_ASSERT(V->ne[0] == 512);
            lm_ggml_cuda_flash_attn_ext_tile_case<576, 512>(ctx, dst);
        } break;
        default: {
            LM_GGML_ABORT("Unsupported head size");
        } break;
    }
}
