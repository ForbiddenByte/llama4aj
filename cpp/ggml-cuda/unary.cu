#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return lm_ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return lm_ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return round(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op((float)x[i]);
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    unary_op_kernel<op><<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <float (*op)(float)>
void lm_ggml_cuda_op_unary(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src0->type == dst->type);

    if (src0->type == LM_GGML_TYPE_F16) {
        unary_cuda<op>((const half *)src0_d, (half *)dst_d, lm_ggml_nelements(src0), stream);
    } else {
        unary_cuda<op>((const float *)src0_d, (float *)dst_d, lm_ggml_nelements(src0), stream);
    }
}

void lm_ggml_cuda_op_abs(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void lm_ggml_cuda_op_sgn(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void lm_ggml_cuda_op_neg(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void lm_ggml_cuda_op_step(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_step>(ctx, dst);
}

void lm_ggml_cuda_op_gelu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void lm_ggml_cuda_op_gelu_erf(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void lm_ggml_cuda_op_gelu_quick(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void lm_ggml_cuda_op_silu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void lm_ggml_cuda_op_tanh(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void lm_ggml_cuda_op_relu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void lm_ggml_cuda_op_sigmoid(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void lm_ggml_cuda_op_hardsigmoid(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void lm_ggml_cuda_op_hardswish(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void lm_ggml_cuda_op_exp(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void lm_ggml_cuda_op_sqr(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void lm_ggml_cuda_op_sqrt(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void lm_ggml_cuda_op_sin(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void lm_ggml_cuda_op_cos(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void lm_ggml_cuda_op_log(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_log>(ctx, dst);
}

void lm_ggml_cuda_op_elu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void lm_ggml_cuda_op_floor(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void lm_ggml_cuda_op_ceil(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void lm_ggml_cuda_op_round(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_round>(ctx, dst);
}

void lm_ggml_cuda_op_trunc(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void lm_ggml_cuda_op_expm1(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void lm_ggml_cuda_op_softplus(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    unary_gated_op_kernel<op><<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1);
}

template <float (*op)(float)>
void lm_ggml_cuda_op_unary_gated(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const lm_ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src0));
    LM_GGML_ASSERT(src0->nb[0] == lm_ggml_element_size(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src0->type == dst->type);
    LM_GGML_ASSERT(dst->ne[0] == nc);
    LM_GGML_ASSERT(lm_ggml_nrows(dst) == lm_ggml_nrows(src0));

    if (src1) {
        LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src1));
        LM_GGML_ASSERT(src1->nb[0] == lm_ggml_element_size(src1));
        LM_GGML_ASSERT(src1->ne[0] == nc);
        LM_GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == LM_GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, lm_ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, lm_ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
    }
}

void lm_ggml_cuda_op_reglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void lm_ggml_cuda_op_geglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void lm_ggml_cuda_op_swiglu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void lm_ggml_cuda_op_geglu_erf(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void lm_ggml_cuda_op_geglu_quick(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    lm_ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = lm_ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void lm_ggml_cuda_op_swiglu_oai(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const lm_ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src0));
    LM_GGML_ASSERT(src0->nb[0] == lm_ggml_element_size(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(src0->type == dst->type);
    LM_GGML_ASSERT(dst->ne[0] == nc);
    LM_GGML_ASSERT(lm_ggml_nrows(dst) == lm_ggml_nrows(src0));

    if (src1) {
        LM_GGML_ASSERT(lm_ggml_is_contiguous_1(src1));
        LM_GGML_ASSERT(src1->nb[0] == lm_ggml_element_size(src1));
        LM_GGML_ASSERT(src1->ne[0] == nc);
        LM_GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = lm_ggml_get_op_params_i32(dst, 1);
    const float alpha = lm_ggml_get_op_params_f32(dst, 2);
    const float limit = lm_ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, lm_ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = lm_ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = lm_ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void lm_ggml_cuda_op_xielu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = lm_ggml_get_op_params_f32(dst, 1);
    const float alpha_p = lm_ggml_get_op_params_f32(dst, 2);
    const float beta    = lm_ggml_get_op_params_f32(dst, 3);
    const float eps     = lm_ggml_get_op_params_f32(dst, 4);

    if (src0->type == LM_GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, lm_ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, lm_ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void lm_ggml_cuda_op_silu_back(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const lm_ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src0->type == dst->type);

    if (src0->type == LM_GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, lm_ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, lm_ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void lm_ggml_cuda_op_leaky_relu(lm_ggml_backend_cuda_context & ctx, lm_ggml_tensor * dst) {
    const lm_ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32 || src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32 ||  dst->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == LM_GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, lm_ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, lm_ggml_nelements(src0), negative_slope, stream);
    }
}
