#include "dequantize.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

template <typename T>
kernel void kernel_pad_impl(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t i03 = i3;
    const int32_t i02 = i2;
    const int32_t i01 = i1;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            dst_ptr[i0] = src0_ptr[i0];
        } else {
            dst_ptr[i0] = 0.0f;
        }
    }
}

typedef decltype(kernel_pad_impl<float>) kernel_pad_t;

template [[host_name("kernel_pad_f32")]]   kernel kernel_pad_t kernel_pad_impl<float>;
template [[host_name("kernel_pad_f32_4")]] kernel kernel_pad_t kernel_pad_impl<float4>;

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N>
kernel void kernel_fwht_f32(
        constant ggml_metal_kargs_fwht & args,
        device const float * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = src[i*NW + lane]*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

typedef decltype(kernel_fwht_f32<64>) kernel_fwht_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_t kernel_fwht_f32<64>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_t kernel_fwht_f32<128>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_t kernel_fwht_f32<256>;
template [[host_name("kernel_fwht_f32_512")]] kernel kernel_fwht_t kernel_fwht_f32<512>;

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + it*args.nb_w1);
    }

    float w[hc];
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        w[ih] = simd_shuffle(weight_lane, ih);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        result = fma(*(device const float *) (xb + ih*args.nb_x1), w[ih], result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = result;
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}


// ============================================================================
// MoE Expert Cache matvec — reads quantized weights from a pool slab
// at per-row offsets and computes dot products with q8_1 activations.
// One thread per (hit, output_row) pair.
// Template parameter: block_t = the quantized block type.
// ============================================================================

// Per-type block dot functions. MSL type-checks every branch of a runtime if,
// so dispatch happens via overload resolution (each overload only touches its
// own block type's members). One thread per (hit, output row) pair.
// Q8_0: signed 8-bit quants, one 32-wide act block
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q8_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_w = (float)w_block->d;
    const float d_a = (float)a_block->d;
    int sum_q = 0;
    for (int i = 0; i < QK8_0; i++) {
        sum_q += (int)w_block->qs[i] * (int)a_block->qs[i];
    }
    sum += d_w * d_a * (float)sum_q;
}

// Q4_0: 4-bit nibbles, offset by -8, one 32-wide act block.
// qs packing (CPU reference): byte j = element j (low nibble) | element
// j+16 (high nibble), so element j pairs with act element j.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_w = (float)w_block->d;
    const float d_a = (float)a_block->d;
    int sum_q = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t nibbles = w_block->qs[i];
        sum_q += ((int)(nibbles & 0xF) - 8) * (int)a_block->qs[i];
        sum_q += ((int)(nibbles >> 4)  - 8) * (int)a_block->qs[i + 16];
    }
    sum += d_w * d_a * (float)sum_q;
}

// Q4_K: 8 sub-blocks of 32 elements, each with a 6-bit scale and 6-bit min
// packed across scales[12] (get_scale_min_k4). Sub-block sb pairs with act
// block sb; formula matches the CUDA reference:
//   d_all * d8[sb] * sc6 * dot1 - d_min * d8[sb] * mn6 * dot2
// where dot1 = sum(q4*qs) and dot2 = sum(qs) over the 32 elements.
// qs packing (CPU reference): sub-block pair 2k/2k+1 shares qs[32k..32k+31],
// even sub-block in the low nibbles, odd in the high nibbles.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint8_t * q = w_block->qs;
    device const uint8_t * sc = w_block->scales;
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;

    for (int sb = 0; sb < 8; sb++) {
        int sc6, mn6;
        if (sb < 4) {
            sc6 = sc[sb] & 0x3F;
            mn6 = sc[sb + 4] & 0x3F;
        } else {
            sc6 = (sc[sb + 4] & 0xF) | ((sc[sb - 4] & 0xC0) >> 2);
            mn6 = (sc[sb + 4] >> 4) | ((sc[sb] & 0xC0) >> 2);
        }
        const float dl = d_all * (float)sc6;
        const float ml = d_min * (float)mn6;
        const float d_a = (float)a_block[sb].d;

        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 32; j++) {
            const uint8_t nibbles = q[(sb >> 1) * 32 + j];
            const int n0 = (sb & 1) ? (nibbles >> 4) : (nibbles & 0xF);
            const int qs = (int)a_block[sb].qs[j];
            dot1 += n0 * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q6_K: 16 sub-blocks of 16 elements, int8 scale per sub-block, 8 act blocks
// of 32. Element e uses weight scale scales[e/16] and act block e/32, so two
// weight sub-blocks share one act block. q6 value packing (CPU reference):
//   ql byte (e%64) + 64*(e/128), high nibble iff (e/64) is odd
//   qh byte (e%32) + 32*(e/128), bits 2*((e/32) % 4)
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q6_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;

    for (int e = 0; e < QK_K; e++) {
        const int sb = e / 16;
        const int ab = e / 32;
        const float dl = d_all * (float)w_block->scales[sb];
        const float d_a = (float)a_block[ab].d;

        const uint8_t low = w_block->ql[(e % 64) + 64 * (e / 128)];
        const uint8_t high = w_block->qh[(e % 32) + 32 * (e / 128)];
        const int q4 = ((e / 64) & 1) ? (low >> 4) : (low & 0xF);
        const int q2 = (high >> (2 * ((e / 32) % 4))) & 0x3;
        const int q6 = q4 | (q2 << 4);
        const float w_val = dl * (float)(q6 - 32);

        sum += w_val * d_a * (float)a_block[ab].qs[e % 32];
    }
}

// Q5_K: 8 sub-blocks of 32. Same scale/min packing as Q4_K (get_scale_min_k4)
// plus a per-element high bit in qh. q5 = nibble | (high << 4) with no offset
// (the min term shifts it; see vec_dot_q5_K_q8_1_impl_vmmq, which also uses
// the unshifted vl|vh). The high bit of element (sb*32 + j) sits at qh[j]
// bit (2*(sb >> 1) + (sb & 1)).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint8_t * q = w_block->qs;
    device const uint8_t * qh = w_block->qh;
    device const uint8_t * sc = w_block->scales;
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;

    for (int sb = 0; sb < 8; sb++) {
        int sc6, mn6;
        if (sb < 4) {
            sc6 = sc[sb] & 0x3F;
            mn6 = sc[sb + 4] & 0x3F;
        } else {
            sc6 = (sc[sb + 4] & 0xF) | ((sc[sb - 4] & 0xC0) >> 2);
            mn6 = (sc[sb + 4] >> 4) | ((sc[sb] & 0xC0) >> 2);
        }
        const float dl = d_all * (float)sc6;
        const float ml = d_min * (float)mn6;
        const float d_a = (float)a_block[sb].d;

        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 32; j++) {
            const uint8_t nibbles = q[(sb >> 1) * 32 + j];
            const int q5 = (sb & 1) ? (nibbles >> 4) : (nibbles & 0xF);
            const int high = (qh[j] >> (2 * (sb >> 1) + (sb & 1))) & 1;
            const int qs = (int)a_block[sb].qs[j];
            dot1 += (q5 | (high << 4)) * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q1_0: 128 elements per block, one scale, 1 bit per element (+1/-1). The
// weight block spans four 32-wide act chunks, each with its own q8_1 scale.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q1_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int c = 0; c < 4; c++) {
        const float d_a = (float)a_block[c].d;
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            const int e = c*32 + j;
            const int bit = (w_block->qs[e >> 3] >> (e & 7)) & 1;
            sumi += (bit ? 1 : -1) * (int)a_block[c].qs[j];
        }
        s += d_a * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// Q2_0: one scale and 2 bits per element, grouped in 32-element activation blocks.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q2_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int c = 0; c < QK2_0/32; c++) {
        const float d_a = (float)a_block[c].d;
        int sumi = 0;
        for (int j = 0; j < 32; j++) {
            const int e = c*32 + j;
            const int q2 = (w_block->qs[e >> 2] >> (2 * (e & 3))) & 3;
            sumi += (q2 - 1) * (int)a_block[c].qs[j];
        }
        s += d_a * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// Q4_1: nibbles 0..15, min term via the q8_1 activation sum (CPU reference:
// (d*d8)*sumi + m*s). qs packing like Q4_0.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q4_1 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float m = (float)w_block->m;
    const float d_a = (float)a_block->d;
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t nibbles = w_block->qs[i];
        sumi += (int)(nibbles & 0xF) * (int)a_block->qs[i];
        sumi += (int)(nibbles >> 4) * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi + m * (float)a_block->s;
}

// Q5_0: nibble | (high bit << 4) - 16, single scale, no min term. High bit of
// element e is bit e of the 32-bit qh field (4 bytes, CPU packing).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_0 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float d_a = (float)a_block->d;
    const uint32_t qh = (uint32_t)w_block->qh[0] | ((uint32_t)w_block->qh[1] << 8) |
                        ((uint32_t)w_block->qh[2] << 16) | ((uint32_t)w_block->qh[3] << 24);
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const int v0 = ((int)(w_block->qs[i] & 0xF) | (((int)(qh >> i) & 1) << 4)) - 16;
        const int v1 = ((int)(w_block->qs[i] >> 4) | (((int)(qh >> (i + 16)) & 1) << 4)) - 16;
        sumi += v0 * (int)a_block->qs[i] + v1 * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi;
}

// Q5_1: nibble | (high bit << 4) in 0..31, min term via the act sum (CPU
// reference: (d*d8)*sumi + m*s).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q5_1 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = (float)w_block->d;
    const float m = (float)w_block->m;
    const float d_a = (float)a_block->d;
    const uint32_t qh = (uint32_t)w_block->qh[0] | ((uint32_t)w_block->qh[1] << 8) |
                        ((uint32_t)w_block->qh[2] << 16) | ((uint32_t)w_block->qh[3] << 24);
    int sumi = 0;
    for (int i = 0; i < 16; i++) {
        const int v0 = (int)(w_block->qs[i] & 0xF) | (((int)(qh >> i) & 1) << 4);
        const int v1 = (int)(w_block->qs[i] >> 4) | (((int)(qh >> (i + 16)) & 1) << 4);
        sumi += v0 * (int)a_block->qs[i] + v1 * (int)a_block->qs[i + 16];
    }
    sum += (d * d_a) * (float)sumi + m * (float)a_block->s;
}

// Q2_K: 16 sub-blocks of 16 elements. scales[sb] packs a 4-bit scale (low) and
// a 4-bit min (high). q2 codes are raw 0..3; the min term recenters them.
// Sub-block sb pairs with act block (ab + sb/2). value packing (CPU
// vec_dot_q2_K_q8_K reference): element e uses byte qs[32*(e/128) + e%32],
// 2-bit field 2*((e/32)%4).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q2_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;
    const float d_min = (float)w_block->dmin;
    for (int sb = 0; sb < 16; sb++) {
        const float dl = d_all * (float)(w_block->scales[sb] & 0xF);
        const float ml = d_min * (float)(w_block->scales[sb] >> 4);
        const float d_a = (float)a_block[sb / 2].d;
        int dot1 = 0;
        int dot2 = 0;
        for (int j = 0; j < 16; j++) {
            const int e = sb*16 + j;
            const int q2 = (w_block->qs[32*(e/128) + e%32] >> (2*((e/32)%4))) & 3;
            const int qs = (int)a_block[sb / 2].qs[j + 16*(sb & 1)];
            dot1 += q2 * qs;
            dot2 += qs;
        }
        sum += d_a * (dl * (float)dot1 - ml * (float)dot2);
    }
}

// Q3_K: 16 sub-blocks of 16 elements, 6-bit scale per sub-block packed across
// scales[12] (CPU vec_dot_q3_K_q8_K reference unpacking). value = (2-bit
// code) - 4 + 4*hmask bit; scale = sc6 - 32; no min term.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_q3_K * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_all = (float)w_block->d;
    for (int sb = 0; sb < 16; sb++) {
        // 6-bit scales packed across scales[12] (CUDA vec_dot_q3_K_q8_1
        // unpacking): 16 low nibbles in scales[0..7], 16x2 high bits in
        // scales[8..11].
        const int sc_low  = (w_block->scales[sb % 8] >> (4 * (sb / 8))) & 0xF;
        const int sc_high = (w_block->scales[8 + sb % 4] >> (2 * (sb / 4))) & 0x3;
        const float dl = d_all * (float)((sc_low | (sc_high << 4)) - 32);
        const float d_a = (float)a_block[sb / 2].d;
        int dot1 = 0;
        for (int j = 0; j < 16; j++) {
            const int e = sb*16 + j;
            const int low2 = (w_block->qs[32*(e/128) + e%32] >> (2*((e/32)%4))) & 3;
            const int hbit = (w_block->hmask[e%32] >> (4*(e/128) + (e/32)%4)) & 1;
            dot1 += (low2 - 4 + 4*hbit) * (int)a_block[sb / 2].qs[j + 16*(sb & 1)];
        }
        sum += d_a * dl * (float)dot1;
    }
}

// IQ2_XXS: 8 act windows of 32. Each window uses 8 bytes of qs: 4 grid
// indices (aux32_g) and 4 bytes of scale+signs (aux32_s); the 4-bit scale
// sits in the top nibble, signs are 7-bit groups (ksigns_iq2xs). scale factor
// (2*scale+1)/8 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_xxs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        device const uint16_t * q2 = w_block->qs + 4*ib32;
        const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        const int ls = 2*((int)((aux32_s >> 28) & 0xF)) + 1;
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)((aux32_g >> (8*l)) & 0xFF);
            const uint8_t signs = ksigns_iq2xs[(aux32_s >> (7*l)) & 127];
            constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + idx);
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (signs >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ2_XS: per 32-window, 4 uint16 words hold a 9-bit grid index and 7 sign
// bits each; scales[ib32] holds two 4-bit scales (per 16-element half).
// scale factor (2*scale+1)/8 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_xs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls1 = 2*((int)(w_block->scales[ib32] & 0xF)) + 1;
        const int ls2 = 2*((int)(w_block->scales[ib32] >> 4)) + 1;
        device const uint16_t * q2 = w_block->qs + 4*ib32;
        const float d_a = (float)a_block[ib32].d;
        int sumi1 = 0;
        int sumi2 = 0;
        for (int l = 0; l < 4; l++) {
            const uint16_t q2w = q2[l];
            const int idx = (int)(q2w & 511);
            const uint8_t signs = ksigns_iq2xs[q2w >> 9];
            constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + idx);
            int sumi = 0;
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (signs >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
            if (l < 2) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        s += d_a * (float)(ls1*sumi1 + ls2*sumi2);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ2_S: per 32-window, 4 grid bytes + 4 sign bytes (at qs + 32), a 2-bit high
// index from qh[ib32], and two 4-bit scales. scale factor (2*scale+1)/8.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq2_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls1 = 2*((int)(w_block->scales[ib32] & 0xF)) + 1;
        const int ls2 = 2*((int)(w_block->scales[ib32] >> 4)) + 1;
        const uint8_t qh = w_block->qh[ib32];
        const float d_a = (float)a_block[ib32].d;
        int sumi1 = 0;
        int sumi2 = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)w_block->qs[4*ib32 + l] | ((qh << (8 - 2*l)) & 0x300);
            const uint8_t sg = w_block->qs[32 + 4*ib32 + l];
            constant uint8_t * grid = (constant uint8_t *)(iq2s_grid + idx);
            int sumi = 0;
            for (int i = 0; i < 8; i++) {
                const int g = (int)grid[i];
                const int sgn = (sg >> i) & 1;
                sumi += (sgn ? -g : g) * (int)a_block[ib32].qs[8*l + i];
            }
            if (l < 2) { sumi1 += sumi; } else { sumi2 += sumi; }
        }
        s += d_a * (float)(ls1*sumi1 + ls2*sumi2);
    }
    sum += (float)w_block->d * (1.0f/8.0f) * s;
}

// IQ3_XXS: per 32-window, 8 grid bytes in qs[0..63] and 4 scale+sign bytes in
// qs[64..95]. Two 4-byte grid entries and 7-bit sign groups per l; scale
// factor (2*scale+1)/4 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq3_xxs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const uint32_t aux32 = (uint32_t)w_block->qs[64 + 4*ib32 + 0] |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 1] << 8) |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 2] << 16) |
                               ((uint32_t)w_block->qs[64 + 4*ib32 + 3] << 24);
        const int ls = 2*((int)((aux32 >> 28) & 0xF)) + 1;
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + w_block->qs[8*ib32 + 2*l + 0]);
            constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + w_block->qs[8*ib32 + 2*l + 1]);
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            for (int i = 0; i < 4; i++) {
                const int g1 = (int)grid1[i];
                const int g2 = (int)grid2[i];
                const int s1 = (signs >> i) & 1;
                const int s2 = (signs >> (i + 4)) & 1;
                sumi += (s1 ? -g1 : g1) * (int)a_block[ib32].qs[8*l + i];
                sumi += (s2 ? -g2 : g2) * (int)a_block[ib32].qs[8*l + i + 4];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * (1.0f/4.0f) * s;
}

// IQ3_S: per 32-window, 8 grid bytes, 4 sign bytes, qh[ib32] for the 9th
// index bit, and a 4-bit scale. scale factor 1 + 2*scale (no /8; CPU
// reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq3_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const int ls = 2*((int)(w_block->scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xF) + 1;
        const uint8_t qh = w_block->qh[ib32];
        const float d_a = (float)a_block[ib32].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (w_block->qs[8*ib32 + 2*l + 0] | ((qh << (8 - 2*l)) & 256)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (w_block->qs[8*ib32 + 2*l + 1] | ((qh << (7 - 2*l)) & 256)));
            const uint8_t sg = w_block->signs[4*ib32 + l];
            for (int i = 0; i < 4; i++) {
                const int g1 = (int)grid1[i];
                const int g2 = (int)grid2[i];
                const int s1 = (sg >> i) & 1;
                const int s2 = (sg >> (i + 4)) & 1;
                sumi += (s1 ? -g1 : g1) * (int)a_block[ib32].qs[8*l + i];
                sumi += (s2 ? -g2 : g2) * (int)a_block[ib32].qs[8*l + i + 4];
            }
        }
        s += d_a * (float)(sumi * ls);
    }
    sum += (float)w_block->d * s;
}

// IQ1_S: per 32-window, 4 grid bytes in qs, qh[ib] supplies a 3-bit high index
// and the scale/delta. grid values are 4-bit per byte (iq1s_grid_gpu). The
// min term uses the q8_1 activation sum (CUDA reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq1_s * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const uint16_t qh = w_block->qh[ib];
        const int ls = 2*((qh >> 12) & 7) + 1;
        const float delta = (qh & 0x8000) ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
        const float d_a = (float)a_block[ib].d;
        int sumi = 0;
        for (int l = 0; l < 4; l++) {
            const int idx = (int)w_block->qs[4*ib + l] | (((qh >> (3*l)) & 7) << 8);
            constant uint32_t * e = iq1s_grid_gpu + idx;
            const uint32_t ev = *e;
            for (int i = 0; i < 4; i++) {
                const int g0 = (int)((ev >> (8*i)) & 0xF);
                const int g1 = (int)((ev >> (8*i + 4)) & 0xF);
                sumi += g0 * (int)a_block[ib].qs[8*l + i];
                sumi += g1 * (int)a_block[ib].qs[8*l + i + 4];
            }
        }
        s += (float)ls * (d_a * (float)sumi + (float)a_block[ib].s * delta);
    }
    sum += (float)w_block->d * s;
}

// IQ1_M: no block d; the scale is reconstructed from the four uint16 scales
// words. Per 32-window: 4 grid bytes, 2 qh bytes (deltas), two 6-bit scales.
// delta terms use the raw q8 sum (CUDA reference), act scale from ds.x.
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq1_m * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    device const uint16_t * sc = (device const uint16_t *)w_block->scales;
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const uint8_t qh0 = w_block->qh[2*ib];
        const uint8_t qh1 = w_block->qh[2*ib + 1];
        const float delta0 = (qh0 & 0x08) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta1 = (qh0 & 0x80) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta2 = (qh1 & 0x08) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const float delta3 = (qh1 & 0x80) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        const uint16_t sc16 = sc[ib >> 1];
        const int ls1 = 2*((sc16 >> (6*(ib & 1) + 0)) & 7) + 1;
        const int ls2 = 2*((sc16 >> (6*(ib & 1) + 3)) & 7) + 1;
        const float d_a = (float)a_block[ib].d;
        int sumi0 = 0;
        int sumi1 = 0;
        float sumf0 = 0.0f;
        float sumf1 = 0.0f;
        for (int l = 0; l < 4; l++) {
            const uint8_t qhl = (l < 2) ? qh0 : qh1;
            const int idx = (int)w_block->qs[4*ib + l] | ((qhl << (8 - 4*(l % 2))) & 0x700);
            const float delta = l == 0 ? delta0 : l == 1 ? delta1 : l == 2 ? delta2 : delta3;
            constant uint32_t * e = iq1s_grid_gpu + idx;
            const uint32_t ev = *e;
            int sumi = 0;
            int sumy = 0;
            for (int i = 0; i < 4; i++) {
                const int g0 = (int)((ev >> (8*i)) & 0xF);
                const int g1 = (int)((ev >> (8*i + 4)) & 0xF);
                sumi += g0 * (int)a_block[ib].qs[8*l + i] + g1 * (int)a_block[ib].qs[8*l + i + 4];
                sumy += (int)a_block[ib].qs[8*l + i] + (int)a_block[ib].qs[8*l + i + 4];
            }
            if (l < 2) { sumi0 += sumi; sumf0 += delta * (float)sumy; }
            else       { sumi1 += sumi; sumf1 += delta * (float)sumy; }
        }
        s += d_a * (((float)sumi0 + sumf0) * (float)ls1 + ((float)sumi1 + sumf1) * (float)ls2);
    }
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    sum += (float)scale.f16 * s;
}

// IQ4_NL: 32 elements per block, one scale, 4-bit non-linear codes
// (kvalues_iq4nl), 2 per byte (CPU vec_dot_iq4_nl_q8_1 reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq4_nl * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d_a = (float)a_block[0].d;
    int sumi = 0;
    for (int j = 0; j < 16; j++) {
        const uint8_t byte = w_block->qs[j];
        sumi += (int)kvalues_iq4nl_f[byte & 0xF] * (int)a_block[0].qs[j];
        sumi += (int)kvalues_iq4nl_f[byte >> 4] * (int)a_block[0].qs[j + 16];
    }
    sum += (float)w_block->d * d_a * (float)sumi;
}

// IQ4_XS: per 32-window, 16 qs bytes and a 6-bit scale (4 bits from scales_l,
// 2 from scales_h), offset by -32 (CPU reference).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_iq4_xs * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    float s = 0.0f;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((w_block->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) |
                       (((w_block->scales_h >> (2*ib)) & 3) << 4);
        const float dl = (float)(ls - 32);
        const float d_a = (float)a_block[ib].d;
        int sumi = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t byte = w_block->qs[16*ib + j];
            sumi += (int)kvalues_iq4nl_f[byte & 0xF] * (int)a_block[ib].qs[j];
            sumi += (int)kvalues_iq4nl_f[byte >> 4] * (int)a_block[ib].qs[j + 16];
        }
        s += d_a * dl * (float)sumi;
    }
    sum += (float)w_block->d * s;
}

// MXFP4: 32 elements, one E8M0 exponent, 2 E2M1 values per byte. The Metal
// kvalues_mxfp4_f table already holds the half-step values (CPU/CUDA
// reference: e8m0 * 0.5 * 2*E2M1 == e8m0 * E2M1).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_mxfp4 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    const float d = e8m0_to_fp32(w_block->e) * (float)a_block->d;
    int sumi = 0;
    for (int j = 0; j < 16; j++) {
        const uint8_t byte = w_block->qs[j];
        sumi += (int)kvalues_mxfp4_f[byte & 0xF] * (int)a_block->qs[j];
        sumi += (int)kvalues_mxfp4_f[byte >> 4] * (int)a_block->qs[j + 16];
    }
    sum += d * (float)sumi;
}

// NVFP4: 64 elements = 4 sub-blocks of 16, each with a UE4M3 scale; 2 E2M1
// values per byte (CPU vec_dot_nvfp4 reference packing).
static inline void kernel_moe_cache_mv_block_dot(
    device const block_nvfp4 * w_block,
    device const block_q8_1 * a_block,
    thread float & sum) {
    for (int s = 0; s < 4; s++) {
        const float d = nvfp4_ue4m3_to_fp32(w_block->d[s]) * (float)a_block[s >> 1].d;
        const int off = (s & 1) * 16;
        int sumi = 0;
        for (int j = 0; j < 8; j++) {
            const uint8_t byte = w_block->qs[8*s + j];
            sumi += (int)kvalues_mxfp4_f[byte & 0xF] * (int)a_block[s >> 1].qs[off + j];
            sumi += (int)kvalues_mxfp4_f[byte >> 4] * (int)a_block[s >> 1].qs[off + j + 8];
        }
        sum += d * (float)sumi;
    }
}

// Host binds all six scalars as one packed blob at buffer index 4
// (ggml_metal_encoder_set_bytes(enc, args, sizeof(args), 4)), so the kernel must
// take a single struct there. Six separate `constant int64_t &` parameters would
// claim indices 4..9, leaving every field after n_in unbound.
struct moe_cache_mv_args {
    int64_t n_in;
    int64_t n_out;
    int64_t expert_stride;
    int64_t row_stride;
    int64_t n_hits;
    int64_t padded_n_in;
};

template <typename block_t, short qk>
kernel void kernel_moe_cache_mv_generic(
    device const char * slab,
    device const int32_t * ids,
    device const block_q8_1 * act_q8,
    device float * dst,
    constant moe_cache_mv_args & args,
    uint id [[thread_position_in_grid]]
) {
    const int64_t n_in          = args.n_in;
    const int64_t n_out         = args.n_out;
    const int64_t expert_stride = args.expert_stride;
    const int64_t row_stride    = args.row_stride;
    const int64_t n_hits        = args.n_hits;
    const int64_t padded_n_in   = args.padded_n_in;
    const int64_t hit = (int64_t)id / n_out;
    const int64_t row = (int64_t)id - hit * n_out;

    if (hit >= n_hits) return;

    const int slot = ids[hit];
    if (slot < 0) {
        dst[hit * n_out + row] = 0.0f;
        return;
    }

    device const block_t * w_row = (device const block_t *)(slab + (int64_t)slot * expert_stride + row * row_stride);
    device const block_q8_1 * act = act_q8 + hit * (padded_n_in / QK8_1);

    const int nb = (int)(n_in / qk);
    float sum = 0.0f;
    for (int ib = 0; ib < nb; ib++) {
        // a 256-column K-type weight block consumes qk / QK8_1 act blocks
        kernel_moe_cache_mv_block_dot(&w_row[ib], &act[ib * (qk / QK8_1)], sum);
    }
    dst[hit * n_out + row] = sum;
}

// Per-type instantiations with host_name for runtime dispatch

template [[host_name("kernel_moe_cache_mv_q8_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q8_0, QK8_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_0, QK4_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q6_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q6_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q1_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q1_0, QK1_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q2_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q2_0, QK2_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q4_1_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q4_1, QK4_1>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_0_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_0, QK5_0>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q5_1_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q5_1, QK5_1>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q2_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q2_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_q3_K_f32")]]
kernel void kernel_moe_cache_mv_generic<block_q3_K, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_xxs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_xxs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_xs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_xs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq2_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq2_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq3_xxs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq3_xxs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq3_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq3_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq1_s_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq1_s, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq1_m_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq1_m, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq4_nl_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq4_nl, QK4_NL>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_iq4_xs_f32")]]
kernel void kernel_moe_cache_mv_generic<block_iq4_xs, QK_K>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_mxfp4_f32")]]
kernel void kernel_moe_cache_mv_generic<block_mxfp4, QK_MXFP4>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

template [[host_name("kernel_moe_cache_mv_nvfp4_f32")]]
kernel void kernel_moe_cache_mv_generic<block_nvfp4, QK_NVFP4>(
    device const char *, device const int32_t *, device const block_q8_1 *,
    device float *, constant moe_cache_mv_args &, uint);

// Shared pre-rotation kernels (used by both TQ3 and TQ4 for mul_mm path)
kernel void kernel_tq3_rotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg] * tq3_signs[tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32;
}

kernel void kernel_tq3_unrotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32 * tq3_signs[tiisg];
}


// ===== TurboQuant4 bulk dequant to fp16 (for prefill FA) =====
// Dequants turbo4 blocks → half buffer. Dispatch before f16 FA during prefill.
// Each thread processes one 128-element block.
kernel void kernel_turbo4_dequant_f16(
        device const block_turbo4_0 * src [[buffer(0)]],
        device       half           * dst [[buffer(1)]],
        constant     uint           & n_blocks [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint blk_idx = tgpig * ntg + tiitg;
    if (blk_idx >= n_blocks) return;

    device const block_turbo4_0 & blk = src[blk_idx];
    device half * out = dst + blk_idx * QK_TURBO4;
    const half norm_h = blk.norm;

    // 4-bit nibble unpack → centroid → scale by norm → write fp16
    for (int j = 0; j < QK_TURBO4; j += 2) {
        const uint8_t qb = blk.qs[j / 2];
        out[j    ] = turbo_centroids_4bit_h[(qb     ) & 0xF] * norm_h;
        out[j + 1] = turbo_centroids_4bit_h[(qb >> 4) & 0xF] * norm_h;
    }
}

// ===== TurboQuant Walsh-Hadamard Transform kernel =====
// O(d log d) rotation for 128-element groups. Replaces dense 128x128 matmul.
// Each thread processes one 128-element group using half4 vectorized butterfly.
// Uses the same WHT signs already defined (turbo_wht_signs1/2, turbo_wht_signs1_h4/2_h4).

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // Each thread handles one 128-element group
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    // Load into half4 vectors for fast butterfly
    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    // Apply first signs (s1 for fwd, s2 for inv)
    for (int i = 0; i < 32; i++) {
        float4 f = float4(in[i*4], in[i*4+1], in[i*4+2], in[i*4+3]);
        half4 s = is_inverse ? turbo_wht_signs2_h4[i] : turbo_wht_signs1_h4[i];
        v[i] = half4(f) * s;
    }

    // WHT butterfly (7 stages, vectorized half4)
    // h=1: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // h=2: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // h=4..64: between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i], b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }

    // Apply second signs + normalize, write output as fp32
    const half4 inv_sqrt = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        half4 s = is_inverse ? turbo_wht_signs1_h4[i] : turbo_wht_signs2_h4[i];
        float4 f = float4(v[i] * inv_sqrt * s);
        out[i*4]   = f.x;
        out[i*4+1] = f.y;
        out[i*4+2] = f.z;
        out[i*4+3] = f.w;
    }
}

constant float convrot_h4[4][4] = {
    { 1.0f,  1.0f,  1.0f, -1.0f},
    { 1.0f,  1.0f, -1.0f,  1.0f},
    { 1.0f, -1.0f,  1.0f,  1.0f},
    {-1.0f,  1.0f,  1.0f,  1.0f},
};

kernel void kernel_convrot(
        device const float * src [[buffer(0)]],
        device       float * dst [[buffer(1)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[2][QK8_CR];

    for (ushort i = tid; i < QK8_CR; i += 64) {
        buf[0][i] = src[group * QK8_CR + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort rd = 0;
    ushort wr = 1;
    for (ushort len = 4, shift = 0; len <= QK8_CR; len *= 4, shift += 2) {
        const ushort quarter = len / 4;
        for (ushort i = tid; i < QK8_CR; i += 64) {
            const ushort base = i & ~(len - 1);
            const ushort row = (i >> shift) & 3;
            const ushort col = i & (quarter - 1);

            float sum = 0.0f;
            for (ushort j = 0; j < 4; ++j) {
                sum += convrot_h4[row][j] * buf[rd][base + j * quarter + col];
            }
            buf[wr][i] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ushort tmp = rd;
        rd = wr;
        wr = tmp;
    }

    for (ushort i = tid; i < QK8_CR; i += 64) {
        dst[group * QK8_CR + i] = buf[rd][i] * (1.0f / 16.0f);
    }
}
