#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];
constant bool  FC_gated_delta_net_rows [[function_constant(FC_GATED_DELTA_NET + 3)]];
constant bool  FC_gated_delta_net_write_rows [[function_constant(FC_GATED_DELTA_NET + 4)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device const char * rows,
        device const char * write_rows,
        device       char * state_dst,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K
#define HAS_ROWS FC_gated_delta_net_rows
#define WRITE_ROWS FC_gated_delta_net_write_rows

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // The ordinary input contains the current state per sequence. In rows mode,
    // rows[i23] selects that state from the recurrent cache.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_seq_base = HAS_ROWS
        ? ((uint)((device const int *) rows)[i23])*(uint)(args.ne21*S_v*S_v)
        : (i23*args.ne21)*S_v*S_v;
    const uint state_in_base = state_seq_base + i21*S_v*S_v + i20*S_v;
    device const float * s_ptr = (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // emit_mode==0: full state snapshots (existing behavior). emit_mode==1: per-token replay
    // ingredients (k,v,g,beta), a fixed-cost trailing final-state block, and (when n_tokens>K)
    // a checkpoint block holding the state immediately before the K-token retained window
    // starts -- see ggml_gated_delta_net's emit_mode contract (ggml.h) and the CPU reference
    // (ggml-cpu/ops.cpp) for the exact layout this mirrors.
    //
    // NOTE: this emit_mode==1 path has not been built or run on real Metal hardware (ported by
    // hand from the CPU/CUDA implementations, which are). ggml_metal_supports_op currently
    // refuses emit_mode==1 unconditionally until it's been verified on real hardware.
    const bool emit_ingr = (args.emit_mode == 1);
    const short snap_rows_per_head = emit_ingr ? (short)4 : (short)S_v;

    // snapshot slot mapping (emit_mode==0): slot 0 = most recent state, slot s = s tokens back.
    // ingredient slot mapping (emit_mode==1): slot 0 = oldest of the K retained tokens
    // (chronological order -- see the CPU reference's comment on why this differs from emit_mode==0).
    // When n_tokens < K, only a subset of slots are written; the rest are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v*S_v*H*n_seqs (emit_mode==0) or 4*S_v*H*n_seqs (emit_mode==1)
    const uint state_size_per_snap = (uint)snap_rows_per_head * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot -- full S_v*S_v state layout (emit_mode==0, and also
    // shared by the final/ckpt blocks below, which are always full-state-shaped)
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    // per-(seq,head) offset within a slot -- ingredient layout (emit_mode==1 only; no per-row
    // term, since k/v/g/beta are packed as 4 whole S_v-wide rows per head, not per-row)
    const uint ingr_head_off = (i23*args.ne21 + i21)*(4u*S_v);

    // emit_mode==1 only: fixed-cost trailing final-state block, and (when n_tokens>K) a
    // checkpoint block -- both free to capture since the recurrence already passes through
    // them, avoiding a second op call over the prefix on the caller's side.
    const bool  needs_ckpt      = emit_ingr && ((int)args.ne22 > (int)K);
    const int   t_ckpt          = (int)args.ne22 - (int)K - 1;
    const uint  final_state_base = attn_size + (uint)K * state_size_per_snap;
    const uint  ckpt_state_base  = final_state_base + (uint)S_v*S_v*args.ne21*args.ne23;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = !emit_ingr && args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        // must run before the pointer-advance below: ingredients need THIS token's k/g/beta,
        // and the full-snapshot write only needs ls[] so ordering doesn't matter for it.
        if (K > 1 || emit_ingr) {
            const int target_slot = emit_ingr
                ? ((int)t - ((int)args.ne22 - (int)K))
                : ((int)args.ne22 - 1 - (int)t);
            if (target_slot >= 0 && target_slot < (int)K) {
                if (!emit_ingr) {
                    device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                    FOR_UNROLL (short j = 0; j < NSG; j++) {
                        const short is = tx*NSG + j;
                        dst_state[is] = ls[j];
                    }
                    if (WRITE_ROWS) {
                        // additionally scatter into the state cache in place of
                        // the folded SET_ROWS. SET_ROWS receives only the trailing
                        // n_write snapshots when T < K; convert the absolute
                        // output slot back to the compact row-index input's
                        // slot-major coordinate.
                        const int write_slot = target_slot - max(0, (int)K - (int)args.ne22);
                        const uint64_t row = ((device const int64_t *) write_rows)[(uint)write_slot * args.ne23 + i23];
                        device float * dst_rows = (device float *) state_dst + row * (uint64_t)(S_v * S_v * args.ne21)
                            + (uint) i21 * S_v * S_v + i20 * S_v;
                        FOR_UNROLL (short j = 0; j < NSG; j++) {
                            const short is = tx*NSG + j;
                            dst_rows[is] = ls[j];
                        }
                    }
                } else {
                    // ingredients: k, v, g, beta -- each padded/broadcast to width S_v, packed
                    // as 4 consecutive rows in that order. k/g/beta don't vary with i20 (the row
                    // this threadgroup owns), so only the i20==0 threadgroup writes them -- it
                    // still covers the full S_v range cooperatively via its own tx/NSG lanes,
                    // same as it already does to read k_ptr/g_ptr for the math above. v DOES
                    // vary with i20 (v_ptr[i20] is this threadgroup's own single element of the
                    // v vector), so every threadgroup writes its own element, via lane tx==0.
                    device float * ingr_base = (device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + ingr_head_off;
                    if (tx == 0) {
                        ingr_base[S_v + i20] = v_ptr[i20];
                    }
                    if (i20 == 0) {
                        FOR_UNROLL (short j = 0; j < NSG; j++) {
                            const short is = tx*NSG + j;
                            ingr_base[is]           = k_ptr[is];
                            ingr_base[2*S_v + is]   = (G == 1) ? g_ptr[0] : g_ptr[is]; // raw, pre-exp
                            ingr_base[3*S_v + is]   = b_ptr[0];
                        }
                    }
                }
            }
        }

        if (needs_ckpt && (int)t == t_ckpt) {
            device float * dst_ckpt = (device float *) (dst) + ckpt_state_base + state_out_base;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                dst_ckpt[is] = ls[j];
            }
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;
    }

    if (K == 1 && !emit_ingr) {
        device float * dst_state = (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }

        if (WRITE_ROWS) {
            // single snapshot slot: scatter it to the cache row in place of
            // the folded SET_ROWS, same as the K > 1 branch above
            const uint64_t row = ((device const int64_t *) write_rows)[i23];
            device float * dst_rows = (device float *) state_dst + row * (uint64_t)(S_v * S_v * args.ne21)
                + (uint) i21 * S_v * S_v + i20 * S_v;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                dst_rows[is] = ls[j];
            }
        }
    }

    // emit_mode==1: also write the true final state (ls[] holds it, since it was updated in
    // place through the whole token loop) -- fixed cost, not scaled by K, unconditional on K.
    if (emit_ingr) {
        device float * dst_final = (device float *) (dst) + final_state_base + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_final[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
#undef WRITE_ROWS
#undef HAS_ROWS
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device const char * rows,
        device const char * write_rows,
        device       char * state_dst,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif
