//! Static comptime topology and VRAM budget math for Kintsugi-27B (Qwen3.8-27B)
//! + DFlash2 on NVIDIA RTX 4070 Ti (Ada Lovelace, sm_89, 12 GB).

pub const ModelSpec = struct {
    pub const n_layers: u32 = 64;
    pub const full_attn_interval: u32 = 4; // layers 3, 7, 11, ..., 63 are full attention
    pub const n_full_attn_layers: u32 = n_layers / full_attn_interval; // 16
    pub const n_gdn_layers: u32 = n_layers - n_full_attn_layers; // 48

    pub const d_model: u32 = 5120;
    pub const d_ff: u32 = 17408;
    pub const n_vocab: u32 = 248320;

    // Full-attention geometry
    pub const n_heads: u32 = 24;
    pub const n_kv_heads: u32 = 4;
    pub const head_dim: u32 = 256;
    pub const rope_dim: u32 = 64;
    pub const rope_theta: f32 = 10_000_000.0;
    pub const rms_eps: f32 = 1.0e-6;

    // GDN (Gated DeltaNet) linear-attention geometry
    pub const ssm_conv_kernel: u32 = 4;
    pub const ssm_state_size: u32 = 128;
    pub const ssm_group_count: u32 = 16;
    pub const ssm_time_step_rank: u32 = 48;
    pub const ssm_inner_size: u32 = 6144;

    // DFlash2 speculative decoding geometry
    pub const dflash_layers: u32 = 5;
    pub const dflash_block_size: u32 = 8; // 1 anchor + 7 draft tokens
    pub const dflash_n_max: u32 = 7;

    // Context & KV cache (Q4_0: 18 bytes per 32 elements = 0.5625 B/elem)
    pub const max_ctx: u32 = 262_144;
    pub const kv_bytes_per_token_per_layer: u64 = 2 * n_kv_heads * (head_dim / 32) * 18; // 1152 B
    pub const kv_bytes_per_token: u64 = kv_bytes_per_token_per_layer * n_full_attn_layers; // 18,432 B

    // GDN recurrent state: 48 layers * (48 heads * 128 * 128 * 4 B + conv state)
    pub const gdn_state_bytes: u64 = @as(u64, n_gdn_layers) *
        (@as(u64, ssm_time_step_rank) * ssm_state_size * ssm_state_size * 4 +
            @as(u64, ssm_inner_size) * (ssm_conv_kernel - 1) * 4);

    // Static decode scratchpad for M=8 verification (tiny compared to ggml's 308 MiB!)
    // Includes residual, norm/hadamard f16, FFN gate/up/down, attn QKV, logits (stage-1)
    pub const decode_scratch_bytes: u64 = 32 * 1024 * 1024; // 32 MiB static arena (covers M=8 + chunked prefill 512)

    pub fn isFullAttnLayer(layer: u32) bool {
        return ((layer + 1) % full_attn_interval) == 0;
    }
};

pub const VramPlan = struct {
    total_bytes: u64,
    free_bytes: u64,
    safety_margin_bytes: u64,
    trunk_weights_bytes: u64,
    head_stage1_bytes: u64,
    dflash_weights_bytes: u64,
    gdn_state_bytes: u64,
    scratch_bytes: u64,
    available_for_kv_bytes: u64,
    max_vram_kv_cells: u64,
    full_256k_kv_bytes: u64,
    host_spill_bytes: u64,

    pub fn compute(
        total_bytes: u64,
        free_bytes: u64,
        trunk_weights_bytes: u64,
        dflash_weights_bytes: u64,
        head_stage1_bytes: u64,
        safety_margin_mib: u64,
    ) VramPlan {
        const margin = safety_margin_mib * 1024 * 1024;
        const fixed = trunk_weights_bytes + head_stage1_bytes + dflash_weights_bytes +
            ModelSpec.gdn_state_bytes + ModelSpec.decode_scratch_bytes;
        const budget = if (free_bytes > margin) free_bytes - margin else 0;
        const kv_avail = if (budget > fixed) budget - fixed else 0;
        // Align KV cells down to 4096-token pages (matches CUDA VMM 2 MiB granularity nicely)
        const raw_cells = kv_avail / ModelSpec.kv_bytes_per_token;
        const cells = (raw_cells / 4096) * 4096;
        const full_kv = @as(u64, ModelSpec.max_ctx) * ModelSpec.kv_bytes_per_token;
        const vram_kv_used = cells * ModelSpec.kv_bytes_per_token;
        const spill = if (full_kv > vram_kv_used) full_kv - vram_kv_used else 0;

        return .{
            .total_bytes = total_bytes,
            .free_bytes = free_bytes,
            .safety_margin_bytes = margin,
            .trunk_weights_bytes = trunk_weights_bytes,
            .head_stage1_bytes = head_stage1_bytes,
            .dflash_weights_bytes = dflash_weights_bytes,
            .gdn_state_bytes = ModelSpec.gdn_state_bytes,
            .scratch_bytes = ModelSpec.decode_scratch_bytes,
            .available_for_kv_bytes = kv_avail,
            .max_vram_kv_cells = cells,
            .full_256k_kv_bytes = full_kv,
            .host_spill_bytes = spill,
        };
    }
};

test "Qwen3.8-27B layer topology and KV byte math" {
    const std = @import("std");
    var full_cnt: u32 = 0;
    var gdn_cnt: u32 = 0;
    for (0..ModelSpec.n_layers) |l| {
        if (ModelSpec.isFullAttnLayer(@intCast(l))) {
            full_cnt += 1;
        } else {
            gdn_cnt += 1;
        }
    }
    try std.testing.expectEqual(@as(u32, 16), full_cnt);
    try std.testing.expectEqual(@as(u32, 48), gdn_cnt);
    try std.testing.expectEqual(@as(u64, 18_432), ModelSpec.kv_bytes_per_token);
    // 262,144 * 18,432 = 4,831,838,208 B (4608 MiB)
    try std.testing.expectEqual(@as(u64, 4_831_838_208), @as(u64, ModelSpec.max_ctx) * ModelSpec.kv_bytes_per_token);
}
