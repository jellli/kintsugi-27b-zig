//! Independent CPU ground-truth Escha ~2.5-bit trellis decoder and matmul verifier.
//! Shares ZERO code with kernels/escha_sm89.cu — decodes tiles directly via GGUF escha_dep_k2/k3.

const std = @import("std");

pub fn f16BitsToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn f32ToF16Bits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

/// Escha MCG codebook A (exact match to ggml_escha_codebook).
pub fn eschaCodebook(idx: u32) f32 {
    const x: u32 = ((idx *% 0xcbac1fed) & 0x8fff8fff) ^ 0x3b603b60;
    const lo: f16 = @bitCast(@as(u16, @truncate(x)));
    const hi: f16 = @bitCast(@as(u16, @truncate(x >> 16)));
    return @floatCast(lo + hi);
}

/// Orthonormal 128-pt Sylvester-Hadamard transform in place over every 128-elem block.
pub fn hadamard128InPlace(v: []f32) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, 128.0));
    var off: usize = 0;
    while (off < v.len) : (off += 128) {
        const b = v[off .. off + 128];
        var len: usize = 1;
        while (len < 128) : (len <<= 1) {
            var i: usize = 0;
            while (i < 128) : (i += len << 1) {
                for (0..len) |j| {
                    const a0 = b[i + j];
                    const a1 = b[i + j + len];
                    b[i + j] = a0 + a1;
                    b[i + j + len] = a0 - a1;
                }
            }
        }
        for (b) |*x| x.* *= scale;
    }
}

/// Decode one 16x16 Escha tile into row-major [16][16] f32 using GGUF dep table [256][16] i16.
pub fn decodeTile(payload: []const u8, dep: []const i16, tile: *[256]f32) void {
    for (0..256) |p| {
        const d = dep[p * 16 ..][0..16];
        var idx: u32 = 0;
        for (0..16) |b| {
            const bit_pos: usize = @intCast(d[b]);
            const bit: u32 = (payload[bit_pos >> 3] >> @as(u3, @truncate(bit_pos & 7))) & 1;
            idx |= bit << @as(u5, @truncate(b));
        }
        tile[p] = eschaCodebook(idx);
    }
}

/// Reference single-row Escha matmul: y[OC] = T128(T128(x * rin) @ W) * rout.
/// Decodes every tile independently from raw GGUF escha_code + dep bytes.
pub fn eschaMatVecCpuRef(
    allocator: std.mem.Allocator,
    code_bytes: []const u8,
    rin_f16: []const u16,
    rout_f16: []const u16,
    dep: []const i16,
    x: []const f32,
    y_out: []f32,
    k_bits: usize,
    ic: usize,
    oc: usize,
) !void {
    const nit = ic / 16;
    const nct = oc / 16;
    const tile_bytes = 32 * k_bits; // 16*K int16 = 32*K bytes

    const u = try allocator.alloc(f32, ic);
    defer allocator.free(u);
    for (0..ic) |i| {
        u[i] = x[i] * f16BitsToF32(rin_f16[i]);
    }
    hadamard128InPlace(u);

    @memset(y_out, 0.0);
    var tile: [256]f32 = undefined;

    for (0..nit) |i| {
        const ub = u[i * 16 ..][0..16];
        for (0..nct) |j| {
            const t_off = (i * nct + j) * tile_bytes;
            decodeTile(code_bytes[t_off .. t_off + tile_bytes], dep, &tile);
            const ab = y_out[j * 16 ..][0..16];
            for (0..16) |r| {
                const ur = ub[r];
                if (ur == 0.0) continue;
                const tr = tile[r * 16 ..][0..16];
                for (0..16) |c| {
                    ab[c] += ur * tr[c];
                }
            }
        }
    }

    hadamard128InPlace(y_out);
    for (0..oc) |c| {
        y_out[c] *= f16BitsToF32(rout_f16[c]);
    }
}

test "eschaCodebook sanity and hadamard128 involution" {
    // Hadamard H128 is orthonormal: H(H(x)) == x
    var v: [128]f32 = undefined;
    var orig: [128]f32 = undefined;
    for (0..128) |i| {
        const val: f32 = @sin(@as(f32, @floatFromInt(i)) * 0.17);
        v[i] = val;
        orig[i] = val;
    }
    hadamard128InPlace(&v);
    hadamard128InPlace(&v);
    for (0..128) |i| {
        try std.testing.expectApproxEqAbs(orig[i], v[i], 1.0e-5);
    }
    // Codebook output is finite and bounded
    for ([_]u32{ 0, 1, 42, 12345, 65535 }) |idx| {
        const c = eschaCodebook(idx);
        try std.testing.expect(std.math.isFinite(c));
    }
}
