//! Zero-copy mmap GGUF v3 reader tailored for Kintsugi-27B & DFlash2.
//! Parses 2,069 tensors + metadata in ~1 ms directly from mapped file bytes.

const std = @import("std");

const posix = struct {
    extern "c" fn open(path: [*:0]const u8, oflag: c_int, ...) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
    extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
    extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
};

const O_RDONLY: c_int = 0;
const SEEK_END: c_int = 2;
const PROT_READ: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MAP_FAILED_ADDR: usize = @bitCast(@as(isize, -1));

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 30,
    _,

    pub fn blockSize(self: GgmlType) u64 {
        return switch (self) {
            .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            _ => 1,
        };
    }

    pub fn typeSize(self: GgmlType) u64 {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8 => 1,
            .i64, .f64 => 8,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            _ => 0,
        };
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    ggml_type: GgmlType,
    offset: u64, // relative to data_offset

    pub fn nElements(self: *const TensorInfo) u64 {
        var n: u64 = 1;
        for (0..self.n_dims) |i| n *= self.dims[i];
        return n;
    }

    pub fn byteSize(self: *const TensorInfo) u64 {
        const bs = self.ggml_type.blockSize();
        const ts = self.ggml_type.typeSize();
        return (self.nElements() / bs) * ts;
    }
};

pub const GgufFile = struct {
    fd: c_int,
    mapped: []align(4096) const u8,
    version: u32,
    n_tensors: u64,
    n_kv: u64,
    alignment: u64,
    data_offset: u64,
    architecture: []const u8,
    tensors: []TensorInfo,
    index: std.StringHashMapUnmanaged(usize),

    pub fn open(allocator: std.mem.Allocator, path_z: [*:0]const u8) !GgufFile {
        const fd = posix.open(path_z, O_RDONLY);
        if (fd < 0) return error.OpenFailed;
        errdefer _ = posix.close(fd);

        const end_off = posix.lseek(fd, 0, SEEK_END);
        if (end_off < 24) return error.InvalidGguf;
        const file_size: usize = @intCast(end_off);

        const raw = posix.mmap(null, file_size, PROT_READ, MAP_PRIVATE, fd, 0) orelse return error.MmapFailed;
        if (@intFromPtr(raw) == MAP_FAILED_ADDR) return error.MmapFailed;
        const mapped: []align(4096) const u8 = @as([*]align(4096) const u8, @ptrCast(@alignCast(raw)))[0..file_size];
        errdefer _ = posix.munmap(raw, file_size);

        var cur: usize = 0;
        if (!std.mem.eql(u8, mapped[0..4], "GGUF")) return error.InvalidMagic;
        cur += 4;

        const version = readInt(u32, mapped, &cur);
        if (version < 2 or version > 3) return error.UnsupportedVersion;
        const n_tensors = readInt(u64, mapped, &cur);
        const n_kv = readInt(u64, mapped, &cur);

        var alignment: u64 = 32;
        var arch: []const u8 = "unknown";

        for (0..n_kv) |_| {
            const key = readStr(mapped, &cur);
            const vtype = readInt(u32, mapped, &cur);
            if (std.mem.eql(u8, key, "general.alignment") and vtype == 4) {
                alignment = readInt(u32, mapped, &cur);
            } else if (std.mem.eql(u8, key, "general.architecture") and vtype == 8) {
                arch = readStr(mapped, &cur);
            } else {
                try skipValue(mapped, &cur, vtype);
            }
        }

        const tensors = try allocator.alloc(TensorInfo, @intCast(n_tensors));
        errdefer allocator.free(tensors);

        var index: std.StringHashMapUnmanaged(usize) = .empty;
        errdefer index.deinit(allocator);
        try index.ensureTotalCapacity(allocator, @intCast(n_tensors));

        for (0..@as(usize, @intCast(n_tensors))) |i| {
            const name = readStr(mapped, &cur);
            const n_dims = readInt(u32, mapped, &cur);
            if (n_dims > 4) return error.TooManyDims;
            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            for (0..n_dims) |d| {
                dims[d] = readInt(u64, mapped, &cur);
            }
            const ttype: GgmlType = @enumFromInt(readInt(u32, mapped, &cur));
            const offset = readInt(u64, mapped, &cur);
            tensors[i] = .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .ggml_type = ttype,
                .offset = offset,
            };
            index.putAssumeCapacity(name, i);
        }

        const data_offset = alignUp(@as(u64, cur), alignment);

        return .{
            .fd = fd,
            .mapped = mapped,
            .version = version,
            .n_tensors = n_tensors,
            .n_kv = n_kv,
            .alignment = alignment,
            .data_offset = data_offset,
            .architecture = arch,
            .tensors = tensors,
            .index = index,
        };
    }

    pub fn deinit(self: *GgufFile, allocator: std.mem.Allocator) void {
        self.index.deinit(allocator);
        allocator.free(self.tensors);
        _ = posix.munmap(@constCast(@as(*const anyopaque, @ptrCast(self.mapped.ptr))), self.mapped.len);
        _ = posix.close(self.fd);
    }

    pub fn find(self: *const GgufFile, name: []const u8) ?*const TensorInfo {
        const idx = self.index.get(name) orelse return null;
        return &self.tensors[idx];
    }

    pub fn tensorBytes(self: *const GgufFile, info: *const TensorInfo) []const u8 {
        const start: usize = @intCast(self.data_offset + info.offset);
        const len: usize = @intCast(info.byteSize());
        return self.mapped[start .. start + len];
    }

    pub const TokenizerMeta = struct {
        tokens: [][]const u8,
        merges: [][]const u8,
        token_types: []i32,
        eos_id: u32 = 248046,
        bos_id: u32 = 248044,
        pad_id: u32 = 248044,
        mask_id: u32 = 248070,

        pub fn deinit(self: *TokenizerMeta, allocator: std.mem.Allocator) void {
            allocator.free(self.tokens);
            allocator.free(self.merges);
            allocator.free(self.token_types);
        }
    };

    pub fn readTokenizerMeta(self: *const GgufFile, allocator: std.mem.Allocator) !TokenizerMeta {
        var cur: usize = 4 + 4 + 8 + 8; // skip magic, version, n_tensors, n_kv
        var meta: TokenizerMeta = .{
            .tokens = &.{},
            .merges = &.{},
            .token_types = &.{},
        };
        errdefer {
            if (meta.tokens.len > 0) allocator.free(meta.tokens);
            if (meta.merges.len > 0) allocator.free(meta.merges);
            if (meta.token_types.len > 0) allocator.free(meta.token_types);
        }

        for (0..@as(usize, @intCast(self.n_kv))) |_| {
            const key = readStr(self.mapped, &cur);
            const vtype = readInt(u32, self.mapped, &cur);
            if (std.mem.eql(u8, key, "tokenizer.ggml.tokens") and vtype == 9) {
                const elem_type = readInt(u32, self.mapped, &cur);
                const len: usize = @intCast(readInt(u64, self.mapped, &cur));
                if (elem_type != 8) return error.InvalidTokenizerMeta;
                const arr = try allocator.alloc([]const u8, len);
                for (0..len) |i| arr[i] = readStr(self.mapped, &cur);
                meta.tokens = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges") and vtype == 9) {
                const elem_type = readInt(u32, self.mapped, &cur);
                const len: usize = @intCast(readInt(u64, self.mapped, &cur));
                if (elem_type != 8) return error.InvalidTokenizerMeta;
                const arr = try allocator.alloc([]const u8, len);
                for (0..len) |i| arr[i] = readStr(self.mapped, &cur);
                meta.merges = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.token_type") and vtype == 9) {
                const elem_type = readInt(u32, self.mapped, &cur);
                const len: usize = @intCast(readInt(u64, self.mapped, &cur));
                if (elem_type != 5) return error.InvalidTokenizerMeta;
                const arr = try allocator.alloc(i32, len);
                for (0..len) |i| arr[i] = readInt(i32, self.mapped, &cur);
                meta.token_types = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id") and vtype == 4) {
                meta.eos_id = readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id") and vtype == 4) {
                meta.bos_id = readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.padding_token_id") and vtype == 4) {
                meta.pad_id = readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.mask_token_id") and vtype == 4) {
                meta.mask_id = readInt(u32, self.mapped, &cur);
            } else {
                try skipValue(self.mapped, &cur, vtype);
            }
        }
        return meta;
    }

    pub fn readDflashTargetLayers(self: *const GgufFile, out: *[5]u32) !usize {
        var cur: usize = 4 + 4 + 8 + 8;
        for (0..@as(usize, @intCast(self.n_kv))) |_| {
            const key = readStr(self.mapped, &cur);
            const vtype = readInt(u32, self.mapped, &cur);
            if (std.mem.eql(u8, key, "dflash.target_layers") and vtype == 9) {
                const elem_type = readInt(u32, self.mapped, &cur);
                const len: usize = @intCast(readInt(u64, self.mapped, &cur));
                const n = @min(len, out.len);
                for (0..len) |i| {
                    const v: u32 = switch (elem_type) {
                        4 => readInt(u32, self.mapped, &cur),
                        5 => @intCast(readInt(i32, self.mapped, &cur)),
                        else => return error.InvalidTargetLayers,
                    };
                    if (i < n) out[i] = v;
                }
                return n;
            } else {
                try skipValue(self.mapped, &cur, vtype);
            }
        }
        return 0;
    }
};

fn alignUp(v: u64, a: u64) u64 {
    return (v + a - 1) & ~(a - 1);
}

fn readInt(comptime T: type, buf: []const u8, cur: *usize) T {
    const sz = @sizeOf(T);
    const val = std.mem.readInt(T, buf[cur.*..][0..sz], .little);
    cur.* += sz;
    return val;
}

fn readStr(buf: []const u8, cur: *usize) []const u8 {
    const len: usize = @intCast(readInt(u64, buf, cur));
    const s = buf[cur.* .. cur.* + len];
    cur.* += len;
    return s;
}

fn skipValue(buf: []const u8, cur: *usize, vtype: u32) !void {
    switch (vtype) {
        0, 1, 7 => cur.* += 1,
        2, 3 => cur.* += 2,
        4, 5, 6 => cur.* += 4,
        10, 11, 12 => cur.* += 8,
        8 => _ = readStr(buf, cur),
        9 => {
            const elem_type = readInt(u32, buf, cur);
            const len: usize = @intCast(readInt(u64, buf, cur));
            switch (elem_type) {
                0, 1, 7 => cur.* += len,
                2, 3 => cur.* += len * 2,
                4, 5, 6 => cur.* += len * 4,
                10, 11, 12 => cur.* += len * 8,
                else => {
                    for (0..len) |_| try skipValue(buf, cur, elem_type);
                },
            }
        },
        else => return error.UnknownGgufValueType,
    }
}

test "GgmlType byte size calculation" {
    const q4: TensorInfo = .{
        .name = "test_q4",
        .n_dims = 2,
        .dims = .{ 5120, 248320, 1, 1 },
        .ggml_type = .q4_0,
        .offset = 0,
    };
    // 5120 * 248320 / 32 * 18 = 715,161,600 B (~682 MiB)
    try std.testing.expectEqual(@as(u64, 715_161_600), q4.byteSize());
}
