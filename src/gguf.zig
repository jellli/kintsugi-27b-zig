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
        for (0..self.n_dims) |i| n *|= self.dims[i];
        return n;
    }

    pub fn checkedNElements(self: *const TensorInfo) !u64 {
        var n: u64 = 1;
        for (0..self.n_dims) |i| {
            n = std.math.mul(u64, n, self.dims[i]) catch return error.InvalidGguf;
        }
        return n;
    }

    pub fn byteSize(self: *const TensorInfo) u64 {
        const bs = self.ggml_type.blockSize();
        const ts = self.ggml_type.typeSize();
        return (self.nElements() / bs) *| ts;
    }

    pub fn checkedByteSize(self: *const TensorInfo) !u64 {
        const bs = self.ggml_type.blockSize();
        const ts = self.ggml_type.typeSize();
        const ne = try self.checkedNElements();
        if (ne % bs != 0) return error.InvalidGguf;
        return std.math.mul(u64, ne / bs, ts) catch return error.InvalidGguf;
    }
};

const MAX_SKIP_DEPTH: usize = 16;

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

        const version = try readInt(u32, mapped, &cur);
        if (version < 2 or version > 3) return error.UnsupportedVersion;
        const n_tensors = try readInt(u64, mapped, &cur);
        const n_kv = try readInt(u64, mapped, &cur);
        // Sanity cap against absurd counts in corrupt headers (each entry consumes >= 8 bytes)
        if (n_tensors > file_size / 8 or n_kv > file_size / 8) return error.InvalidGguf;

        var alignment: u64 = 32;
        var arch: []const u8 = "unknown";

        for (0..@as(usize, @intCast(n_kv))) |_| {
            const key = try readStr(mapped, &cur);
            const vtype = try readInt(u32, mapped, &cur);
            if (std.mem.eql(u8, key, "general.alignment") and vtype == 4) {
                alignment = try readInt(u32, mapped, &cur);
                if (alignment == 0 or (alignment & (alignment - 1)) != 0) return error.InvalidAlignment;
            } else if (std.mem.eql(u8, key, "general.architecture") and vtype == 8) {
                arch = try readStr(mapped, &cur);
            } else {
                try skipValue(mapped, &cur, vtype, 0);
            }
        }

        const n_tensors_usize: usize = @intCast(n_tensors);
        const tensors = try allocator.alloc(TensorInfo, n_tensors_usize);
        errdefer allocator.free(tensors);

        var index: std.StringHashMapUnmanaged(usize) = .empty;
        errdefer index.deinit(allocator);
        try index.ensureTotalCapacity(allocator, @intCast(n_tensors_usize));

        for (0..n_tensors_usize) |i| {
            const name = try readStr(mapped, &cur);
            const n_dims = try readInt(u32, mapped, &cur);
            if (n_dims > 4) return error.TooManyDims;
            var dims: [4]u64 = .{ 1, 1, 1, 1 };
            for (0..n_dims) |d| {
                dims[d] = try readInt(u64, mapped, &cur);
            }
            const ttype: GgmlType = @enumFromInt(try readInt(u32, mapped, &cur));
            const offset = try readInt(u64, mapped, &cur);
            tensors[i] = .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .ggml_type = ttype,
                .offset = offset,
            };
            index.putAssumeCapacity(name, i);
        }

        const data_offset = try alignUp(@as(u64, cur), alignment);
        if (data_offset > mapped.len) return error.UnexpectedEof;

        // Validate every tensor slice fits inside mapped file so tensorBytes() is always in-bounds
        for (tensors) |*t| {
            const bsz = try t.checkedByteSize();
            const start = std.math.add(u64, data_offset, t.offset) catch return error.InvalidGguf;
            const end = std.math.add(u64, start, bsz) catch return error.InvalidGguf;
            if (end > mapped.len) return error.UnexpectedEof;
        }

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
            const key = try readStr(self.mapped, &cur);
            const vtype = try readInt(u32, self.mapped, &cur);
            if (std.mem.eql(u8, key, "tokenizer.ggml.tokens") and vtype == 9) {
                const elem_type = try readInt(u32, self.mapped, &cur);
                const len_u64 = try readInt(u64, self.mapped, &cur);
                if (elem_type != 8 or len_u64 > self.mapped.len / 8) return error.InvalidTokenizerMeta;
                const len: usize = @intCast(len_u64);
                const arr = try allocator.alloc([]const u8, len);
                errdefer allocator.free(arr);
                for (0..len) |i| arr[i] = try readStr(self.mapped, &cur);
                if (meta.tokens.len > 0) allocator.free(meta.tokens);
                meta.tokens = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges") and vtype == 9) {
                const elem_type = try readInt(u32, self.mapped, &cur);
                const len_u64 = try readInt(u64, self.mapped, &cur);
                if (elem_type != 8 or len_u64 > self.mapped.len / 8) return error.InvalidTokenizerMeta;
                const len: usize = @intCast(len_u64);
                const arr = try allocator.alloc([]const u8, len);
                errdefer allocator.free(arr);
                for (0..len) |i| arr[i] = try readStr(self.mapped, &cur);
                if (meta.merges.len > 0) allocator.free(meta.merges);
                meta.merges = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.token_type") and vtype == 9) {
                const elem_type = try readInt(u32, self.mapped, &cur);
                const len_u64 = try readInt(u64, self.mapped, &cur);
                if (elem_type != 5 or len_u64 > self.mapped.len / 4) return error.InvalidTokenizerMeta;
                const len: usize = @intCast(len_u64);
                const arr = try allocator.alloc(i32, len);
                errdefer allocator.free(arr);
                for (0..len) |i| arr[i] = try readInt(i32, self.mapped, &cur);
                if (meta.token_types.len > 0) allocator.free(meta.token_types);
                meta.token_types = arr;
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id") and vtype == 4) {
                meta.eos_id = try readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id") and vtype == 4) {
                meta.bos_id = try readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.padding_token_id") and vtype == 4) {
                meta.pad_id = try readInt(u32, self.mapped, &cur);
            } else if (std.mem.eql(u8, key, "tokenizer.ggml.mask_token_id") and vtype == 4) {
                meta.mask_id = try readInt(u32, self.mapped, &cur);
            } else {
                try skipValue(self.mapped, &cur, vtype, 0);
            }
        }
        return meta;
    }

    pub fn readDflashTargetLayers(self: *const GgufFile, out: *[5]u32) !usize {
        var cur: usize = 4 + 4 + 8 + 8;
        for (0..@as(usize, @intCast(self.n_kv))) |_| {
            const key = try readStr(self.mapped, &cur);
            const vtype = try readInt(u32, self.mapped, &cur);
            if (std.mem.eql(u8, key, "dflash.target_layers") and vtype == 9) {
                const elem_type = try readInt(u32, self.mapped, &cur);
                const len_u64 = try readInt(u64, self.mapped, &cur);
                if (len_u64 > self.mapped.len / 4) return error.InvalidTargetLayers;
                const len: usize = @intCast(len_u64);
                const n = @min(len, out.len);
                for (0..len) |i| {
                    const v: u32 = switch (elem_type) {
                        4 => try readInt(u32, self.mapped, &cur),
                        5 => blk: {
                            const iv = try readInt(i32, self.mapped, &cur);
                            if (iv < 0) return error.InvalidTargetLayers;
                            break :blk @intCast(iv);
                        },
                        else => return error.InvalidTargetLayers,
                    };
                    if (i < n) out[i] = v;
                }
                return n;
            } else {
                try skipValue(self.mapped, &cur, vtype, 0);
            }
        }
        return 0;
    }
};

fn alignUp(v: u64, a: u64) !u64 {
    if (a == 0 or (a & (a - 1)) != 0) return error.InvalidAlignment;
    const sum = std.math.add(u64, v, a - 1) catch return error.InvalidGguf;
    return sum & ~(a - 1);
}

fn readInt(comptime T: type, buf: []const u8, cur: *usize) !T {
    const sz = @sizeOf(T);
    if (cur.* > buf.len or buf.len - cur.* < sz) return error.UnexpectedEof;
    const val = std.mem.readInt(T, buf[cur.*..][0..sz], .little);
    cur.* += sz;
    return val;
}

fn readStr(buf: []const u8, cur: *usize) ![]const u8 {
    const len_u64 = try readInt(u64, buf, cur);
    if (len_u64 > buf.len - cur.*) return error.UnexpectedEof;
    const len: usize = @intCast(len_u64);
    const s = buf[cur.* .. cur.* + len];
    cur.* += len;
    return s;
}

fn skipBytes(buf: []const u8, cur: *usize, n: usize) !void {
    if (cur.* > buf.len or buf.len - cur.* < n) return error.UnexpectedEof;
    cur.* += n;
}

fn skipValue(buf: []const u8, cur: *usize, vtype: u32, depth: usize) !void {
    if (depth > MAX_SKIP_DEPTH) return error.InvalidGguf;
    switch (vtype) {
        0, 1, 7 => try skipBytes(buf, cur, 1),
        2, 3 => try skipBytes(buf, cur, 2),
        4, 5, 6 => try skipBytes(buf, cur, 4),
        10, 11, 12 => try skipBytes(buf, cur, 8),
        8 => _ = try readStr(buf, cur),
        9 => {
            const elem_type = try readInt(u32, buf, cur);
            const len_u64 = try readInt(u64, buf, cur);
            const elem_size: ?u64 = switch (elem_type) {
                0, 1, 7 => 1,
                2, 3 => 2,
                4, 5, 6 => 4,
                10, 11, 12 => 8,
                else => null,
            };
            if (elem_size) |esz| {
                const total = std.math.mul(u64, len_u64, esz) catch return error.UnexpectedEof;
                if (total > buf.len - cur.*) return error.UnexpectedEof;
                cur.* += @intCast(total);
            } else {
                if (len_u64 > buf.len - cur.*) return error.UnexpectedEof;
                for (0..@as(usize, @intCast(len_u64))) |_| {
                    try skipValue(buf, cur, elem_type, depth + 1);
                }
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
    try std.testing.expectEqual(@as(u64, 715_161_600), try q4.checkedByteSize());
}

test "gguf bounds and overflow checks reject malformed data" {
    try std.testing.expectError(error.InvalidAlignment, alignUp(16, 0));
    try std.testing.expectError(error.InvalidAlignment, alignUp(16, 3));
    try std.testing.expectEqual(@as(u64, 32), try alignUp(17, 32));

    var cur: usize = 0;
    const tiny = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    try std.testing.expectError(error.UnexpectedEof, readStr(&tiny, &cur));

    // Array with huge element count overflowing len * 8
    cur = 0;
    const bad_arr = [_]u8{
        10, 0, 0, 0, // elem_type = u64
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x3f, // len = huge
    };
    try std.testing.expectError(error.UnexpectedEof, skipValue(&bad_arr, &cur, 9, 0));
}
