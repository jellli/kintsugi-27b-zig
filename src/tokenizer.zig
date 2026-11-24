//! Qwen3.5 / Kintsugi-27B GPT-2 Byte-Level BPE Tokenizer directly loaded from GGUF metadata.
//! Supports encoding (with special tokens + Qwen3.5 pre-tokenization) and streaming UTF-8 decoding.

const std = @import("std");
const gguf = @import("gguf.zig");

const SpecialToken = struct {
    text: []const u8,
    id: u32,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    meta: gguf.GgufFile.TokenizerMeta,
    vocab_map: std.StringHashMapUnmanaged(u32) = .empty,
    merge_rank: std.StringHashMapUnmanaged(u32) = .empty,
    specials: std.ArrayListUnmanaged(SpecialToken) = .empty,
    /// Maps each byte 0..255 to its GPT-2 Unicode codepoint (stored as UTF-8 bytes)
    byte_to_utf8: [256][2]u8 = undefined,
    byte_to_utf8_len: [256]u8 = undefined,
    /// Maps GPT-2 Unicode codepoints 0..324 -> original byte (or 0xFFFF if invalid)
    cp_to_byte: [325]u16 = undefined,

    pub fn init(allocator: std.mem.Allocator, model: *const gguf.GgufFile) !Tokenizer {
        var meta = try model.readTokenizerMeta(allocator);
        errdefer meta.deinit(allocator);

        var self: Tokenizer = .{
            .allocator = allocator,
            .meta = meta,
        };
        self.initByteTables();

        try self.vocab_map.ensureTotalCapacity(allocator, @intCast(meta.tokens.len));
        errdefer self.vocab_map.deinit(allocator);
        for (meta.tokens, 0..) |tok, idx| {
            self.vocab_map.putAssumeCapacity(tok, @intCast(idx));
            if (idx < meta.token_types.len and (meta.token_types[idx] == 3 or meta.token_types[idx] == 4) and tok.len > 0) {
                try self.specials.append(allocator, .{ .text = tok, .id = @intCast(idx) });
            }
        }
        // Sort specials longest-first for greedy matching
        std.mem.sort(SpecialToken, self.specials.items, {}, struct {
            fn lessThan(_: void, a: SpecialToken, b: SpecialToken) bool {
                return a.text.len > b.text.len;
            }
        }.lessThan);

        try self.merge_rank.ensureTotalCapacity(allocator, @intCast(meta.merges.len));
        errdefer self.merge_rank.deinit(allocator);
        for (meta.merges, 0..) |m, rank| {
            self.merge_rank.putAssumeCapacity(m, @intCast(rank));
        }

        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.vocab_map.deinit(self.allocator);
        self.merge_rank.deinit(self.allocator);
        self.specials.deinit(self.allocator);
        self.meta.deinit(self.allocator);
    }

    fn initByteTables(self: *Tokenizer) void {
        @memset(&self.cp_to_byte, 0xFFFF);
        var next_cp: u21 = 256;
        for (0..256) |b| {
            const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
            const cp: u21 = if (printable) @intCast(b) else blk: {
                const c = next_cp;
                next_cp += 1;
                break :blk c;
            };
            self.cp_to_byte[cp] = @intCast(b);
            var buf: [4]u8 = .{ 0, 0, 0, 0 };
            const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
            self.byte_to_utf8[b] = .{ buf[0], buf[1] };
            self.byte_to_utf8_len[b] = @intCast(n);
        }
    }

    pub fn isEos(self: *const Tokenizer, id: u32) bool {
        return id == self.meta.eos_id or id == self.meta.bos_id; // 248046 <|im_end|>, 248044 <|endoftext|>
    }

    pub fn decodeTokenTo(self: *const Tokenizer, id: u32, out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator) !void {
        if (id >= self.meta.tokens.len) return;
        const piece = self.meta.tokens[id];
        if (id < self.meta.token_types.len and (self.meta.token_types[id] == 3 or self.meta.token_types[id] == 4)) {
            // Control / special token: emit literal text (skip for clean chat output if desired by caller)
            try out.appendSlice(allocator, piece);
            return;
        }
        var i: usize = 0;
        while (i < piece.len) {
            const clen = std.unicode.utf8ByteSequenceLength(piece[i]) catch 1;
            if (i + clen > piece.len) {
                try out.append(allocator, piece[i]);
                i += 1;
                continue;
            }
            const cp = std.unicode.utf8Decode(piece[i .. i + clen]) catch {
                try out.append(allocator, piece[i]);
                i += 1;
                continue;
            };
            if (cp < self.cp_to_byte.len and self.cp_to_byte[cp] != 0xFFFF) {
                try out.append(allocator, @intCast(self.cp_to_byte[cp]));
            } else {
                try out.appendSlice(allocator, piece[i .. i + clen]);
            }
            i += clen;
        }
    }

    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            try self.decodeTokenTo(id, &out, allocator);
        }
        return try out.toOwnedSlice(allocator);
    }

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayListUnmanaged(u32) = .empty;
        errdefer ids.deinit(allocator);

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            var matched: ?SpecialToken = null;
            if (text[pos] == '<') {
                for (self.specials.items) |sp| {
                    if (pos + sp.text.len <= text.len and std.mem.eql(u8, text[pos .. pos + sp.text.len], sp.text)) {
                        matched = sp;
                        break;
                    }
                }
            }
            if (matched) |sp| {
                if (pos > seg_start) {
                    try self.encodePlainSegment(allocator, text[seg_start..pos], &ids);
                }
                try ids.append(allocator, sp.id);
                pos += sp.text.len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        if (seg_start < text.len) {
            try self.encodePlainSegment(allocator, text[seg_start..], &ids);
        }
        return try ids.toOwnedSlice(allocator);
    }

    /// Qwen3.5 / Qwen2 regex-equivalent pre-tokenizer:
    ///   '(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    fn encodePlainSegment(self: *const Tokenizer, allocator: std.mem.Allocator, seg: []const u8, out: *std.ArrayListUnmanaged(u32)) !void {
        var i: usize = 0;
        while (i < seg.len) {
            const start = i;

            // 1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive)
            if (seg[i] == '\'') {
                if (matchContraction(seg[i..])) |clen| {
                    try self.bpePiece(allocator, seg[start .. start + clen], out);
                    i += clen;
                    continue;
                }
            }

            // 2. Single ASCII digit (\p{N})
            if (isAsciiDigit(seg[i])) {
                try self.bpePiece(allocator, seg[i .. i + 1], out);
                i += 1;
                continue;
            }

            // 3. Optional non-\r\n-letter-digit prefix + letters (\p{L}+)
            if (isLetterAt(seg, i)) {
                i = consumeLetters(seg, i);
                try self.bpePiece(allocator, seg[start..i], out);
                continue;
            }
            if (seg[i] != '\r' and seg[i] != '\n' and !isAsciiDigit(seg[i])) {
                const step = utf8Step(seg, i);
                if (i + step < seg.len and isLetterAt(seg, i + step)) {
                    i = consumeLetters(seg, i + step);
                    try self.bpePiece(allocator, seg[start..i], out);
                    continue;
                }
            }

            // 4. Optional space + non-ws-letter-digit punctuation + trailing [\r\n]*
            {
                var p = i;
                if (seg[p] == ' ' and p + 1 < seg.len and !isSpace(seg[p + 1]) and !isLetterAt(seg, p + 1) and !isAsciiDigit(seg[p + 1])) {
                    p += 1;
                }
                if (p < seg.len and !isSpace(seg[p]) and !isLetterAt(seg, p) and !isAsciiDigit(seg[p])) {
                    while (p < seg.len and !isSpace(seg[p]) and !isLetterAt(seg, p) and !isAsciiDigit(seg[p])) {
                        p += utf8Step(seg, p);
                    }
                    while (p < seg.len and (seg[p] == '\r' or seg[p] == '\n')) : (p += 1) {}
                    try self.bpePiece(allocator, seg[start..p], out);
                    i = p;
                    continue;
                }
            }

            // 5. Whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
            if (isSpace(seg[i])) {
                var p = i;
                var last_nl: ?usize = null;
                while (p < seg.len and isSpace(seg[p])) : (p += 1) {
                    if (seg[p] == '\r' or seg[p] == '\n') last_nl = p;
                }
                if (last_nl) |nl| {
                    try self.bpePiece(allocator, seg[start .. nl + 1], out);
                    i = nl + 1;
                } else if (p < seg.len and p - start > 1) {
                    // \s+(?!\S): leave the last space attached to the following token
                    try self.bpePiece(allocator, seg[start .. p - 1], out);
                    i = p - 1;
                } else {
                    try self.bpePiece(allocator, seg[start..p], out);
                    i = p;
                }
                continue;
            }

            // Fallback single byte
            try self.bpePiece(allocator, seg[i .. i + 1], out);
            i += 1;
        }
    }

    fn bpePiece(self: *const Tokenizer, allocator: std.mem.Allocator, raw: []const u8, out: *std.ArrayListUnmanaged(u32)) !void {
        if (raw.len == 0) return;

        // Convert raw bytes to GPT-2 byte-unicode UTF-8 buffer
        var encoded: std.ArrayListUnmanaged(u8) = .empty;
        defer encoded.deinit(allocator);
        try encoded.ensureTotalCapacity(allocator, raw.len * 2);

        const Sym = struct { start: u32, len: u32 };
        var syms: std.ArrayListUnmanaged(Sym) = .empty;
        defer syms.deinit(allocator);
        try syms.ensureTotalCapacity(allocator, raw.len);

        for (raw) |b| {
            const s: u32 = @intCast(encoded.items.len);
            const l = self.byte_to_utf8_len[b];
            encoded.appendSliceAssumeCapacity(self.byte_to_utf8[b][0..l]);
            syms.appendAssumeCapacity(.{ .start = s, .len = l });
        }

        // Fast path: whole piece already in vocab
        if (self.vocab_map.get(encoded.items)) |tok_id| {
            try out.append(allocator, tok_id);
            return;
        }

        var pair_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer pair_buf.deinit(allocator);

        while (syms.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: usize = 0;

            for (0..syms.items.len - 1) |k| {
                const a = encoded.items[syms.items[k].start .. syms.items[k].start + syms.items[k].len];
                const b = encoded.items[syms.items[k + 1].start .. syms.items[k + 1].start + syms.items[k + 1].len];
                pair_buf.clearRetainingCapacity();
                try pair_buf.ensureTotalCapacity(allocator, a.len + 1 + b.len);
                pair_buf.appendSliceAssumeCapacity(a);
                pair_buf.appendAssumeCapacity(' ');
                pair_buf.appendSliceAssumeCapacity(b);
                if (self.merge_rank.get(pair_buf.items)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = k;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;

            // Merge best_idx and best_idx + 1 (always contiguous in `encoded`!)
            syms.items[best_idx].len += syms.items[best_idx + 1].len;
            _ = syms.orderedRemove(best_idx + 1);
        }

        for (syms.items) |sym| {
            const piece = encoded.items[sym.start .. sym.start + sym.len];
            if (self.vocab_map.get(piece)) |id| {
                try out.append(allocator, id);
            } else {
                // Byte-level fallback (every single GPT-2 byte char is guaranteed in vocab)
                var p: usize = 0;
                while (p < piece.len) {
                    const cl = std.unicode.utf8ByteSequenceLength(piece[p]) catch 1;
                    const sub = piece[p .. @min(p + cl, piece.len)];
                    if (self.vocab_map.get(sub)) |bid| try out.append(allocator, bid);
                    p += cl;
                }
            }
        }
    }
};

fn matchContraction(s: []const u8) ?usize {
    if (s.len < 2 or s[0] != '\'') return null;
    const c1 = std.ascii.toLower(s[1]);
    if (c1 == 's' or c1 == 'd' or c1 == 'm' or c1 == 't') return 2;
    if (s.len >= 3) {
        const c2 = std.ascii.toLower(s[2]);
        if ((c1 == 'l' and c2 == 'l') or (c1 == 'v' and c2 == 'e') or (c1 == 'r' and c2 == 'e')) return 3;
    }
    return null;
}

fn isAsciiDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

fn utf8Step(s: []const u8, i: usize) usize {
    const l = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
    return @min(l, s.len - i);
}

fn isLetterAt(s: []const u8, i: usize) bool {
    const c = s[i];
    if ((c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z')) return true;
    if (c < 0x80) return false;
    // Non-ASCII UTF-8: treat CJK / accented / Cyrillic / Greek codepoints as letters (exclude common punctuation blocks)
    const step = utf8Step(s, i);
    const cp = std.unicode.utf8Decode(s[i .. i + step]) catch return false;
    if (cp >= 0x2000 and cp <= 0x206F) return false; // General punctuation
    if (cp >= 0x3000 and cp <= 0x303F) return false; // CJK symbols & punctuation
    if (cp >= 0xFF00 and cp <= 0xFF0F) return false; // Fullwidth ASCII punctuation
    if (cp >= 0xFF1A and cp <= 0xFF20) return false;
    return cp >= 0x00C0;
}

fn consumeLetters(s: []const u8, start: usize) usize {
    var i = start;
    while (i < s.len and isLetterAt(s, i)) {
        i += utf8Step(s, i);
    }
    return i;
}
