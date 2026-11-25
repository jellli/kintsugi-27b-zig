//! Zero-dependency OpenAI-compatible HTTP/1.1 + SSE Streaming Server for kintsugi-zig.
//!
//! Endpoints:
//!   GET  /health               -> {"status":"ok","model":"Kintsugi-27B","speculative":bool,"kv_cells":N}
//!   GET  /v1/models            -> OpenAI models list
//!   POST /tokenize             -> {"tokens":[...]}
//!   POST /v1/completions       -> OpenAI text completions (streaming SSE & non-streaming + timings)
//!   POST /v1/chat/completions  -> OpenAI chat completions (Qwen3.5 ChatML, streaming SSE & non-streaming)

const std = @import("std");
const cuda = @import("cuda.zig");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("tokenizer.zig");
const engine_mod = @import("engine.zig");
const dflash_mod = @import("dflash.zig");

const MSG_NOSIGNAL: u32 = 0x4000;
const MODEL_ID: []const u8 = "Kintsugi-27B";
const MAX_REQ_BODY: usize = 16 * 1024 * 1024;
const MAX_STOPS: usize = 8;

pub const ServeConfig = struct {
    host: []const u8 = "0.0.0.0",
    port: u16 = 8080,
    model_path: []const u8,
    drafter_path: []const u8,
    use_spec: bool = true,
};

const GenParams = struct {
    max_tokens: usize = 2048,
    temperature: f32 = 0.0,
    top_k: u32 = 64,
    top_p: f32 = 0.95,
    seed: u64 = 42,
    stream: bool = false,
    stops: [MAX_STOPS][]const u8 = undefined,
    n_stops: usize = 0,
};

const GenStats = struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    prompt_ms: f64 = 0.0,
    decode_ms: f64 = 0.0,
    spec_steps: usize = 0,
    draft_n: usize = 0,
    draft_n_accepted: usize = 0,
    finish_reason: []const u8 = "stop",
};

const Mode = enum {
    completion,
    chat,
};

fn nowMs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) / 1.0e6;
}

fn unixSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

pub fn parseIpv4(host: []const u8) !u32 {
    if (std.mem.eql(u8, host, "localhost")) {
        return std.mem.nativeToBig(u32, 0x7f000001);
    }
    var parts: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, host, '.');
    for (0..4) |i| {
        const seg = it.next() orelse return error.InvalidHost;
        parts[i] = std.fmt.parseInt(u8, seg, 10) catch return error.InvalidHost;
    }
    if (it.next() != null) return error.InvalidHost;
    const raw: u32 = (@as(u32, parts[0]) << 24) | (@as(u32, parts[1]) << 16) | (@as(u32, parts[2]) << 8) | @as(u32, parts[3]);
    return std.mem.nativeToBig(u32, raw);
}

/// Returns the longest prefix length of `buf` that ends on a complete UTF-8 codepoint boundary.
/// Any trailing incomplete 2/3/4-byte UTF-8 sequence is held back for the next token.
pub fn utf8ReadyPrefixLen(buf: []const u8) usize {
    if (buf.len == 0) return 0;
    const max_back = @min(buf.len, 3);
    var back: usize = 1;
    while (back <= max_back) : (back += 1) {
        const idx = buf.len - back;
        const b = buf[idx];
        if ((b & 0x80) == 0) {
            return buf.len; // ASCII terminates cleanly
        }
        if ((b & 0xC0) != 0x80) {
            // Found UTF-8 leading byte
            const expected: usize = if ((b & 0xE0) == 0xC0)
                2
            else if ((b & 0xF0) == 0xE0)
                3
            else if ((b & 0xF8) == 0xF0)
                4
            else
                1;
            if (back < expected) return idx;
            return buf.len;
        }
    }
    return buf.len;
}

pub fn appendJsonEscaped(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            else => {
                if (c < 0x20) {
                    var ubuf: [6]u8 = undefined;
                    const esc = std.fmt.bufPrint(&ubuf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(allocator, esc);
                } else {
                    try out.append(allocator, c);
                }
            },
        }
    }
}

fn appendFmt(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const tmp = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(tmp);
    try out.appendSlice(allocator, tmp);
}

fn sendAll(fd: std.c.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.send(fd, data.ptr + off, data.len - off, MSG_NOSIGNAL);
        if (n <= 0) return error.ClientDisconnected;
        off += @intCast(n);
    }
}

fn sendHttpJson(fd: std.c.fd_t, allocator: std.mem.Allocator, status: []const u8, body: []const u8) !void {
    const hdr = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 {s}\r\n" ++
            "Content-Type: application/json; charset=utf-8\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Headers: *\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Connection: close\r\n\r\n",
        .{ status, body.len },
    );
    defer allocator.free(hdr);
    try sendAll(fd, hdr);
    if (body.len > 0) {
        try sendAll(fd, body);
    }
}

fn sendSseHeaders(fd: std.c.fd_t) !void {
    const hdr =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream; charset=utf-8\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "X-Accel-Buffering: no\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Headers: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Connection: close\r\n\r\n";
    try sendAll(fd, hdr);
}

const HttpRequest = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
    raw: []u8,
    owned_body: ?[]u8 = null,

    fn deinit(self: *HttpRequest, allocator: std.mem.Allocator) void {
        if (self.owned_body) |ob| allocator.free(ob);
        allocator.free(self.raw);
    }
};

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Decodes an HTTP/1.1 Transfer-Encoding: chunked body.
/// Returns `null` if more bytes are needed to reach the terminating `0\r\n\r\n` chunk.
pub fn tryDecodeChunkedBody(allocator: std.mem.Allocator, data: []const u8) !?[]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var pos: usize = 0;
    while (true) {
        const crlf = std.mem.indexOfPos(u8, data, pos, "\r\n") orelse {
            out.deinit(allocator);
            return null;
        };
        const size_line = data[pos..crlf];
        const semi = std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len;
        const size_str = std.mem.trim(u8, size_line[0..semi], " \t");
        const chunk_len = std.fmt.parseInt(usize, size_str, 16) catch return error.InvalidChunkedBody;
        pos = crlf + 2;

        if (chunk_len == 0) {
            // Wait for trailing CRLF (or trailers + CRLF)
            if (pos + 2 <= data.len and std.mem.eql(u8, data[pos .. pos + 2], "\r\n")) {
                return try out.toOwnedSlice(allocator);
            }
            if (std.mem.indexOfPos(u8, data, pos, "\r\n\r\n") != null) {
                return try out.toOwnedSlice(allocator);
            }
            out.deinit(allocator);
            return null;
        }

        if (out.items.len + chunk_len > MAX_REQ_BODY) return error.BodyTooLarge;
        if (pos + chunk_len + 2 > data.len) {
            out.deinit(allocator);
            return null;
        }
        try out.appendSlice(allocator, data[pos .. pos + chunk_len]);
        if (!std.mem.eql(u8, data[pos + chunk_len .. pos + chunk_len + 2], "\r\n")) {
            return error.InvalidChunkedBody;
        }
        pos += chunk_len + 2;
    }
}

fn readHttpRequest(allocator: std.mem.Allocator, fd: std.c.fd_t) !?HttpRequest {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    var tmp: [4096]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: usize = 0;
    var is_chunked = false;
    var expect_continue = false;

    while (header_end == null) {
        const n = std.c.recv(fd, &tmp, tmp.len, 0);
        if (n == 0) {
            if (buf.items.len == 0) return null;
            break;
        }
        if (n < 0) {
            if (buf.items.len == 0) return null; // Idle preconnect/keep-alive timeout
            return error.RecvFailed;
        }
        try buf.appendSlice(allocator, tmp[0..@intCast(n)]);
        if (buf.items.len > 64 * 1024) return error.HeadersTooLarge;

        if (std.mem.indexOf(u8, buf.items, "\r\n\r\n")) |idx| {
            header_end = idx + 4;
            var lines = std.mem.splitSequence(u8, buf.items[0..idx], "\r\n");
            _ = lines.next(); // request line
            while (lines.next()) |line| {
                if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
                    const key = std.mem.trim(u8, line[0..colon], " \t");
                    const val = std.mem.trim(u8, line[colon + 1 ..], " \t");
                    if (std.ascii.eqlIgnoreCase(key, "content-length")) {
                        content_length = std.fmt.parseInt(usize, val, 10) catch return error.InvalidContentLength;
                        if (content_length > MAX_REQ_BODY) return error.BodyTooLarge;
                    } else if (std.ascii.eqlIgnoreCase(key, "transfer-encoding")) {
                        if (containsIgnoreCase(val, "chunked")) {
                            is_chunked = true;
                        }
                    } else if (std.ascii.eqlIgnoreCase(key, "expect")) {
                        if (containsIgnoreCase(val, "100-continue")) {
                            expect_continue = true;
                        }
                    }
                }
            }
        }
    }

    const hend = header_end orelse return error.MalformedRequest;
    if (expect_continue) {
        sendAll(fd, "HTTP/1.1 100 Continue\r\n\r\n") catch {};
    }

    var owned_body: ?[]u8 = null;
    errdefer if (owned_body) |ob| allocator.free(ob);

    if (is_chunked) {
        while (true) {
            if (try tryDecodeChunkedBody(allocator, buf.items[hend..])) |decoded| {
                owned_body = decoded;
                break;
            }
            const n = std.c.recv(fd, &tmp, tmp.len, 0);
            if (n <= 0) return error.IncompleteChunkedBody;
            try buf.appendSlice(allocator, tmp[0..@intCast(n)]);
            if (buf.items.len > MAX_REQ_BODY + 64 * 1024) return error.BodyTooLarge;
        }
    } else {
        const total_needed = hend + content_length;
        while (buf.items.len < total_needed) {
            const n = std.c.recv(fd, &tmp, tmp.len, 0);
            if (n <= 0) break;
            try buf.appendSlice(allocator, tmp[0..@intCast(n)]);
        }
    }

    const raw = try buf.toOwnedSlice(allocator);
    errdefer allocator.free(raw);

    const line_end = std.mem.indexOf(u8, raw[0..hend], "\r\n") orelse return error.MalformedRequest;
    const req_line = raw[0..line_end];
    var parts = std.mem.splitScalar(u8, req_line, ' ');
    const method = parts.next() orelse return error.MalformedRequest;
    const full_target = parts.next() orelse return error.MalformedRequest;
    var path = if (std.mem.indexOfScalar(u8, full_target, '?')) |q| full_target[0..q] else full_target;
    if (path.len > 1 and path[path.len - 1] == '/') {
        path = path[0 .. path.len - 1];
    }

    const body_slice = if (owned_body) |ob|
        ob
    else
        raw[hend..@min(raw.len, hend + content_length)];

    return .{
        .method = method,
        .path = path,
        .body = body_slice,
        .raw = raw,
        .owned_body = owned_body,
    };
}

fn jsonFloat(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn jsonUsize(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn parseMaxTokens(v: std.json.Value) ?usize {
    return switch (v) {
        .integer => |i| if (i < 0) std.math.maxInt(usize) else @intCast(i),
        else => null,
    };
}

fn parseGenParams(obj: std.json.ObjectMap) GenParams {
    var p: GenParams = .{};
    if (obj.get("max_completion_tokens")) |v| {
        if (parseMaxTokens(v)) |n| p.max_tokens = n;
    } else if (obj.get("max_tokens")) |v| {
        if (parseMaxTokens(v)) |n| p.max_tokens = n;
    } else if (obj.get("n_predict")) |v| {
        if (parseMaxTokens(v)) |n| p.max_tokens = n;
    }
    if (obj.get("temperature")) |v| {
        if (jsonFloat(v)) |f| p.temperature = f;
    }
    if (obj.get("top_k")) |v| {
        if (jsonUsize(v)) |k| p.top_k = @intCast(@min(k, 64));
    }
    if (obj.get("top_p")) |v| {
        if (jsonFloat(v)) |f| p.top_p = f;
    }
    if (obj.get("seed")) |v| {
        if (v == .integer) p.seed = @bitCast(v.integer);
    }
    if (obj.get("stream")) |v| {
        if (v == .bool) p.stream = v.bool;
    }
    if (obj.get("stop")) |v| {
        switch (v) {
            .string => |s| {
                if (s.len > 0) {
                    p.stops[0] = s;
                    p.n_stops = 1;
                }
            },
            .array => |arr| {
                for (arr.items) |item| {
                    if (p.n_stops >= MAX_STOPS) break;
                    if (item == .string and item.string.len > 0) {
                        p.stops[p.n_stops] = item.string;
                        p.n_stops += 1;
                    }
                }
            },
            else => {},
        }
    }
    return p;
}

pub const ReasoningEffort = enum {
    none,
    low,
    medium,
    xhigh,

    pub fn name(self: ReasoningEffort) []const u8 {
        return switch (self) {
            .none => "none",
            .low => "low",
            .medium => "medium",
            .xhigh => "xhigh",
        };
    }

    pub fn instructions(self: ReasoningEffort) []const u8 {
        return switch (self) {
            .xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            .low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            .medium, .none => "",
        };
    }
};

fn parseEffortString(s: []const u8) ?ReasoningEffort {
    if (std.ascii.eqlIgnoreCase(s, "none") or
        std.ascii.eqlIgnoreCase(s, "off") or
        std.ascii.eqlIgnoreCase(s, "disabled") or
        std.ascii.eqlIgnoreCase(s, "minimal"))
    {
        return .none;
    }
    if (std.ascii.eqlIgnoreCase(s, "low")) return .low;
    if (std.ascii.eqlIgnoreCase(s, "medium") or std.ascii.eqlIgnoreCase(s, "med")) return .medium;
    if (std.ascii.eqlIgnoreCase(s, "high") or
        std.ascii.eqlIgnoreCase(s, "xhigh") or
        std.ascii.eqlIgnoreCase(s, "max"))
    {
        return .xhigh;
    }
    return null;
}

pub fn parseReasoningEffort(obj: std.json.ObjectMap) ReasoningEffort {
    // 1. Explicit disable flags
    if (obj.get("enable_thinking")) |tv| {
        if (tv == .bool and !tv.bool) return .none;
    }
    if (obj.get("chat_template_kwargs")) |kw| {
        if (kw == .object) {
            if (kw.object.get("enable_thinking")) |tv| {
                if (tv == .bool and !tv.bool) return .none;
            }
        }
    }
    if (obj.get("thinking")) |th| {
        if (th == .bool and !th.bool) return .none;
        if (th == .object) {
            if (th.object.get("type")) |tt| {
                if (tt == .string and (std.ascii.eqlIgnoreCase(tt.string, "disabled") or std.ascii.eqlIgnoreCase(tt.string, "none") or std.ascii.eqlIgnoreCase(tt.string, "off"))) {
                    return .none;
                }
            }
            if (th.object.get("enabled")) |en| {
                if (en == .bool and !en.bool) return .none;
            }
        }
    }
    if (obj.get("reasoning")) |ro| {
        if (ro == .bool and !ro.bool) return .none;
        if (ro == .object) {
            if (ro.object.get("enabled")) |en| {
                if (en == .bool and !en.bool) return .none;
            }
        }
    }

    // 2. Explicit effort string: reasoning_effort / reasoning.effort / thinking.effort / chat_template_kwargs.reasoning_effort / effort
    if (obj.get("reasoning_effort")) |rv| {
        if (rv == .string) {
            if (parseEffortString(rv.string)) |eff| return eff;
        }
    }
    if (obj.get("reasoning")) |ro| {
        if (ro == .object) {
            if (ro.object.get("effort")) |rv| {
                if (rv == .string) {
                    if (parseEffortString(rv.string)) |eff| return eff;
                }
            }
        } else if (ro == .string) {
            if (parseEffortString(ro.string)) |eff| return eff;
        }
    }
    if (obj.get("thinking")) |th| {
        if (th == .object) {
            if (th.object.get("effort")) |rv| {
                if (rv == .string) {
                    if (parseEffortString(rv.string)) |eff| return eff;
                }
            }
            if (th.object.get("budget_tokens")) |bv| {
                const budget: i64 = switch (bv) {
                    .integer => |i| i,
                    .float => |f| @intFromFloat(f),
                    else => -1,
                };
                if (budget == 0) return .none;
                if (budget > 0) {
                    if (budget <= 2048) return .low;
                    if (budget <= 8192) return .medium;
                    return .xhigh;
                }
            }
        } else if (th == .string) {
            if (parseEffortString(th.string)) |eff| return eff;
        }
    }
    if (obj.get("chat_template_kwargs")) |kw| {
        if (kw == .object) {
            if (kw.object.get("reasoning_effort")) |rv| {
                if (rv == .string) {
                    if (parseEffortString(rv.string)) |eff| return eff;
                }
            }
        }
    }
    if (obj.get("effort")) |ev| {
        if (ev == .string) {
            if (parseEffortString(ev.string)) |eff| return eff;
        }
    }

    // 3. Default matches official Kintsugi-27B GGUF chat_template (reasoning_effort|default('xhigh'))
    return .xhigh;
}

fn appendMessageContent(
    out: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
    val_opt: ?std.json.Value,
) !void {
    const cv = val_opt orelse return;
    switch (cv) {
        .string => |s| try out.appendSlice(allocator, s),
        .array => |arr| {
            var wrote_part = false;
            for (arr.items) |part| {
                if (part == .string) {
                    if (wrote_part) try out.append(allocator, '\n');
                    try out.appendSlice(allocator, part.string);
                    wrote_part = true;
                } else if (part == .object) {
                    if (part.object.get("text")) |tv| {
                        if (tv == .string) {
                            if (wrote_part) try out.append(allocator, '\n');
                            try out.appendSlice(allocator, tv.string);
                            wrote_part = true;
                        }
                    }
                }
            }
        },
        else => {},
    }
}

fn appendJsonValue(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, val: std.json.Value) !void {
    const s = try std.json.Stringify.valueAlloc(allocator, val, .{});
    defer allocator.free(s);
    try out.appendSlice(allocator, s);
}

pub fn formatChatMl(
    allocator: std.mem.Allocator,
    messages: []const std.json.Value,
    effort: ReasoningEffort,
    tools_opt: ?[]const std.json.Value,
) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    const reasoning_instr = effort.instructions();
    const has_tools = if (tools_opt) |ts| ts.len > 0 else false;

    // 1. System block (matching Kintsugi-27B GGUF tokenizer.chat_template)
    var first_is_system = false;
    var sys_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer sys_buf.deinit(allocator);
    if (messages.len > 0 and messages[0] == .object) {
        if (messages[0].object.get("role")) |rv| {
            if (rv == .string and std.mem.eql(u8, rv.string, "system")) {
                first_is_system = true;
                try appendMessageContent(&sys_buf, allocator, messages[0].object.get("content"));
            }
        }
    }
    const sys_trimmed = std.mem.trim(u8, sys_buf.items, " \t\r\n");

    if (has_tools) {
        try out.appendSlice(allocator, "<|im_start|>system\n");
        if (reasoning_instr.len > 0) {
            try out.appendSlice(allocator, reasoning_instr);
            try out.appendSlice(allocator, "\n\n");
        }
        try out.appendSlice(allocator, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (tools_opt.?) |tool| {
            try out.append(allocator, '\n');
            try appendJsonValue(&out, allocator, tool);
        }
        try out.appendSlice(
            allocator,
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" ++
                "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" ++
                "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" ++
                "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" ++
                "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" ++
                "- Required parameters MUST be specified\n" ++
                "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" ++
                "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" ++
                "</IMPORTANT>",
        );
        if (sys_trimmed.len > 0) {
            try out.appendSlice(allocator, "\n\n");
            try out.appendSlice(allocator, sys_trimmed);
        }
        try out.appendSlice(allocator, "<|im_end|>\n");
    } else if (sys_trimmed.len > 0 or reasoning_instr.len > 0) {
        try out.appendSlice(allocator, "<|im_start|>system\n");
        if (reasoning_instr.len > 0) {
            try out.appendSlice(allocator, reasoning_instr);
            if (sys_trimmed.len > 0) try out.appendSlice(allocator, "\n\n");
        }
        if (sys_trimmed.len > 0) {
            try out.appendSlice(allocator, sys_trimmed);
        }
        try out.appendSlice(allocator, "<|im_end|>\n");
    }

    // 2. Conversation turns
    var msg_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer msg_buf.deinit(allocator);
    var in_tool_group = false;

    for (messages, 0..) |m, idx| {
        if (m != .object) continue;
        const role = if (m.object.get("role")) |rv| (if (rv == .string) rv.string else "user") else "user";
        if (idx == 0 and first_is_system) continue;

        if (std.mem.eql(u8, role, "tool")) {
            if (!in_tool_group) {
                try out.appendSlice(allocator, "<|im_start|>user");
                in_tool_group = true;
            }
            msg_buf.clearRetainingCapacity();
            try appendMessageContent(&msg_buf, allocator, m.object.get("content"));
            try out.appendSlice(allocator, "\n<tool_response>\n");
            try out.appendSlice(allocator, std.mem.trim(u8, msg_buf.items, " \t\r\n"));
            try out.appendSlice(allocator, "\n</tool_response>");
            continue;
        }

        if (in_tool_group) {
            try out.appendSlice(allocator, "<|im_end|>\n");
            in_tool_group = false;
        }

        msg_buf.clearRetainingCapacity();
        try appendMessageContent(&msg_buf, allocator, m.object.get("content"));
        const content_trimmed = std.mem.trim(u8, msg_buf.items, " \t\r\n");

        if (std.mem.eql(u8, role, "assistant")) {
            try out.appendSlice(allocator, "<|im_start|>assistant\n");
            const rc_raw = if (m.object.get("reasoning_content") orelse m.object.get("reasoning")) |rv|
                (if (rv == .string) rv.string else "")
            else
                "";
            const rc_trimmed = std.mem.trim(u8, rc_raw, " \t\r\n");

            if (rc_trimmed.len == 0 and std.mem.startsWith(u8, content_trimmed, "<think>")) {
                // Client already embedded <think>...</think> inline in content
                try out.appendSlice(allocator, content_trimmed);
            } else {
                try out.appendSlice(allocator, "<think>\n");
                try out.appendSlice(allocator, rc_trimmed);
                try out.appendSlice(allocator, "\n</think>\n\n");
                try out.appendSlice(allocator, content_trimmed);
            }

            if (m.object.get("tool_calls")) |tcv| {
                if (tcv == .array) {
                    for (tcv.array.items, 0..) |tc_item, tc_i| {
                        if (tc_item != .object) continue;
                        const fn_obj = if (tc_item.object.get("function")) |fv|
                            (if (fv == .object) fv.object else tc_item.object)
                        else
                            tc_item.object;
                        const fn_name = if (fn_obj.get("name")) |nv| (if (nv == .string) nv.string else "") else "";
                        if (tc_i == 0) {
                            if (content_trimmed.len > 0) try out.appendSlice(allocator, "\n\n");
                        } else {
                            try out.append(allocator, '\n');
                        }
                        try out.appendSlice(allocator, "<tool_call>\n<function=");
                        try out.appendSlice(allocator, fn_name);
                        try out.appendSlice(allocator, ">\n");
                        if (fn_obj.get("arguments")) |av| {
                            var parsed_args: ?std.json.Parsed(std.json.Value) = null;
                            defer if (parsed_args) |*pa| pa.deinit();
                            const args_map_opt: ?std.json.ObjectMap = switch (av) {
                                .object => |o| o,
                                .string => |s| blk: {
                                    parsed_args = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch null;
                                    if (parsed_args) |pa| {
                                        if (pa.value == .object) break :blk pa.value.object;
                                    }
                                    break :blk null;
                                },
                                else => null,
                            };
                            if (args_map_opt) |amap| {
                                var it = amap.iterator();
                                while (it.next()) |entry| {
                                    try out.appendSlice(allocator, "<parameter=");
                                    try out.appendSlice(allocator, entry.key_ptr.*);
                                    try out.appendSlice(allocator, ">\n");
                                    if (entry.value_ptr.* == .string) {
                                        try out.appendSlice(allocator, entry.value_ptr.string);
                                    } else {
                                        try appendJsonValue(&out, allocator, entry.value_ptr.*);
                                    }
                                    try out.appendSlice(allocator, "\n</parameter>\n");
                                }
                            }
                        }
                        try out.appendSlice(allocator, "</function>\n</tool_call>");
                    }
                }
            }
            try out.appendSlice(allocator, "<|im_end|>\n");
        } else {
            try out.appendSlice(allocator, "<|im_start|>");
            try out.appendSlice(allocator, role);
            try out.append(allocator, '\n');
            try out.appendSlice(allocator, content_trimmed);
            try out.appendSlice(allocator, "<|im_end|>\n");
        }
    }
    if (in_tool_group) {
        try out.appendSlice(allocator, "<|im_end|>\n");
    }

    // 3. Generation prompt (exact GGUF chat_template!)
    try out.appendSlice(allocator, "<|im_start|>assistant\n");
    if (effort == .none) {
        try out.appendSlice(allocator, "<think>\n\n</think>\n\n");
    } else {
        try out.appendSlice(allocator, "<think>\n");
    }
    return try out.toOwnedSlice(allocator);
}

fn findStopInText(text: []const u8, stops: []const []const u8) ?usize {
    var earliest: ?usize = null;
    for (stops) |st| {
        if (std.mem.indexOf(u8, text, st)) |idx| {
            if (earliest == null or idx < earliest.?) earliest = idx;
        }
    }
    return earliest;
}

fn sendSseChunk(
    allocator: std.mem.Allocator,
    fd: std.c.fd_t,
    mode: Mode,
    req_id: []const u8,
    created: i64,
    text_delta: ?[]const u8,
    is_reasoning: bool,
    send_role: bool,
    finish_reason: ?[]const u8,
    stats_opt: ?GenStats,
) !void {
    var j: std.ArrayListUnmanaged(u8) = .empty;
    defer j.deinit(allocator);

    const obj_name = if (mode == .chat) "chat.completion.chunk" else "text_completion";
    try appendFmt(&j, allocator, "data: {{\"id\":\"{s}\",\"object\":\"{s}\",\"created\":{d},\"model\":\"{s}\",\"choices\":[{{\"index\":0,", .{
        req_id, obj_name, created, MODEL_ID,
    });

    if (mode == .chat) {
        try j.appendSlice(allocator, "\"delta\":{");
        var need_comma = false;
        if (send_role) {
            try j.appendSlice(allocator, "\"role\":\"assistant\"");
            need_comma = true;
        }
        if (text_delta) |td| {
            if (need_comma) try j.append(allocator, ',');
            if (is_reasoning) {
                try j.appendSlice(allocator, "\"reasoning_content\":\"");
            } else {
                try j.appendSlice(allocator, "\"content\":\"");
            }
            try appendJsonEscaped(&j, allocator, td);
            try j.append(allocator, '"');
        }
        try j.append(allocator, '}');
    } else {
        try j.appendSlice(allocator, "\"text\":\"");
        if (text_delta) |td| {
            try appendJsonEscaped(&j, allocator, td);
        }
        try j.append(allocator, '"');
    }

    if (finish_reason) |fr| {
        try appendFmt(&j, allocator, ",\"finish_reason\":\"{s}\"}}]", .{fr});
    } else {
        try j.appendSlice(allocator, ",\"finish_reason\":null}]");
    }

    if (stats_opt) |st| {
        const pps = if (st.prompt_ms > 0.0) @as(f64, @floatFromInt(st.prompt_tokens)) * 1000.0 / st.prompt_ms else 0.0;
        const dps = if (st.decode_ms > 0.0) @as(f64, @floatFromInt(st.completion_tokens)) * 1000.0 / st.decode_ms else 0.0;
        try appendFmt(
            &j,
            allocator,
            ",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}" ++
                ",\"timings\":{{\"prompt_n\":{d},\"prompt_ms\":{d:.2},\"prompt_per_second\":{d:.2}," ++
                "\"predicted_n\":{d},\"predicted_ms\":{d:.2},\"predicted_per_second\":{d:.2}," ++
                "\"spec_steps\":{d},\"draft_n\":{d},\"draft_n_accepted\":{d}}}",
            .{
                st.prompt_tokens,
                st.completion_tokens,
                st.prompt_tokens + st.completion_tokens,
                st.prompt_tokens,
                st.prompt_ms,
                pps,
                st.completion_tokens,
                st.decode_ms,
                dps,
                st.spec_steps,
                st.draft_n,
                st.draft_n_accepted,
            },
        );
    }

    try j.appendSlice(allocator, "}\n\n");
    try sendAll(fd, j.items);
}

fn sendToolCallSseStart(
    allocator: std.mem.Allocator,
    fd: std.c.fd_t,
    req_id: []const u8,
    created: i64,
    call_index: usize,
    call_id: []const u8,
    fn_name: []const u8,
) !void {
    var j: std.ArrayListUnmanaged(u8) = .empty;
    defer j.deinit(allocator);
    try appendFmt(
        &j,
        allocator,
        "data: {{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{d},\"id\":\"",
        .{ req_id, created, MODEL_ID, call_index },
    );
    try appendJsonEscaped(&j, allocator, call_id);
    try j.appendSlice(allocator, "\",\"type\":\"function\",\"function\":{\"name\":\"");
    try appendJsonEscaped(&j, allocator, fn_name);
    try j.appendSlice(allocator, "\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
    try sendAll(fd, j.items);
}

fn sendToolCallSseArgs(
    allocator: std.mem.Allocator,
    fd: std.c.fd_t,
    req_id: []const u8,
    created: i64,
    call_index: usize,
    args_delta: []const u8,
) !void {
    if (args_delta.len == 0) return;
    var j: std.ArrayListUnmanaged(u8) = .empty;
    defer j.deinit(allocator);
    try appendFmt(
        &j,
        allocator,
        "data: {{\"id\":\"{s}\",\"object\":\"chat.completion.chunk\",\"created\":{d},\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":{d},\"function\":{{\"arguments\":\"",
        .{ req_id, created, MODEL_ID, call_index },
    );
    try appendJsonEscaped(&j, allocator, args_delta);
    try j.appendSlice(allocator, "\"}}]},\"finish_reason\":null}]}\n\n");
    try sendAll(fd, j.items);
}

const ToolCallRecord = struct {
    id: []u8,
    name: []u8,
    arguments: std.ArrayListUnmanaged(u8) = .empty,

    fn deinit(self: *ToolCallRecord, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        self.arguments.deinit(allocator);
    }
};

fn partialTagSuffixLen(haystack: []const u8, needle: []const u8) usize {
    const max_k = @min(haystack.len, needle.len - 1);
    var k: usize = max_k;
    while (k > 0) : (k -= 1) {
        if (std.mem.endsWith(u8, haystack, needle[0..k])) return k;
    }
    return 0;
}

fn isAllAsciiWs(s: []const u8) bool {
    for (s) |c| {
        if (c != ' ' and c != '\t' and c != '\r' and c != '\n') return false;
    }
    return true;
}

fn stripOneTrailingNl(s: []const u8) []const u8 {
    if (std.mem.endsWith(u8, s, "\r\n")) return s[0 .. s.len - 2];
    if (std.mem.endsWith(u8, s, "\n")) return s[0 .. s.len - 1];
    return s;
}

fn lookupToolParamType(
    tools_opt: ?[]const std.json.Value,
    fn_name: []const u8,
    param_name: []const u8,
) ?[]const u8 {
    const tools = tools_opt orelse return null;
    for (tools) |tool| {
        if (tool != .object) continue;
        const fn_obj = if (tool.object.get("function")) |fv|
            (if (fv == .object) fv.object else tool.object)
        else
            tool.object;
        const name_val = fn_obj.get("name") orelse continue;
        if (name_val != .string or !std.mem.eql(u8, name_val.string, fn_name)) continue;
        const params_val = fn_obj.get("parameters") orelse return null;
        if (params_val != .object) return null;
        const props_val = params_val.object.get("properties") orelse return null;
        if (props_val != .object) return null;
        const prop_val = props_val.object.get(param_name) orelse return null;
        if (prop_val != .object) return null;
        const type_val = prop_val.object.get("type") orelse return null;
        if (type_val == .string) return type_val.string;
        return null;
    }
    return null;
}

/// Streaming parser for Qwen3 / Kintsugi-27B XML tool calls:
///   `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>`
/// Also supports JSON fallback `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
/// String-typed parameters stream live into `delta.tool_calls[].function.arguments` with a 14-byte
/// tail holdback so large HTML/code file writes never stall or trigger client read timeouts.
const ToolCallParser = struct {
    const Phase = enum {
        outside,
        await_call_body,
        in_json_call,
        await_param,
        in_string_param,
        in_typed_param,
        await_call_close,
    };

    allocator: std.mem.Allocator,
    req_id: []const u8,
    tools_opt: ?[]const std.json.Value,
    phase: Phase = .outside,
    buf: std.ArrayListUnmanaged(u8) = .empty,
    calls: std.ArrayListUnmanaged(ToolCallRecord) = .empty,
    call_opened_brace: bool = false,
    strip_leading_nl: bool = false,
    cur_param_type: ?[]const u8 = null,

    fn init(allocator: std.mem.Allocator, req_id: []const u8, tools_opt: ?[]const std.json.Value) ToolCallParser {
        return .{
            .allocator = allocator,
            .req_id = req_id,
            .tools_opt = tools_opt,
        };
    }

    fn deinit(self: *ToolCallParser) void {
        self.buf.deinit(self.allocator);
        for (self.calls.items) |*c| c.deinit(self.allocator);
        self.calls.deinit(self.allocator);
    }

    fn consumeFront(self: *ToolCallParser, n: usize) void {
        if (n == 0) return;
        const rem = self.buf.items.len - n;
        if (rem > 0) {
            std.mem.copyForwards(u8, self.buf.items[0..rem], self.buf.items[n..]);
        }
        self.buf.shrinkRetainingCapacity(rem);
    }

    fn trimFrontWs(self: *ToolCallParser) void {
        var i: usize = 0;
        while (i < self.buf.items.len) : (i += 1) {
            const c = self.buf.items[i];
            if (c != ' ' and c != '\t' and c != '\r' and c != '\n') break;
        }
        self.consumeFront(i);
    }

    /// Strip at most one leading `\r\n` or `\n`. Returns false if we need more bytes (`\r` at EOF).
    fn tryStripOneLeadingNl(self: *ToolCallParser) bool {
        if (!self.strip_leading_nl) return true;
        if (self.buf.items.len == 0) return false;
        if (self.buf.items[0] == '\r') {
            if (self.buf.items.len == 1) return false;
            if (self.buf.items[1] == '\n') {
                self.consumeFront(2);
            } else {
                self.consumeFront(1);
            }
        } else if (self.buf.items[0] == '\n') {
            self.consumeFront(1);
        }
        self.strip_leading_nl = false;
        return true;
    }

    fn beginCall(self: *ToolCallParser, fn_name: []const u8, sink: anytype) !void {
        const call_idx = self.calls.items.len;
        const id = try std.fmt.allocPrint(self.allocator, "call_{s}_{d}", .{ self.req_id, call_idx });
        errdefer self.allocator.free(id);
        const name_dup = try self.allocator.dupe(u8, fn_name);
        errdefer self.allocator.free(name_dup);

        try self.calls.append(self.allocator, .{
            .id = id,
            .name = name_dup,
        });
        self.call_opened_brace = false;
        try sink.onToolCallStart(call_idx, id, name_dup);
    }

    fn emitArgsDelta(self: *ToolCallParser, delta: []const u8, sink: anytype) !void {
        if (delta.len == 0 or self.calls.items.len == 0) return;
        const call_idx = self.calls.items.len - 1;
        try self.calls.items[call_idx].arguments.appendSlice(self.allocator, delta);
        try sink.onToolCallArgsDelta(call_idx, delta);
    }

    fn emitJsonEscapedValueChunk(self: *ToolCallParser, raw_chunk: []const u8, close_quote: bool, sink: anytype) !void {
        var tmp: std.ArrayListUnmanaged(u8) = .empty;
        defer tmp.deinit(self.allocator);
        if (raw_chunk.len > 0) {
            try appendJsonEscaped(&tmp, self.allocator, raw_chunk);
        }
        if (close_quote) try tmp.append(self.allocator, '"');
        try self.emitArgsDelta(tmp.items, sink);
    }

    fn emitTypedParamValue(self: *ToolCallParser, raw_val: []const u8, sink: anytype) !void {
        const trimmed = std.mem.trim(u8, raw_val, " \t\r\n");
        const is_bool_schema = if (self.cur_param_type) |t| std.mem.eql(u8, t, "boolean") else false;
        const is_string_schema = if (self.cur_param_type) |t| std.mem.eql(u8, t, "string") else false;

        if (!is_string_schema and trimmed.len > 0) {
            if (is_bool_schema) {
                if (std.ascii.eqlIgnoreCase(trimmed, "true")) {
                    try self.emitArgsDelta("true", sink);
                    return;
                }
                if (std.ascii.eqlIgnoreCase(trimmed, "false")) {
                    try self.emitArgsDelta("false", sink);
                    return;
                }
            }
            if (std.json.parseFromSlice(std.json.Value, self.allocator, trimmed, .{})) |parsed| {
                defer parsed.deinit();
                if (parsed.value != .string) {
                    try self.emitArgsDelta(trimmed, sink);
                    return;
                }
            } else |_| {}
        }

        // Default / string fallback: JSON-encoded string literal
        var tmp: std.ArrayListUnmanaged(u8) = .empty;
        defer tmp.deinit(self.allocator);
        try tmp.append(self.allocator, '"');
        try appendJsonEscaped(&tmp, self.allocator, raw_val);
        try tmp.append(self.allocator, '"');
        try self.emitArgsDelta(tmp.items, sink);
    }

    fn emitJsonFallbackCall(self: *ToolCallParser, raw_json: []const u8, sink: anytype) !bool {
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, raw_json, .{}) catch return false;
        defer parsed.deinit();
        if (parsed.value != .object) return false;
        const name_val = parsed.value.object.get("name") orelse return false;
        if (name_val != .string or name_val.string.len == 0) return false;

        try self.beginCall(name_val.string, sink);
        if (parsed.value.object.get("arguments")) |av| {
            switch (av) {
                .string => |s| try self.emitArgsDelta(if (s.len > 0) s else "{}", sink),
                else => {
                    const s = try std.json.Stringify.valueAlloc(self.allocator, av, .{});
                    defer self.allocator.free(s);
                    try self.emitArgsDelta(s, sink);
                },
            }
        } else {
            try self.emitArgsDelta("{}", sink);
        }
        return true;
    }

    fn feed(self: *ToolCallParser, piece: []const u8, sink: anytype) !void {
        if (piece.len > 0) {
            try self.buf.appendSlice(self.allocator, piece);
        }

        while (self.buf.items.len > 0) {
            switch (self.phase) {
                .outside => {
                    if (std.mem.indexOf(u8, self.buf.items, "<tool_call>")) |idx| {
                        const before = self.buf.items[0..idx];
                        if (before.len > 0 and !(self.calls.items.len > 0 and isAllAsciiWs(before))) {
                            try sink.onContent(before);
                        }
                        self.consumeFront(idx + "<tool_call>".len);
                        self.phase = .await_call_body;
                        continue;
                    }
                    // Suppress pure inter-tool-call whitespace while waiting to see if another <tool_call> follows
                    if (self.calls.items.len > 0 and isAllAsciiWs(self.buf.items)) {
                        break;
                    }
                    const hold = partialTagSuffixLen(self.buf.items, "<tool_call>");
                    const safe_len = self.buf.items.len - hold;
                    if (self.calls.items.len > 0 and isAllAsciiWs(self.buf.items[0..safe_len])) {
                        break;
                    }
                    const ready = utf8ReadyPrefixLen(self.buf.items[0..safe_len]);
                    if (ready > 0) {
                        try sink.onContent(self.buf.items[0..ready]);
                        self.consumeFront(ready);
                    }
                    break;
                },

                .await_call_body => {
                    self.trimFrontWs();
                    if (self.buf.items.len == 0) break;

                    if (self.buf.items[0] == '{') {
                        self.phase = .in_json_call;
                        continue;
                    }
                    if (std.mem.startsWith(u8, self.buf.items, "<function=")) {
                        const gt_idx = std.mem.indexOfScalarPos(u8, self.buf.items, "<function=".len, '>') orelse {
                            if (self.buf.items.len > 256) {
                                // Malformed huge header; emit raw text and bail out
                                try sink.onContent("<tool_call>");
                                try sink.onContent(self.buf.items);
                                self.buf.clearRetainingCapacity();
                                self.phase = .outside;
                            }
                            break;
                        };
                        const fn_name = std.mem.trim(u8, self.buf.items["<function=".len .. gt_idx], " \t\r\n");
                        try self.beginCall(fn_name, sink);
                        self.consumeFront(gt_idx + 1);
                        self.phase = .await_param;
                        continue;
                    }
                    if (std.mem.startsWith(u8, self.buf.items, "</tool_call>")) {
                        self.consumeFront("</tool_call>".len);
                        self.phase = .outside;
                        continue;
                    }
                    if (std.mem.startsWith(u8, "<function=", self.buf.items) or
                        std.mem.startsWith(u8, "</tool_call>", self.buf.items))
                    {
                        break;
                    }
                    // Unexpected body inside <tool_call>: fall back to content
                    try sink.onContent("<tool_call>");
                    self.phase = .outside;
                    continue;
                },

                .in_json_call => {
                    const end_idx = std.mem.indexOf(u8, self.buf.items, "</tool_call>") orelse break;
                    const raw_json = std.mem.trim(u8, self.buf.items[0..end_idx], " \t\r\n");
                    const ok = try self.emitJsonFallbackCall(raw_json, sink);
                    if (!ok) {
                        try sink.onContent("<tool_call>");
                        try sink.onContent(self.buf.items[0 .. end_idx + "</tool_call>".len]);
                    }
                    self.consumeFront(end_idx + "</tool_call>".len);
                    self.phase = .outside;
                    continue;
                },

                .await_param => {
                    self.trimFrontWs();
                    if (self.buf.items.len == 0) break;

                    if (std.mem.startsWith(u8, self.buf.items, "<parameter=")) {
                        const gt_idx = std.mem.indexOfScalarPos(u8, self.buf.items, "<parameter=".len, '>') orelse break;
                        const param_name = std.mem.trim(u8, self.buf.items["<parameter=".len .. gt_idx], " \t\r\n");
                        const cur_fn_name = self.calls.items[self.calls.items.len - 1].name;
                        const ptype = lookupToolParamType(self.tools_opt, cur_fn_name, param_name);
                        const is_str = if (ptype) |t| std.mem.eql(u8, t, "string") else false;

                        var key_prefix: std.ArrayListUnmanaged(u8) = .empty;
                        defer key_prefix.deinit(self.allocator);
                        if (!self.call_opened_brace) {
                            try key_prefix.append(self.allocator, '{');
                            self.call_opened_brace = true;
                        } else {
                            try key_prefix.append(self.allocator, ',');
                        }
                        try key_prefix.append(self.allocator, '"');
                        try appendJsonEscaped(&key_prefix, self.allocator, param_name);
                        try key_prefix.appendSlice(self.allocator, "\":");
                        if (is_str) try key_prefix.append(self.allocator, '"');

                        try self.emitArgsDelta(key_prefix.items, sink);
                        self.consumeFront(gt_idx + 1);
                        self.cur_param_type = ptype;
                        self.strip_leading_nl = true;
                        self.phase = if (is_str) .in_string_param else .in_typed_param;
                        continue;
                    }
                    if (std.mem.startsWith(u8, self.buf.items, "</function>")) {
                        self.consumeFront("</function>".len);
                        try self.emitArgsDelta(if (self.call_opened_brace) "}" else "{}", sink);
                        self.phase = .await_call_close;
                        continue;
                    }
                    if (std.mem.startsWith(u8, self.buf.items, "</tool_call>")) {
                        self.consumeFront("</tool_call>".len);
                        try self.emitArgsDelta(if (self.call_opened_brace) "}" else "{}", sink);
                        self.phase = .outside;
                        continue;
                    }
                    if (std.mem.startsWith(u8, "<parameter=", self.buf.items) or
                        std.mem.startsWith(u8, "</function>", self.buf.items) or
                        std.mem.startsWith(u8, "</tool_call>", self.buf.items))
                    {
                        break;
                    }
                    // Defensive recovery: skip unexpected non-tag byte until next '<'
                    if (std.mem.indexOfScalar(u8, self.buf.items, '<')) |next_lt| {
                        self.consumeFront(if (next_lt == 0) 1 else next_lt);
                        continue;
                    }
                    self.buf.clearRetainingCapacity();
                    break;
                },

                .in_string_param => {
                    if (!self.tryStripOneLeadingNl()) break;
                    if (std.mem.indexOf(u8, self.buf.items, "</parameter>")) |end_idx| {
                        const val_chunk = stripOneTrailingNl(self.buf.items[0..end_idx]);
                        try self.emitJsonEscapedValueChunk(val_chunk, true, sink);
                        self.consumeFront(end_idx + "</parameter>".len);
                        self.phase = .await_param;
                        continue;
                    }
                    // Hold back enough bytes to never prematurely flush "\r\n</parameter>"
                    const max_hold: usize = "</parameter>".len + 2;
                    if (self.buf.items.len > max_hold) {
                        const safe_len = self.buf.items.len - max_hold;
                        const ready = utf8ReadyPrefixLen(self.buf.items[0..safe_len]);
                        if (ready > 0) {
                            try self.emitJsonEscapedValueChunk(self.buf.items[0..ready], false, sink);
                            self.consumeFront(ready);
                        }
                    }
                    break;
                },

                .in_typed_param => {
                    if (!self.tryStripOneLeadingNl()) break;
                    if (std.mem.indexOf(u8, self.buf.items, "</parameter>")) |end_idx| {
                        const val_chunk = stripOneTrailingNl(self.buf.items[0..end_idx]);
                        try self.emitTypedParamValue(val_chunk, sink);
                        self.consumeFront(end_idx + "</parameter>".len);
                        self.phase = .await_param;
                        continue;
                    }
                    break;
                },

                .await_call_close => {
                    self.trimFrontWs();
                    if (self.buf.items.len == 0) break;
                    if (std.mem.startsWith(u8, self.buf.items, "</tool_call>")) {
                        self.consumeFront("</tool_call>".len);
                        self.phase = .outside;
                        continue;
                    }
                    if (std.mem.startsWith(u8, "</tool_call>", self.buf.items)) {
                        break;
                    }
                    if (std.mem.startsWith(u8, self.buf.items, "<function=")) {
                        self.phase = .await_call_body;
                        continue;
                    }
                    self.phase = .outside;
                    continue;
                },
            }
        }
    }

    fn finish(self: *ToolCallParser, sink: anytype) !void {
        switch (self.phase) {
            .outside => {
                if (self.buf.items.len > 0 and !(self.calls.items.len > 0 and isAllAsciiWs(self.buf.items))) {
                    try sink.onContent(self.buf.items);
                }
            },
            .in_string_param => {
                const rem = stripOneTrailingNl(self.buf.items);
                try self.emitJsonEscapedValueChunk(rem, true, sink);
                try self.emitArgsDelta("}", sink);
            },
            .in_typed_param => {
                const rem = stripOneTrailingNl(self.buf.items);
                try self.emitTypedParamValue(rem, sink);
                try self.emitArgsDelta("}", sink);
            },
            .await_param => {
                try self.emitArgsDelta(if (self.call_opened_brace) "}" else "{}", sink);
            },
            .in_json_call => {
                const raw_json = std.mem.trim(u8, self.buf.items, " \t\r\n");
                if (!(try self.emitJsonFallbackCall(raw_json, sink))) {
                    try sink.onContent("<tool_call>");
                    try sink.onContent(self.buf.items);
                }
            },
            .await_call_body => {
                if (self.buf.items.len > 0 and !isAllAsciiWs(self.buf.items)) {
                    try sink.onContent("<tool_call>");
                    try sink.onContent(self.buf.items);
                }
            },
            .await_call_close => {},
        }
        self.buf.clearRetainingCapacity();
        self.phase = .outside;
    }
};

const MAX_CHECKPOINTS: usize = 4;

const PrefixCheckpoint = struct {
    valid: bool = false,
    pinned: bool = false,
    pos: u32 = 0,
    last_used: u64 = 0,
    gdn_state: ?[]u8 = null,
    gdn_conv: ?[]u8 = null,
    dflash_kv: ?[]u8 = null,

    fn deinit(self: *PrefixCheckpoint, allocator: std.mem.Allocator) void {
        if (self.gdn_state) |s| allocator.free(s);
        if (self.gdn_conv) |c| allocator.free(c);
        if (self.dflash_kv) |d| allocator.free(d);
        self.* = .{};
    }
};

fn longestCommonPrefix(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

const ServerState = struct {
    allocator: std.mem.Allocator,
    tok: *tokenizer_mod.Tokenizer,
    eng: *engine_mod.Engine,
    dflash: ?*dflash_mod.DFlashEngine,
    req_counter: u64 = 0,
    active_tokens: std.ArrayListUnmanaged(u32) = .empty,
    checkpoints: [MAX_CHECKPOINTS]PrefixCheckpoint = .{ .{}, .{}, .{}, .{} },
    ckpt_clock: u64 = 0,

    fn deinit(self: *ServerState) void {
        for (&self.checkpoints) |*c| c.deinit(self.allocator);
        self.active_tokens.deinit(self.allocator);
    }

    fn nextReqId(self: *ServerState, prefix: []const u8, buf: []u8) []const u8 {
        self.req_counter += 1;
        return std.fmt.bufPrint(buf, "{s}-kintsugi-{d}", .{ prefix, self.req_counter }) catch "kintsugi-0";
    }

    fn invalidateCheckpointsAfter(self: *ServerState, keep_pos: u32) void {
        for (&self.checkpoints) |*c| {
            if (c.valid and c.pos > keep_pos) {
                c.valid = false;
                c.pinned = false;
            }
        }
    }

    fn saveCheckpoint(self: *ServerState, pos: u32, pinned: bool) !void {
        if (pos == 0) return;
        self.ckpt_clock += 1;

        // Reuse existing valid slot at same pos if present
        for (&self.checkpoints) |*c| {
            if (c.valid and c.pos == pos) {
                c.pinned = c.pinned or pinned;
                c.last_used = self.ckpt_clock;
                return;
            }
        }

        // Pick an invalid slot first; otherwise evict LRU non-pinned slot (or oldest overall)
        var target_idx: usize = 0;
        var found_free = false;
        for (self.checkpoints, 0..) |c, idx| {
            if (!c.valid) {
                target_idx = idx;
                found_free = true;
                break;
            }
        }
        if (!found_free) {
            var best_lu: u64 = std.math.maxInt(u64);
            var found_unpinned = false;
            for (self.checkpoints, 0..) |c, idx| {
                if (!c.pinned and c.last_used < best_lu) {
                    best_lu = c.last_used;
                    target_idx = idx;
                    found_unpinned = true;
                }
            }
            if (!found_unpinned) {
                best_lu = std.math.maxInt(u64);
                for (self.checkpoints, 0..) |c, idx| {
                    if (c.last_used < best_lu) {
                        best_lu = c.last_used;
                        target_idx = idx;
                    }
                }
            }
        }

        var slot = &self.checkpoints[target_idx];
        if (slot.gdn_state == null) {
            slot.gdn_state = try self.allocator.alloc(u8, engine_mod.GDN_STATE_BYTES);
        }
        if (slot.gdn_conv == null) {
            slot.gdn_conv = try self.allocator.alloc(u8, engine_mod.GDN_CONV_BYTES);
        }
        if (self.dflash != null and slot.dflash_kv == null) {
            slot.dflash_kv = try self.allocator.alloc(u8, dflash_mod.KV_RING_BYTES);
        }

        try self.eng.saveGdnStateToHost(slot.gdn_state.?, slot.gdn_conv.?);
        if (self.dflash) |df| {
            try df.saveKvRingToHost(slot.dflash_kv.?);
        }
        slot.pos = pos;
        slot.valid = true;
        slot.pinned = pinned;
        slot.last_used = self.ckpt_clock;
    }

    fn prefillSegment(self: *ServerState, tokens: []const u32, need_logits: bool) !engine_mod.StepOutput {
        errdefer {
            self.active_tokens.clearRetainingCapacity();
            self.invalidateCheckpointsAfter(0);
        }
        const res = if (self.dflash) |df|
            try df.prefillRange(self.eng, tokens, need_logits)
        else
            try self.eng.prefillRange(tokens, need_logits);
        try self.active_tokens.appendSlice(self.allocator, tokens);
        return res;
    }

    /// Detect the token length of the first ChatML message (`...<|im_end|>\n`, e.g. system prompt + tools)
    /// when it is long enough to be worth pinning as a reusable prefix checkpoint.
    fn detectSystemBoundary(self: *ServerState, prompt: []const u8, prompt_toks: []const u32) ?usize {
        const marker = "<|im_end|>\n";
        const idx = std.mem.indexOf(u8, prompt, marker) orelse return null;
        const byte_end = idx + marker.len;
        if (byte_end < 512 or byte_end >= prompt.len) return null;
        const sys_toks = self.tok.encode(self.allocator, prompt[0..byte_end]) catch return null;
        defer self.allocator.free(sys_toks);
        if (sys_toks.len < 256 or sys_toks.len >= prompt_toks.len) return null;
        if (!std.mem.eql(u32, sys_toks, prompt_toks[0..sys_toks.len])) return null;
        return sys_toks.len;
    }

    /// If the ChatML prompt ends with `<think>\n\n</think>\n\n` (effort=none) or `<think>\n` (thinking enabled),
    /// return the token length right BEFORE `<think>` (at `<|im_start|>assistant\n`).
    /// On the next turn, clients that strip historical `reasoning_content` send `<think>\n\n</think>\n\n`
    /// where `\n\n` merges into a single BPE token (differing from `<think>\n` at the last token).
    /// Checkpointing right before `<think>` guarantees multi-turn cache hits whether clients keep or strip reasoning!
    fn detectTurnBoundary(self: *ServerState, prompt: []const u8, prompt_toks: []const u32) ?usize {
        const suffix: []const u8 = if (std.mem.endsWith(u8, prompt, "<think>\n\n</think>\n\n"))
            "<think>\n\n</think>\n\n"
        else if (std.mem.endsWith(u8, prompt, "<think>\n"))
            "<think>\n"
        else
            return null;
        const prefix_bytes = prompt.len - suffix.len;
        if (prefix_bytes == 0) return null;
        const pre_toks = self.tok.encode(self.allocator, prompt[0..prefix_bytes]) catch return null;
        defer self.allocator.free(pre_toks);
        if (pre_toks.len < 64 or pre_toks.len >= prompt_toks.len) return null;
        if (!std.mem.eql(u32, pre_toks, prompt_toks[0..pre_toks.len])) return null;
        return pre_toks.len;
    }

    fn runGeneration(
        self: *ServerState,
        fd: std.c.fd_t,
        mode: Mode,
        prompt: []const u8,
        params_in: GenParams,
        effort: ReasoningEffort,
        tools_opt: ?[]const std.json.Value,
    ) !void {
        const prompt_toks = try self.tok.encode(self.allocator, prompt);
        defer self.allocator.free(prompt_toks);

        if (prompt_toks.len == 0) {
            try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"prompt must not be empty\"}}");
            return;
        }
        const kv_reserve: usize = engine_mod.MAX_M;
        if (prompt_toks.len + kv_reserve >= self.eng.max_kv_cells) {
            std.debug.print(
                "[kintsugi-zig] 400 prompt too long: prompt={d}, kv_pool={d}\n",
                .{ prompt_toks.len, self.eng.max_kv_cells },
            );
            const msg = try std.fmt.allocPrint(
                self.allocator,
                "{{\"error\":{{\"message\":\"prompt ({d} tokens) exceeds VRAM KV pool ({d} cells)\"}}}}",
                .{ prompt_toks.len, self.eng.max_kv_cells },
            );
            defer self.allocator.free(msg);
            try sendHttpJson(fd, self.allocator, "400 Bad Request", msg);
            return;
        }

        var params = params_in;
        const max_avail: usize = self.eng.max_kv_cells - prompt_toks.len - kv_reserve;
        if (params.max_tokens > max_avail) {
            std.debug.print(
                "[kintsugi-zig] clamping max_tokens {d} -> {d} (prompt={d}, kv_pool={d})\n",
                .{ params.max_tokens, max_avail, prompt_toks.len, self.eng.max_kv_cells },
            );
            params.max_tokens = max_avail;
        }

        var id_buf: [64]u8 = undefined;
        const req_id = self.nextReqId(if (mode == .chat) "chatcmpl" else "cmpl", &id_buf);
        const created = unixSeconds();

        if (params.stream) {
            try sendSseHeaders(fd);
            if (mode == .chat) {
                try sendSseChunk(self.allocator, fd, mode, req_id, created, null, false, true, null, null);
            }
        }

        // Prefix cache lookup: find deepest checkpoint <= min(lcp, prompt_toks.len - 1)
        const tp0 = nowMs();
        const lcp = longestCommonPrefix(self.active_tokens.items, prompt_toks);
        const max_resume: usize = @min(lcp, prompt_toks.len - 1);

        var best_ckpt_idx: ?usize = null;
        var best_ckpt_pos: u32 = 0;
        for (self.checkpoints, 0..) |c, idx| {
            if (c.valid and c.pos > 0 and c.pos <= max_resume and c.pos >= best_ckpt_pos) {
                best_ckpt_pos = c.pos;
                best_ckpt_idx = idx;
            }
        }

        var resume_pos: usize = 0;
        if (best_ckpt_idx) |cidx| {
            self.ckpt_clock += 1;
            const ckpt = &self.checkpoints[cidx];
            ckpt.last_used = self.ckpt_clock;
            resume_pos = ckpt.pos;
            self.invalidateCheckpointsAfter(ckpt.pos);
            try self.eng.loadGdnStateFromHost(ckpt.gdn_state.?, ckpt.gdn_conv.?, ckpt.pos);
            if (self.dflash) |df| {
                try df.loadKvRingFromHost(ckpt.dflash_kv.?);
            }
            self.active_tokens.shrinkRetainingCapacity(resume_pos);
        } else {
            self.invalidateCheckpointsAfter(0);
            self.active_tokens.clearRetainingCapacity();
            try self.eng.reset();
        }
        self.eng.capture_enabled = (self.dflash != null);

        var rng = std.Random.DefaultPrng.init(params.seed);

        // Prefill uncached suffix (saving a pinned system-prompt checkpoint & turn-end checkpoint if crossed)
        var cur_pos: usize = resume_pos;
        if (self.detectSystemBoundary(prompt, prompt_toks)) |sys_end| {
            if (cur_pos < sys_end and sys_end < prompt_toks.len) {
                _ = try self.prefillSegment(prompt_toks[cur_pos..sys_end], false);
                try self.saveCheckpoint(@intCast(sys_end), true);
                cur_pos = sys_end;
            }
        }
        if (self.detectTurnBoundary(prompt, prompt_toks)) |turn_end| {
            if (cur_pos < turn_end and turn_end < prompt_toks.len) {
                _ = try self.prefillSegment(prompt_toks[cur_pos..turn_end], false);
                try self.saveCheckpoint(@intCast(turn_end), false);
                cur_pos = turn_end;
            }
        }
        const pf = try self.prefillSegment(prompt_toks[cur_pos..], true);

        // Save prompt-end checkpoint so the next multi-turn request resumes right at prompt_toks.len!
        if (prompt_toks.len >= 64) {
            try self.saveCheckpoint(@intCast(prompt_toks.len), false);
        }
        const tp1 = nowMs();

        var stats: GenStats = .{
            .prompt_tokens = prompt_toks.len,
            .prompt_ms = tp1 - tp0,
            .finish_reason = "length",
        };

        var reasoning_text: std.ArrayListUnmanaged(u8) = .empty;
        defer reasoning_text.deinit(self.allocator);

        var full_text: std.ArrayListUnmanaged(u8) = .empty;
        defer full_text.deinit(self.allocator);

        var stream_pending: std.ArrayListUnmanaged(u8) = .empty;
        defer stream_pending.deinit(self.allocator);

        var piece_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer piece_buf.deinit(self.allocator);

        var max_stop_len: usize = 0;
        for (params.stops[0..params.n_stops]) |st| {
            if (st.len > max_stop_len) max_stop_len = st.len;
        }

        const has_tools = mode == .chat and (if (tools_opt) |ts| ts.len > 0 else false);
        var tool_parser = ToolCallParser.init(self.allocator, req_id, tools_opt);
        defer tool_parser.deinit();

        const LiveToolSink = struct {
            allocator: std.mem.Allocator,
            fd: std.c.fd_t,
            mode: Mode,
            req_id: []const u8,
            created: i64,
            params: GenParams,
            max_sl: usize,
            full_t: *std.ArrayListUnmanaged(u8),
            pend: *std.ArrayListUnmanaged(u8),
            stats: *GenStats,
            hit_stop: bool = false,

            fn onContent(s: *@This(), chunk: []const u8) !void {
                if (chunk.len == 0 or s.hit_stop) return;
                try s.full_t.appendSlice(s.allocator, chunk);
                if (s.params.stream) {
                    try s.pend.appendSlice(s.allocator, chunk);
                    if (s.params.n_stops > 0) {
                        if (findStopInText(s.pend.items, s.params.stops[0..s.params.n_stops])) |sidx| {
                            if (sidx > 0) {
                                try sendSseChunk(s.allocator, s.fd, s.mode, s.req_id, s.created, s.pend.items[0..sidx], false, false, null, null);
                            }
                            s.pend.clearRetainingCapacity();
                            s.stats.finish_reason = "stop";
                            s.hit_stop = true;
                            return;
                        }
                        const safe_stop_len = if (s.pend.items.len > s.max_sl) s.pend.items.len - (s.max_sl - 1) else 0;
                        const ready = utf8ReadyPrefixLen(s.pend.items[0..safe_stop_len]);
                        if (ready > 0) {
                            try sendSseChunk(s.allocator, s.fd, s.mode, s.req_id, s.created, s.pend.items[0..ready], false, false, null, null);
                            const rem = s.pend.items.len - ready;
                            std.mem.copyForwards(u8, s.pend.items[0..rem], s.pend.items[ready..]);
                            s.pend.shrinkRetainingCapacity(rem);
                        }
                    } else {
                        const ready = utf8ReadyPrefixLen(s.pend.items);
                        if (ready > 0) {
                            try sendSseChunk(s.allocator, s.fd, s.mode, s.req_id, s.created, s.pend.items[0..ready], false, false, null, null);
                            const rem = s.pend.items.len - ready;
                            std.mem.copyForwards(u8, s.pend.items[0..rem], s.pend.items[ready..]);
                            s.pend.shrinkRetainingCapacity(rem);
                        }
                    }
                } else if (s.params.n_stops > 0) {
                    if (findStopInText(s.full_t.items, s.params.stops[0..s.params.n_stops])) |sidx| {
                        s.full_t.shrinkRetainingCapacity(sidx);
                        s.stats.finish_reason = "stop";
                        s.hit_stop = true;
                    }
                }
            }

            fn onToolCallStart(s: *@This(), call_idx: usize, call_id: []const u8, fn_name: []const u8) !void {
                if (!s.params.stream) return;
                if (s.pend.items.len > 0) {
                    try sendSseChunk(s.allocator, s.fd, s.mode, s.req_id, s.created, s.pend.items, false, false, null, null);
                    s.pend.clearRetainingCapacity();
                }
                try sendToolCallSseStart(s.allocator, s.fd, s.req_id, s.created, call_idx, call_id, fn_name);
            }

            fn onToolCallArgsDelta(s: *@This(), call_idx: usize, args_delta: []const u8) !void {
                if (!s.params.stream) return;
                try sendToolCallSseArgs(s.allocator, s.fd, s.req_id, s.created, call_idx, args_delta);
            }
        };

        var tool_sink: LiveToolSink = .{
            .allocator = self.allocator,
            .fd = fd,
            .mode = mode,
            .req_id = req_id,
            .created = created,
            .params = params,
            .max_sl = max_stop_len,
            .full_t = &full_text,
            .pend = &stream_pending,
            .stats = &stats,
        };

        const ThinkState = struct {
            in_thinking: bool,
            strip_post_think_nl: bool = false,
            has_tools: bool,
            reasoning: *std.ArrayListUnmanaged(u8),
            parser: *ToolCallParser,
            sink: *LiveToolSink,
        };
        var think_st: ThinkState = .{
            .in_thinking = (mode == .chat and effort != .none),
            .has_tools = has_tools,
            .reasoning = &reasoning_text,
            .parser = &tool_parser,
            .sink = &tool_sink,
        };

        const emitToken = struct {
            fn call(
                st_self: *ServerState,
                tok_id: u32,
                fd_c: std.c.fd_t,
                mode_c: Mode,
                req_id_c: []const u8,
                created_c: i64,
                p_c: GenParams,
                p_buf: *std.ArrayListUnmanaged(u8),
                pend: *std.ArrayListUnmanaged(u8),
                t_st: *ThinkState,
                g_stats: *GenStats,
            ) !bool {
                if (st_self.tok.isEos(tok_id)) {
                    g_stats.finish_reason = "stop";
                    return true;
                }
                g_stats.completion_tokens += 1;
                p_buf.clearRetainingCapacity();
                try st_self.tok.decodeTokenTo(tok_id, p_buf, st_self.allocator);

                var piece: []const u8 = p_buf.items;

                // Separate <think>...</think> into OpenAI/DeepSeek reasoning_content vs content
                if (mode_c == .chat and t_st.in_thinking) {
                    if (std.mem.eql(u8, piece, "<think>")) {
                        // Prompt already opened <think>\n; ignore any redundant opening tag
                        if (g_stats.completion_tokens >= p_c.max_tokens) {
                            g_stats.finish_reason = "length";
                            return true;
                        }
                        return false;
                    }
                    if (std.mem.indexOf(u8, piece, "</think>")) |close_idx| {
                        const before = piece[0..close_idx];
                        if (before.len > 0) {
                            try t_st.reasoning.appendSlice(st_self.allocator, before);
                            if (p_c.stream) try pend.appendSlice(st_self.allocator, before);
                        }
                        // Flush all remaining reasoning bytes before switching to content mode
                        if (p_c.stream and pend.items.len > 0) {
                            try sendSseChunk(st_self.allocator, fd_c, mode_c, req_id_c, created_c, pend.items, true, false, null, null);
                            pend.clearRetainingCapacity();
                        }
                        t_st.in_thinking = false;
                        t_st.strip_post_think_nl = true;

                        const after = piece[close_idx + "</think>".len ..];
                        const after_trimmed = std.mem.trimStart(u8, after, "\r\n");
                        if (after_trimmed.len == 0) {
                            if (g_stats.completion_tokens >= p_c.max_tokens) {
                                g_stats.finish_reason = "length";
                                return true;
                            }
                            return false;
                        }
                        t_st.strip_post_think_nl = false;
                        piece = after_trimmed;
                    }
                } else if (mode_c == .chat and t_st.strip_post_think_nl) {
                    const trimmed = std.mem.trimStart(u8, piece, "\r\n");
                    if (trimmed.len == 0) {
                        if (g_stats.completion_tokens >= p_c.max_tokens) {
                            g_stats.finish_reason = "length";
                            return true;
                        }
                        return false;
                    }
                    t_st.strip_post_think_nl = false;
                    piece = trimmed;
                }

                if (mode_c == .chat and t_st.in_thinking) {
                    try t_st.reasoning.appendSlice(st_self.allocator, piece);
                    if (p_c.stream) {
                        try pend.appendSlice(st_self.allocator, piece);
                        const ready = utf8ReadyPrefixLen(pend.items);
                        if (ready > 0) {
                            try sendSseChunk(st_self.allocator, fd_c, mode_c, req_id_c, created_c, pend.items[0..ready], true, false, null, null);
                            const rem = pend.items.len - ready;
                            std.mem.copyForwards(u8, pend.items[0..rem], pend.items[ready..]);
                            pend.shrinkRetainingCapacity(rem);
                        }
                    }
                } else if (t_st.has_tools) {
                    try t_st.parser.feed(piece, t_st.sink);
                    if (t_st.sink.hit_stop) return true;
                } else {
                    try t_st.sink.onContent(piece);
                    if (t_st.sink.hit_stop) return true;
                }

                if (g_stats.completion_tokens >= p_c.max_tokens) {
                    g_stats.finish_reason = "length";
                    return true;
                }
                return false;
            }
        }.call;

        var cur = engine_mod.sampleFromCandidates(&pf.cands[0], params.temperature, params.top_k, params.top_p, &rng);

        const td0 = nowMs();
        if (params.max_tokens > 0) {
            var stopped = try emitToken(self, cur, fd, mode, req_id, created, params, &piece_buf, &stream_pending, &think_st, &stats);

            if (!stopped) {
                if (self.dflash) |df| {
                    while (!stopped and stats.completion_tokens < params.max_tokens) {
                        var draft_toks: [dflash_mod.N_DRAFT]u32 = undefined;
                        const n_draft = try df.draft(self.eng, cur, self.eng.pos, 0.0, &draft_toks);
                        stats.draft_n += n_draft;

                        var batch: [dflash_mod.BLOCK_SIZE]u32 = undefined;
                        batch[0] = cur;
                        for (0..n_draft) |k| batch[1 + k] = draft_toks[k];
                        const m: u32 = 1 + n_draft;

                        const out = try self.eng.forward(batch[0..m], false, true);
                        stats.spec_steps += 1;

                        var n_acc: u32 = 0;
                        var next_tok: u32 = 0;

                        for (0..n_draft) |k| {
                            const t_tok = if (params.temperature <= 0.0 or params.top_k == 1)
                                out.tokens[k]
                            else
                                engine_mod.sampleFromCandidates(&out.cands[k], params.temperature, params.top_k, params.top_p, &rng);
                            stopped = try emitToken(self, t_tok, fd, mode, req_id, created, params, &piece_buf, &stream_pending, &think_st, &stats);
                            if (stopped) break;
                            if (t_tok == draft_toks[k]) {
                                n_acc += 1;
                            } else {
                                next_tok = t_tok;
                                break;
                            }
                        }
                        stats.draft_n_accepted += n_acc;

                        if (!stopped and n_acc == n_draft) {
                            const bonus = if (params.temperature <= 0.0 or params.top_k == 1)
                                out.tokens[n_draft]
                            else
                                engine_mod.sampleFromCandidates(&out.cands[n_draft], params.temperature, params.top_k, params.top_p, &rng);
                            stopped = try emitToken(self, bonus, fd, mode, req_id, created, params, &piece_buf, &stream_pending, &think_st, &stats);
                            next_tok = bonus;
                        }
                        if (stopped) break;

                        const n_commit = n_acc + 1;
                        const pos_before = self.eng.pos;
                        try self.eng.commitSpeculative(n_commit);
                        try df.encodeAndInject(self.eng, n_commit, pos_before);
                        try self.active_tokens.appendSlice(self.allocator, batch[0..n_commit]);
                        cur = next_tok;
                    }
                } else {
                    while (!stopped and stats.completion_tokens < params.max_tokens) {
                        const out = try self.eng.forward(&[_]u32{cur}, true, true);
                        try self.active_tokens.append(self.allocator, cur);
                        cur = engine_mod.sampleFromCandidates(&out.cands[0], params.temperature, params.top_k, params.top_p, &rng);
                        stopped = try emitToken(self, cur, fd, mode, req_id, created, params, &piece_buf, &stream_pending, &think_st, &stats);
                    }
                }
            }
        }
        if (has_tools) {
            try tool_parser.finish(&tool_sink);
            if (tool_parser.calls.items.len > 0 and std.mem.eql(u8, stats.finish_reason, "stop")) {
                stats.finish_reason = "tool_calls";
            }
        }
        const td1 = nowMs();
        stats.decode_ms = td1 - td0;

        if (params.stream) {
            // Flush any buffered UTF-8 / stop-tail bytes before final chunk
            if (stream_pending.items.len > 0) {
                try sendSseChunk(self.allocator, fd, mode, req_id, created, stream_pending.items, mode == .chat and think_st.in_thinking, false, null, null);
                stream_pending.clearRetainingCapacity();
            }
            try sendSseChunk(self.allocator, fd, mode, req_id, created, null, false, false, stats.finish_reason, stats);
            try sendAll(fd, "data: [DONE]\n\n");
        } else {
            var j: std.ArrayListUnmanaged(u8) = .empty;
            defer j.deinit(self.allocator);

            const pps = if (stats.prompt_ms > 0.0) @as(f64, @floatFromInt(stats.prompt_tokens)) * 1000.0 / stats.prompt_ms else 0.0;
            const dps = if (stats.decode_ms > 0.0) @as(f64, @floatFromInt(stats.completion_tokens)) * 1000.0 / stats.decode_ms else 0.0;

            if (mode == .chat) {
                try appendFmt(&j, self.allocator, "{{\"id\":\"{s}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":", .{
                    req_id, created, MODEL_ID,
                });
                if (tool_parser.calls.items.len > 0 and std.mem.trim(u8, full_text.items, " \t\r\n").len == 0) {
                    try j.appendSlice(self.allocator, "null");
                } else {
                    try j.append(self.allocator, '"');
                    try appendJsonEscaped(&j, self.allocator, full_text.items);
                    try j.append(self.allocator, '"');
                }
                if (reasoning_text.items.len > 0) {
                    try j.appendSlice(self.allocator, ",\"reasoning_content\":\"");
                    try appendJsonEscaped(&j, self.allocator, reasoning_text.items);
                    try j.append(self.allocator, '"');
                }
                if (tool_parser.calls.items.len > 0) {
                    try j.appendSlice(self.allocator, ",\"tool_calls\":[");
                    for (tool_parser.calls.items, 0..) |tc, ci| {
                        if (ci > 0) try j.append(self.allocator, ',');
                        try j.appendSlice(self.allocator, "{\"id\":\"");
                        try appendJsonEscaped(&j, self.allocator, tc.id);
                        try j.appendSlice(self.allocator, "\",\"type\":\"function\",\"function\":{\"name\":\"");
                        try appendJsonEscaped(&j, self.allocator, tc.name);
                        try j.appendSlice(self.allocator, "\",\"arguments\":\"");
                        try appendJsonEscaped(&j, self.allocator, tc.arguments.items);
                        try j.appendSlice(self.allocator, "\"}}");
                    }
                    try j.append(self.allocator, ']');
                }
                try appendFmt(&j, self.allocator, "}},\"finish_reason\":\"{s}\"}}]", .{stats.finish_reason});
            } else {
                try appendFmt(&j, self.allocator, "{{\"id\":\"{s}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"{s}\",\"choices\":[{{\"text\":\"", .{
                    req_id, created, MODEL_ID,
                });
                try appendJsonEscaped(&j, self.allocator, full_text.items);
                try appendFmt(&j, self.allocator, "\",\"index\":0,\"finish_reason\":\"{s}\"}}]", .{stats.finish_reason});
            }

            try appendFmt(
                &j,
                self.allocator,
                ",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}" ++
                    ",\"timings\":{{\"prompt_n\":{d},\"prompt_ms\":{d:.2},\"prompt_per_second\":{d:.2}," ++
                    "\"predicted_n\":{d},\"predicted_ms\":{d:.2},\"predicted_per_second\":{d:.2}," ++
                    "\"spec_steps\":{d},\"draft_n\":{d},\"draft_n_accepted\":{d}}}}}",
                .{
                    stats.prompt_tokens,
                    stats.completion_tokens,
                    stats.prompt_tokens + stats.completion_tokens,
                    stats.prompt_tokens,
                    stats.prompt_ms,
                    pps,
                    stats.completion_tokens,
                    stats.decode_ms,
                    dps,
                    stats.spec_steps,
                    stats.draft_n,
                    stats.draft_n_accepted,
                },
            );
            try sendHttpJson(fd, self.allocator, "200 OK", j.items);
        }

        // Also save post-generation checkpoint if decode committed >= 16 tokens (helps clients that keep <think>)
        if (self.eng.pos >= prompt_toks.len + 16) {
            self.saveCheckpoint(self.eng.pos, false) catch {};
        }

        const dps_log = if (stats.decode_ms > 0.0) @as(f64, @floatFromInt(stats.completion_tokens)) * 1000.0 / stats.decode_ms else 0.0;
        const new_prompt_toks = prompt_toks.len - resume_pos;
        std.debug.print(
            "[kintsugi-zig] {s} stream={} effort={s} tools={d} | prompt={d} (cached={d}, new={d}, {d:.1} ms) | gen={d} ({d:.1} ms, {d:.2} tok/s) | spec={d}/{d} ({d} steps)\n",
            .{
                if (mode == .chat) "POST /v1/chat/completions" else "POST /v1/completions",
                params.stream,
                effort.name(),
                tool_parser.calls.items.len,
                stats.prompt_tokens,
                resume_pos,
                new_prompt_toks,
                stats.prompt_ms,
                stats.completion_tokens,
                stats.decode_ms,
                dps_log,
                stats.draft_n_accepted,
                stats.draft_n,
                stats.spec_steps,
            },
        );
    }

    fn handleClient(self: *ServerState, fd: std.c.fd_t) !void {
        var req = (try readHttpRequest(self.allocator, fd)) orelse return;
        defer req.deinit(self.allocator);

        if (std.mem.eql(u8, req.method, "OPTIONS")) {
            try sendHttpJson(fd, self.allocator, "204 No Content", "");
            return;
        }

        if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, req.path, "/health") or std.mem.eql(u8, req.path, "/"))) {
            const body = try std.fmt.allocPrint(
                self.allocator,
                "{{\"status\":\"ok\",\"model\":\"{s}\",\"speculative\":{},\"kv_cells\":{d}}}",
                .{ MODEL_ID, self.dflash != null, self.eng.max_kv_cells },
            );
            defer self.allocator.free(body);
            try sendHttpJson(fd, self.allocator, "200 OK", body);
            return;
        }

        if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, req.path, "/v1/models") or std.mem.eql(u8, req.path, "/models"))) {
            const body =
                "{\"object\":\"list\",\"data\":[{\"id\":\"Kintsugi-27B\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"kintsugi-zig\"}]}";
            try sendHttpJson(fd, self.allocator, "200 OK", body);
            return;
        }

        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/tokenize")) {
            var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, req.body, .{}) catch {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"invalid JSON\"}}");
                return;
            };
            defer parsed.deinit();
            if (parsed.value != .object) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"expected JSON object\"}}");
                return;
            }
            const content_val = parsed.value.object.get("content") orelse parsed.value.object.get("prompt") orelse parsed.value.object.get("text");
            if (content_val == null or content_val.? != .string) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"missing string field 'content'\"}}");
                return;
            }
            const toks = try self.tok.encode(self.allocator, content_val.?.string);
            defer self.allocator.free(toks);

            var j: std.ArrayListUnmanaged(u8) = .empty;
            defer j.deinit(self.allocator);
            try j.appendSlice(self.allocator, "{\"tokens\":[");
            for (toks, 0..) |t, idx| {
                if (idx > 0) try j.append(self.allocator, ',');
                try appendFmt(&j, self.allocator, "{d}", .{t});
            }
            try j.appendSlice(self.allocator, "]}");
            try sendHttpJson(fd, self.allocator, "200 OK", j.items);
            return;
        }

        if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/v1/completions") or std.mem.eql(u8, req.path, "/completions"))) {
            var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, req.body, .{}) catch {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"invalid JSON\"}}");
                return;
            };
            defer parsed.deinit();
            if (parsed.value != .object) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"expected JSON object\"}}");
                return;
            }
            const prompt_val = parsed.value.object.get("prompt") orelse {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"missing 'prompt'\"}}");
                return;
            };
            if (prompt_val != .string) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"'prompt' must be a string\"}}");
                return;
            }
            const params = parseGenParams(parsed.value.object);
            try self.runGeneration(fd, .completion, prompt_val.string, params, .none, null);
            return;
        }

        if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/v1/chat/completions") or std.mem.eql(u8, req.path, "/chat/completions"))) {
            var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, req.body, .{}) catch {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"invalid JSON\"}}");
                return;
            };
            defer parsed.deinit();
            if (parsed.value != .object) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"expected JSON object\"}}");
                return;
            }
            const msgs_val = parsed.value.object.get("messages") orelse {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"missing 'messages'\"}}");
                return;
            };
            if (msgs_val != .array) {
                try sendHttpJson(fd, self.allocator, "400 Bad Request", "{\"error\":{\"message\":\"'messages' must be an array\"}}");
                return;
            }
            const effort = parseReasoningEffort(parsed.value.object);
            const tools_opt: ?[]const std.json.Value = if (parsed.value.object.get("tools")) |tv|
                (if (tv == .array) tv.array.items else null)
            else
                null;
            const formatted = try formatChatMl(self.allocator, msgs_val.array.items, effort, tools_opt);
            defer self.allocator.free(formatted);
            const params = parseGenParams(parsed.value.object);
            try self.runGeneration(fd, .chat, formatted, params, effort, tools_opt);
            return;
        }

        std.debug.print("[kintsugi-zig] 404 {s} {s}\n", .{ req.method, req.path });
        try sendHttpJson(fd, self.allocator, "404 Not Found", "{\"error\":{\"message\":\"endpoint not found\"}}");
    }
};

const ListenSocket = struct {
    fd: std.c.fd_t,
    dual_stack: bool,
};

fn createListenSocket(host: []const u8, port: u16) !ListenSocket {
    const is_wildcard = std.mem.eql(u8, host, "0.0.0.0") or std.mem.eql(u8, host, "::");
    const is_v6_loopback = std.mem.eql(u8, host, "::1");

    if (is_wildcard or is_v6_loopback) {
        const fd6 = std.c.socket(std.c.AF.INET6, std.c.SOCK.STREAM, 0);
        if (fd6 >= 0) {
            var reuse: c_int = 1;
            _ = std.c.setsockopt(fd6, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, &reuse, @sizeOf(c_int));
            var v6only: c_int = if (is_v6_loopback) 1 else 0;
            _ = std.c.setsockopt(fd6, std.c.IPPROTO.IPV6, std.c.IPV6.V6ONLY, &v6only, @sizeOf(c_int));

            var addr6: std.c.sockaddr.in6 = .{
                .family = std.c.AF.INET6,
                .port = std.mem.nativeToBig(u16, port),
                .flowinfo = 0,
                .addr = @splat(0),
                .scope_id = 0,
            };
            if (is_v6_loopback) {
                addr6.addr[15] = 1;
            }
            if (std.c.bind(fd6, @ptrCast(&addr6), @sizeOf(std.c.sockaddr.in6)) == 0) {
                if (std.c.listen(fd6, 32) == 0) {
                    return .{ .fd = fd6, .dual_stack = is_wildcard };
                }
            }
            _ = std.c.close(fd6);
            if (is_v6_loopback) {
                std.debug.print("error: failed to bind [{s}]:{d}\n", .{ host, port });
                return error.BindFailed;
            }
            // Fall through to IPv4 bind if IPv6 wildcard bind failed
        }
    }

    const fd4 = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd4 < 0) return error.SocketCreateFailed;
    errdefer _ = std.c.close(fd4);

    var reuse: c_int = 1;
    _ = std.c.setsockopt(fd4, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, &reuse, @sizeOf(c_int));

    var addr4: std.c.sockaddr.in = .{
        .family = std.c.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = try parseIpv4(if (std.mem.eql(u8, host, "::")) "0.0.0.0" else host),
        .zero = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    if (std.c.bind(fd4, @ptrCast(&addr4), @sizeOf(std.c.sockaddr.in)) < 0) {
        std.debug.print("error: failed to bind {s}:{d}\n", .{ host, port });
        return error.BindFailed;
    }
    if (std.c.listen(fd4, 32) < 0) {
        return error.ListenFailed;
    }
    return .{ .fd = fd4, .dual_stack = false };
}

pub fn run(allocator: std.mem.Allocator, cfg: ServeConfig) !void {
    const t0 = nowMs();
    var ctx = try cuda.Context.init(0);
    defer ctx.deinit();
    var ks = try engine_mod.Kernels.load();
    defer ks.deinit();
    var model = try model_mod.TargetModel.load(allocator, &ctx, ks.fn_q8_to_q4, cfg.model_path);
    defer model.deinit();
    var tok = try tokenizer_mod.Tokenizer.init(allocator, &model.gguf_file);
    defer tok.deinit();
    var eng = try engine_mod.Engine.init(allocator, &ctx, &ks, &model, if (cfg.use_spec) 1060 else 160);
    defer eng.deinit();

    var dflash_opt: ?dflash_mod.DFlashEngine = if (cfg.use_spec)
        try dflash_mod.DFlashEngine.load(allocator, &ctx, cfg.drafter_path, &eng)
    else
        null;
    defer if (dflash_opt) |*df| df.deinit();

    const t1 = nowMs();
    const vinfo = try ctx.queryInfo();

    const listener = try createListenSocket(cfg.host, cfg.port);
    defer _ = std.c.close(listener.fd);

    std.debug.print(
        "kintsugi-zig OpenAI server listening on http://{s}:{d}{s}\n" ++
            "  Model      : {s} (loaded in {d:.0} ms | spec={} | KV pool: {d} Q4_0 cells | VRAM free: {d} MiB)\n" ++
            "  Endpoints  : GET /, GET /health, GET /v1/models, POST /tokenize, POST /v1/completions, POST /v1/chat/completions\n",
        .{
            cfg.host,
            cfg.port,
            if (listener.dual_stack) " (dual-stack IPv4 + IPv6)" else "",
            MODEL_ID,
            t1 - t0,
            cfg.use_spec,
            eng.max_kv_cells,
            vinfo.free_vram / (1024 * 1024),
        },
    );

    var state: ServerState = .{
        .allocator = allocator,
        .tok = &tok,
        .eng = &eng,
        .dflash = if (dflash_opt != null) &dflash_opt.? else null,
    };
    defer state.deinit();

    const rcv_timeout: std.c.timeval = .{ .sec = 10, .usec = 0 };

    while (true) {
        var client_addr: std.c.sockaddr.storage = undefined;
        var client_len: std.c.socklen_t = @sizeOf(std.c.sockaddr.storage);
        const client_fd = std.c.accept(listener.fd, @ptrCast(&client_addr), &client_len);
        if (client_fd < 0) continue;
        defer _ = std.c.close(client_fd);

        _ = std.c.setsockopt(client_fd, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, &rcv_timeout, @sizeOf(std.c.timeval));

        state.handleClient(client_fd) catch |err| {
            std.debug.print("[kintsugi-zig] request error: {}\n", .{err});
        };
    }
}

test "utf8ReadyPrefixLen holds back incomplete trailing multibyte sequences" {
    // Complete ASCII + 3-byte Chinese char "中" (0xE4 0xB8 0xAD)
    const full = "hi\xe4\xb8\xad";
    try std.testing.expectEqual(@as(usize, 5), utf8ReadyPrefixLen(full));
    try std.testing.expectEqual(@as(usize, 2), utf8ReadyPrefixLen(full[0..3]));
    try std.testing.expectEqual(@as(usize, 2), utf8ReadyPrefixLen(full[0..4]));
    // 4-byte emoji (0xF0 0x9F 0x9A 0x80)
    const emoji = "a\xf0\x9f\x9a\x80";
    try std.testing.expectEqual(@as(usize, 5), utf8ReadyPrefixLen(emoji));
    try std.testing.expectEqual(@as(usize, 1), utf8ReadyPrefixLen(emoji[0..4]));
}

test "appendJsonEscaped escapes quotes newlines and control chars" {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try appendJsonEscaped(&buf, std.testing.allocator, "a\"b\\c\nd\te\x01");
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd\\te\\u0001", buf.items);
}

test "parseIpv4 parses localhost and dotted quads" {
    try std.testing.expectEqual(std.mem.nativeToBig(u32, 0x7f000001), try parseIpv4("127.0.0.1"));
    try std.testing.expectEqual(std.mem.nativeToBig(u32, 0x7f000001), try parseIpv4("localhost"));
    try std.testing.expectEqual(std.mem.nativeToBig(u32, 0x00000000), try parseIpv4("0.0.0.0"));
}

test "tryDecodeChunkedBody decodes multi-chunk HTTP bodies and handles partial data" {
    const partial = "5\r\nhello\r\n6\r\n world";
    try std.testing.expect((try tryDecodeChunkedBody(std.testing.allocator, partial)) == null);

    const complete = "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
    const decoded = (try tryDecodeChunkedBody(std.testing.allocator, complete)).?;
    defer std.testing.allocator.free(decoded);
    try std.testing.expectEqualStrings("hello world", decoded);
}

test "formatChatMl supports both string and multimodal array message content" {
    const payload =
        \\[{"role":"system","content":"You are helpful."},{"role":"user","content":[{"type":"text","text":"Line 1"},{"type":"text","text":"Line 2"}]}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, payload, .{});
    defer parsed.deinit();

    const formatted = try formatChatMl(std.testing.allocator, parsed.value.array.items, .none, null);
    defer std.testing.allocator.free(formatted);

    try std.testing.expectEqualStrings(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nLine 1\nLine 2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        formatted,
    );
}

test "formatChatMl injects official reasoning_effort instructions and <think> prompt" {
    const payload =
        \\[{"role":"user","content":"2+2?"},{"role":"assistant","reasoning_content":"2+2=4","content":"4"},{"role":"user","content":"3+3?"}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, payload, .{});
    defer parsed.deinit();

    const low = try formatChatMl(std.testing.allocator, parsed.value.array.items, .low, null);
    defer std.testing.allocator.free(low);
    try std.testing.expect(std.mem.startsWith(u8, low, "<|im_start|>system\nReasoning effort is set to low."));
    try std.testing.expect(std.mem.indexOf(u8, low, "<|im_start|>assistant\n<think>\n2+2=4\n</think>\n\n4<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, low, "<|im_start|>assistant\n<think>\n"));

    const med = try formatChatMl(std.testing.allocator, parsed.value.array.items, .medium, null);
    defer std.testing.allocator.free(med);
    try std.testing.expect(std.mem.startsWith(u8, med, "<|im_start|>user\n2+2?<|im_end|>\n"));
    try std.testing.expect(std.mem.endsWith(u8, med, "<|im_start|>assistant\n<think>\n"));

    const xhigh = try formatChatMl(std.testing.allocator, parsed.value.array.items, .xhigh, null);
    defer std.testing.allocator.free(xhigh);
    try std.testing.expect(std.mem.startsWith(u8, xhigh, "<|im_start|>system\nReasoning effort is set to xhigh."));
    try std.testing.expect(std.mem.endsWith(u8, xhigh, "<|im_start|>assistant\n<think>\n"));
}

test "parseReasoningEffort handles OpenAI, DeepSeek, Anthropic and vLLM fields" {
    const cases = [_]struct { json: []const u8, expected: ReasoningEffort }{
        .{ .json = "{}", .expected = .xhigh },
        .{ .json = "{\"reasoning_effort\":\"low\"}", .expected = .low },
        .{ .json = "{\"reasoning_effort\":\"medium\"}", .expected = .medium },
        .{ .json = "{\"reasoning_effort\":\"high\"}", .expected = .xhigh },
        .{ .json = "{\"reasoning_effort\":\"xhigh\"}", .expected = .xhigh },
        .{ .json = "{\"reasoning_effort\":\"none\"}", .expected = .none },
        .{ .json = "{\"reasoning\":{\"effort\":\"low\"}}", .expected = .low },
        .{ .json = "{\"reasoning\":{\"enabled\":false}}", .expected = .none },
        .{ .json = "{\"enable_thinking\":false}", .expected = .none },
        .{ .json = "{\"thinking\":{\"type\":\"disabled\"}}", .expected = .none },
        .{ .json = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}}", .expected = .low },
        .{ .json = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":4096}}", .expected = .medium },
        .{ .json = "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}", .expected = .low },
        .{ .json = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .expected = .none },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.json, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(c.expected, parseReasoningEffort(parsed.value.object));
    }
}

test "ToolCallParser streams HTML write tool call byte-by-byte without losing UTF-8 or quotes" {
    const tools_json =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}}}]
    ;
    var parsed_tools = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, tools_json, .{});
    defer parsed_tools.deinit();

    const MockSink = struct {
        allocator: std.mem.Allocator,
        content: std.ArrayListUnmanaged(u8) = .empty,
        starts: usize = 0,
        streamed_args: std.ArrayListUnmanaged(u8) = .empty,
        args_chunks: usize = 0,

        fn deinit(s: *@This()) void {
            s.content.deinit(s.allocator);
            s.streamed_args.deinit(s.allocator);
        }
        fn onContent(s: *@This(), chunk: []const u8) !void {
            try s.content.appendSlice(s.allocator, chunk);
        }
        fn onToolCallStart(s: *@This(), _: usize, _: []const u8, _: []const u8) !void {
            s.starts += 1;
        }
        fn onToolCallArgsDelta(s: *@This(), _: usize, delta: []const u8) !void {
            s.args_chunks += 1;
            try s.streamed_args.appendSlice(s.allocator, delta);
        }
    };

    const html = "<!DOCTYPE html>\n<html><body><h1 class=\"title\">你好世界</h1><script>const x = \"a\\nb\";</script></body></html>";
    const xml =
        "<tool_call>\n<function=write>\n<parameter=file_path>\nindex.html\n</parameter>\n<parameter=content>\n" ++
        html ++
        "\n</parameter>\n</function>\n</tool_call>";

    var sink: MockSink = .{ .allocator = std.testing.allocator };
    defer sink.deinit();

    var parser = ToolCallParser.init(std.testing.allocator, "req1", parsed_tools.value.array.items);
    defer parser.deinit();

    // Feed 1 byte at a time to stress-test every tag boundary and multi-byte UTF-8 holdback
    for (xml) |b| {
        try parser.feed(&[_]u8{b}, &sink);
    }
    try parser.finish(&sink);

    try std.testing.expectEqualStrings("", sink.content.items);
    try std.testing.expectEqual(@as(usize, 1), sink.starts);
    try std.testing.expect(sink.args_chunks > 5); // actually streamed incrementally!
    try std.testing.expectEqual(@as(usize, 1), parser.calls.items.len);
    try std.testing.expectEqualStrings("write", parser.calls.items[0].name);
    try std.testing.expectEqualStrings(parser.calls.items[0].arguments.items, sink.streamed_args.items);

    var args_parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, sink.streamed_args.items, .{});
    defer args_parsed.deinit();
    try std.testing.expectEqualStrings("index.html", args_parsed.value.object.get("file_path").?.string);
    try std.testing.expectEqualStrings(html, args_parsed.value.object.get("content").?.string);
}

test "ToolCallParser handles pre-text, multiple calls, typed params, zero-arg calls, and JSON fallback" {
    const tools_json =
        \\[{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"exact":{"type":"boolean"}}}}},{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}]
    ;
    var parsed_tools = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, tools_json, .{});
    defer parsed_tools.deinit();

    const MockSink = struct {
        allocator: std.mem.Allocator,
        content: std.ArrayListUnmanaged(u8) = .empty,
        fn deinit(s: *@This()) void {
            s.content.deinit(s.allocator);
        }
        fn onContent(s: *@This(), chunk: []const u8) !void {
            try s.content.appendSlice(s.allocator, chunk);
        }
        fn onToolCallStart(_: *@This(), _: usize, _: []const u8, _: []const u8) !void {}
        fn onToolCallArgsDelta(_: *@This(), _: usize, _: []const u8) !void {}
    };

    const raw =
        "Let me check.\n\n" ++
        "<tool_call>\n<function=search>\n<parameter=query>\n123\n</parameter>\n<parameter=limit>\n5\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n</function>\n</tool_call>\n" ++
        "<tool_call>\n<function=ping>\n</function>\n</tool_call>\n" ++
        "<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>";

    var sink: MockSink = .{ .allocator = std.testing.allocator };
    defer sink.deinit();

    var parser = ToolCallParser.init(std.testing.allocator, "req2", parsed_tools.value.array.items);
    defer parser.deinit();

    // Feed in 3-byte slices
    var i: usize = 0;
    while (i < raw.len) : (i += 3) {
        const end = @min(i + 3, raw.len);
        try parser.feed(raw[i..end], &sink);
    }
    try parser.finish(&sink);

    try std.testing.expectEqualStrings("Let me check.\n\n", sink.content.items);
    try std.testing.expectEqual(@as(usize, 3), parser.calls.items.len);

    try std.testing.expectEqualStrings("search", parser.calls.items[0].name);
    var a0 = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, parser.calls.items[0].arguments.items, .{});
    defer a0.deinit();
    // "query" is schema type "string" -> stays string "123", not integer!
    try std.testing.expectEqualStrings("123", a0.value.object.get("query").?.string);
    try std.testing.expectEqual(@as(i64, 5), a0.value.object.get("limit").?.integer);
    try std.testing.expectEqual(true, a0.value.object.get("exact").?.bool);

    try std.testing.expectEqualStrings("ping", parser.calls.items[1].name);
    try std.testing.expectEqualStrings("{}", parser.calls.items[1].arguments.items);

    try std.testing.expectEqualStrings("ping", parser.calls.items[2].name);
    try std.testing.expectEqualStrings("{}", parser.calls.items[2].arguments.items);
}
