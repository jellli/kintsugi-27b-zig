//! Zero-overhead Zig bindings to NVIDIA CUDA Driver API (libcuda.so.1).
//! No libcudart, no libstdc++, direct access to sm_89 CUBIN loading, streams, events, and CUDA VMM.

const std = @import("std");

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUcontext = ?*opaque {};
pub const CUmodule = ?*opaque {};
pub const CUfunction = ?*opaque {};
pub const CUstream = ?*opaque {};
pub const CUevent = ?*opaque {};
pub const CUgraph = ?*opaque {};
pub const CUgraphExec = ?*opaque {};
pub const CUmemGenericAllocationHandle = u64;

pub const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 0x02;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

pub const CUdevice_attribute = enum(c_int) {
    max_threads_per_block = 1,
    max_shared_memory_per_block = 8,
    warp_size = 10,
    multiprocessor_count = 16,
    memory_clock_rate = 36,
    global_memory_bus_width = 37,
    l2_cache_size = 38,
    compute_capability_major = 75,
    compute_capability_minor = 76,
    max_shared_memory_per_block_optin = 97,
    virtual_memory_management_supported = 102,
};

pub const CUmemAllocationType = enum(c_int) {
    pinned = 1,
};

pub const CUmemLocationType = enum(c_int) {
    device = 1,
    host = 2,
};

pub const CUmemLocation = extern struct {
    type: CUmemLocationType,
    id: c_int,
};

pub const CUmemAllocationProp = extern struct {
    type: CUmemAllocationType,
    requestedHandleTypes: c_int = 0,
    location: CUmemLocation,
    win32HandleMetaData: ?*anyopaque = null,
    allocFlags: extern struct {
        compressionType: u8 = 0,
        gpuDirectRDMACapable: u8 = 0,
        usage: u16 = 0,
        reserved: [4]u8 = .{ 0, 0, 0, 0 },
    } = .{},
};

pub const CUmemAllocationGranularity_flags = enum(c_int) {
    minimum = 0,
    recommended = 1,
};

pub const CUmemAccess_flags = enum(c_int) {
    none = 0,
    read = 1,
    readwrite = 3,
};

pub const CUmemAccessDesc = extern struct {
    location: CUmemLocation,
    flags: CUmemAccess_flags,
};

extern "c" fn cuInit(flags: c_uint) CUresult;
extern "c" fn cuDriverGetVersion(driverVersion: *c_int) CUresult;
extern "c" fn cuDeviceGetCount(count: *c_int) CUresult;
extern "c" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
extern "c" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: CUdevice) CUresult;
extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: CUdevice_attribute, dev: CUdevice) CUresult;
extern "c" fn cuDeviceTotalMem_v2(bytes: *usize, dev: CUdevice) CUresult;
extern "c" fn cuCtxCreate_v2(pctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
extern "c" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
extern "c" fn cuCtxSynchronize() CUresult;
extern "c" fn cuMemGetInfo_v2(free: *usize, total: *usize) CUresult;
extern "c" fn cuMemAlloc_v2(dptr: *CUdeviceptr, bytesize: usize) CUresult;
extern "c" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
extern "c" fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: u8, N: usize) CUresult;
extern "c" fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: u8, N: usize, hStream: CUstream) CUresult;
extern "c" fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const anyopaque, ByteCount: usize) CUresult;
extern "c" fn cuMemcpyHtoDAsync_v2(dstDevice: CUdeviceptr, srcHost: *const anyopaque, ByteCount: usize, hStream: CUstream) CUresult;
extern "c" fn cuMemcpyDtoH_v2(dstHost: *anyopaque, srcDevice: CUdeviceptr, ByteCount: usize) CUresult;
extern "c" fn cuMemcpyDtoHAsync_v2(dstHost: *anyopaque, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) CUresult;
extern "c" fn cuMemcpyDtoDAsync_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) CUresult;
extern "c" fn cuMemHostRegister_v2(p: *anyopaque, bytesize: usize, Flags: c_uint) CUresult;
extern "c" fn cuMemHostUnregister(p: *anyopaque) CUresult;
extern "c" fn cuMemHostGetDevicePointer_v2(pdptr: *CUdeviceptr, p: *anyopaque, Flags: c_uint) CUresult;
extern "c" fn cuStreamCreate(phStream: *CUstream, Flags: c_uint) CUresult;
extern "c" fn cuStreamDestroy_v2(hStream: CUstream) CUresult;
extern "c" fn cuStreamSynchronize(hStream: CUstream) CUresult;
extern "c" fn cuStreamBeginCapture_v2(hStream: CUstream, mode: c_int) CUresult;
extern "c" fn cuStreamEndCapture(hStream: CUstream, phGraph: *CUgraph) CUresult;
extern "c" fn cuGraphInstantiateWithFlags(phGraphExec: *CUgraphExec, hGraph: CUgraph, flags: u64) CUresult;
extern "c" fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) CUresult;
extern "c" fn cuGraphDestroy(hGraph: CUgraph) CUresult;
extern "c" fn cuGraphExecDestroy(hGraphExec: CUgraphExec) CUresult;
extern "c" fn cuEventCreate(phEvent: *CUevent, Flags: c_uint) CUresult;
extern "c" fn cuEventDestroy_v2(hEvent: CUevent) CUresult;
extern "c" fn cuEventRecord(hEvent: CUevent, hStream: CUstream) CUresult;
extern "c" fn cuEventSynchronize(hEvent: CUevent) CUresult;
extern "c" fn cuEventElapsedTime(pMilliseconds: *f32, hStart: CUevent, hEnd: CUevent) CUresult;
extern "c" fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
extern "c" fn cuModuleUnload(hmod: CUmodule) CUresult;
extern "c" fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
extern "c" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) CUresult;
extern "c" fn cuMemGetAllocationGranularity(
    granularity: *usize,
    prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) CUresult;
extern "c" fn cuGetErrorName(err: CUresult, pStr: *[*:0]const u8) CUresult;
extern "c" fn cuGetErrorString(err: CUresult, pStr: *[*:0]const u8) CUresult;

pub const CudaError = error{CudaDriverError};

pub fn check(res: CUresult) CudaError!void {
    if (res == 0) return;
    var name: [*:0]const u8 = "UNKNOWN";
    var desc: [*:0]const u8 = "unknown error";
    _ = cuGetErrorName(res, &name);
    _ = cuGetErrorString(res, &desc);
    std.log.err("CUDA Driver error {d} ({s}): {s}", .{ res, name, desc });
    return error.CudaDriverError;
}

pub const DeviceInfo = struct {
    ordinal: c_int,
    name_buf: [128]u8 = undefined,
    name_len: usize = 0,
    cc_major: c_int,
    cc_minor: c_int,
    sm_count: c_int,
    l2_cache_bytes: c_int,
    max_smem_optin: c_int,
    vmm_supported: bool,
    vmm_min_granularity: usize,
    total_vram: usize,
    free_vram: usize,
    driver_version: c_int,

    pub fn name(self: *const DeviceInfo) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

pub const Context = struct {
    dev: CUdevice,
    ctx: CUcontext,
    stream: CUstream,

    pub fn init(ordinal: c_int) CudaError!Context {
        try check(cuInit(0));
        var count: c_int = 0;
        try check(cuDeviceGetCount(&count));
        if (ordinal >= count) return error.CudaDriverError;
        var dev: CUdevice = 0;
        try check(cuDeviceGet(&dev, ordinal));
        var ctx: CUcontext = null;
        try check(cuCtxCreate_v2(&ctx, 0, dev));
        var stream: CUstream = null;
        try check(cuStreamCreate(&stream, 0));
        return .{ .dev = dev, .ctx = ctx, .stream = stream };
    }

    pub fn deinit(self: *Context) void {
        _ = cuStreamDestroy_v2(self.stream);
        _ = cuCtxDestroy_v2(self.ctx);
    }

    pub fn queryInfo(self: *const Context) CudaError!DeviceInfo {
        var info: DeviceInfo = .{
            .ordinal = self.dev,
            .cc_major = 0,
            .cc_minor = 0,
            .sm_count = 0,
            .l2_cache_bytes = 0,
            .max_smem_optin = 0,
            .vmm_supported = false,
            .vmm_min_granularity = 0,
            .total_vram = 0,
            .free_vram = 0,
            .driver_version = 0,
        };
        try check(cuDriverGetVersion(&info.driver_version));
        @memset(&info.name_buf, 0);
        try check(cuDeviceGetName(&info.name_buf, info.name_buf.len, self.dev));
        info.name_len = std.mem.indexOfScalar(u8, &info.name_buf, 0) orelse info.name_buf.len;

        try check(cuDeviceGetAttribute(&info.cc_major, .compute_capability_major, self.dev));
        try check(cuDeviceGetAttribute(&info.cc_minor, .compute_capability_minor, self.dev));
        try check(cuDeviceGetAttribute(&info.sm_count, .multiprocessor_count, self.dev));
        try check(cuDeviceGetAttribute(&info.l2_cache_bytes, .l2_cache_size, self.dev));
        try check(cuDeviceGetAttribute(&info.max_smem_optin, .max_shared_memory_per_block_optin, self.dev));
        var vmm: c_int = 0;
        try check(cuDeviceGetAttribute(&vmm, .virtual_memory_management_supported, self.dev));
        info.vmm_supported = vmm != 0;

        try check(cuMemGetInfo_v2(&info.free_vram, &info.total_vram));

        if (info.vmm_supported) {
            const prop: CUmemAllocationProp = .{
                .type = .pinned,
                .location = .{ .type = .device, .id = self.dev },
            };
            try check(cuMemGetAllocationGranularity(&info.vmm_min_granularity, &prop, .minimum));
        }
        return info;
    }

    pub fn sync(self: *const Context) CudaError!void {
        try check(cuStreamSynchronize(self.stream));
    }

    pub fn alloc(bytes: usize) CudaError!CUdeviceptr {
        var ptr: CUdeviceptr = 0;
        try check(cuMemAlloc_v2(&ptr, bytes));
        return ptr;
    }

    pub fn free(ptr: CUdeviceptr) void {
        if (ptr != 0) _ = cuMemFree_v2(ptr);
    }

    pub fn copyHtoD(dst: CUdeviceptr, src: []const u8) CudaError!void {
        try check(cuMemcpyHtoD_v2(dst, src.ptr, src.len));
    }

    pub fn copyHtoDAsync(self: *const Context, dst: CUdeviceptr, src: []const u8) CudaError!void {
        try check(cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, self.stream));
    }

    pub fn copyDtoH(dst: []u8, src: CUdeviceptr) CudaError!void {
        try check(cuMemcpyDtoH_v2(dst.ptr, src, dst.len));
    }

    pub fn copyDtoHAsync(self: *const Context, dst: []u8, src: CUdeviceptr) CudaError!void {
        try check(cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, self.stream));
    }

    pub fn copyDtoDAsync(self: *const Context, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) CudaError!void {
        try check(cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream));
    }

    pub fn memsetZero(dst: CUdeviceptr, bytes: usize) CudaError!void {
        try check(cuMemsetD8_v2(dst, 0, bytes));
    }

    pub fn memsetZeroAsync(self: *const Context, dst: CUdeviceptr, bytes: usize) CudaError!void {
        try check(cuMemsetD8Async(dst, 0, bytes, self.stream));
    }

    pub fn hostRegisterDeviceMap(ptr: *anyopaque, bytes: usize) CudaError!CUdeviceptr {
        try check(cuMemHostRegister_v2(ptr, bytes, CU_MEMHOSTREGISTER_DEVICEMAP));
        var dptr: CUdeviceptr = 0;
        try check(cuMemHostGetDevicePointer_v2(&dptr, ptr, 0));
        return dptr;
    }

    pub fn hostUnregister(ptr: *anyopaque) void {
        _ = cuMemHostUnregister(ptr);
    }

    pub fn beginCapture(self: *const Context) CudaError!void {
        try check(cuStreamBeginCapture_v2(self.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    }

    pub fn endCaptureExec(self: *const Context) CudaError!GraphExec {
        var g: CUgraph = null;
        try check(cuStreamEndCapture(self.stream, &g));
        defer _ = cuGraphDestroy(g);
        var exec: CUgraphExec = null;
        try check(cuGraphInstantiateWithFlags(&exec, g, 0));
        return .{ .handle = exec };
    }
};

pub const GraphExec = struct {
    handle: CUgraphExec,

    pub fn launch(self: *const GraphExec, stream: CUstream) CudaError!void {
        try check(cuGraphLaunch(self.handle, stream));
    }

    pub fn deinit(self: *GraphExec) void {
        if (self.handle != null) {
            _ = cuGraphExecDestroy(self.handle);
            self.handle = null;
        }
    }
};

pub const Module = struct {
    handle: CUmodule,

    pub fn loadData(cubin: []const u8) CudaError!Module {
        var m: CUmodule = null;
        try check(cuModuleLoadData(&m, cubin.ptr));
        return .{ .handle = m };
    }

    pub fn deinit(self: *Module) void {
        _ = cuModuleUnload(self.handle);
    }

    pub fn getFunction(self: *const Module, name: [*:0]const u8) CudaError!CUfunction {
        var f: CUfunction = null;
        try check(cuModuleGetFunction(&f, self.handle, name));
        return f;
    }
};

pub fn launch(
    func: CUfunction,
    grid: [3]u32,
    block: [3]u32,
    smem_bytes: u32,
    stream: CUstream,
    params: []?*anyopaque,
) CudaError!void {
    try check(cuLaunchKernel(
        func,
        grid[0],
        grid[1],
        grid[2],
        block[0],
        block[1],
        block[2],
        smem_bytes,
        stream,
        params.ptr,
        null,
    ));
}

pub const Timer = struct {
    start_ev: CUevent,
    stop_ev: CUevent,

    pub fn init() CudaError!Timer {
        var s: CUevent = null;
        var e: CUevent = null;
        try check(cuEventCreate(&s, 0));
        try check(cuEventCreate(&e, 0));
        return .{ .start_ev = s, .stop_ev = e };
    }

    pub fn deinit(self: *Timer) void {
        _ = cuEventDestroy_v2(self.start_ev);
        _ = cuEventDestroy_v2(self.stop_ev);
    }

    pub fn begin(self: *const Timer, stream: CUstream) CudaError!void {
        try check(cuEventRecord(self.start_ev, stream));
    }

    pub fn endMs(self: *const Timer, stream: CUstream) CudaError!f32 {
        try check(cuEventRecord(self.stop_ev, stream));
        try check(cuEventSynchronize(self.stop_ev));
        var ms: f32 = 0.0;
        try check(cuEventElapsedTime(&ms, self.start_ev, self.stop_ev));
        return ms;
    }
};
