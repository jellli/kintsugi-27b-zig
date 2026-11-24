#!/usr/bin/env python3
"""Compile a CUDA .cu file directly into a native sm_89 .cubin using CUDA 12.8 NVRTC.

Zero host C++ compiler dependency (bypasses Fedora GCC 16 / libc++ header mismatches).
"""
import ctypes
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <input.cu> <output.cubin> [sm_arch]", file=sys.stderr)
        return 2

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    arch = sys.argv[3] if len(sys.argv) > 3 else "sm_89"

    cuda_home = os.environ.get(
        "CUDA_HOME", os.path.expanduser("~/.local/toolchains/cuda-12.8")
    )
    cuda_lib = os.path.join(cuda_home, "lib")
    builtins_path = os.path.join(cuda_lib, "libnvrtc-builtins.so.12.8")
    if os.path.exists(builtins_path):
        ctypes.CDLL(builtins_path, mode=ctypes.RTLD_GLOBAL)
    nvrtc_path = os.path.join(cuda_lib, "libnvrtc.so.12")
    if not os.path.exists(nvrtc_path):
        nvrtc_path = "libnvrtc.so.12"

    nvrtc = ctypes.CDLL(nvrtc_path)
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    nvrtc.nvrtcGetErrorString.argtypes = [ctypes.c_int]

    with open(in_path, "rb") as f:
        src = f.read()

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog), src, os.path.basename(in_path).encode(), 0, None, None
    )
    if rc != 0:
        print(f"nvrtcCreateProgram failed: {nvrtc.nvrtcGetErrorString(rc).decode()}", file=sys.stderr)
        return 1

    opts = [
        f"--gpu-architecture={arch}".encode(),
        f"-I{os.path.join(cuda_home, 'include')}".encode(),
        f"-I{os.path.dirname(os.path.abspath(in_path))}".encode(),
        b"--std=c++17",
        b"--use_fast_math",
        b"-lineinfo",
    ]
    c_opts = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)

    log_sz = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
    if log_sz.value > 1:
        log_buf = ctypes.create_string_buffer(log_sz.value)
        nvrtc.nvrtcGetProgramLog(prog, log_buf)
        text = log_buf.value.decode("utf-8", "replace").strip()
        if text:
            print(text, file=sys.stderr)

    if rc != 0:
        print(
            f"nvrtcCompileProgram failed ({in_path}): {nvrtc.nvrtcGetErrorString(rc).decode()}",
            file=sys.stderr,
        )
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return 1

    cubin_sz = ctypes.c_size_t()
    rc = nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_sz))
    if rc != 0:
        print(f"nvrtcGetCUBINSize failed: {nvrtc.nvrtcGetErrorString(rc).decode()}", file=sys.stderr)
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return 1

    cubin = ctypes.create_string_buffer(cubin_sz.value)
    rc = nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    if rc != 0:
        print(f"nvrtcGetCUBIN failed: {nvrtc.nvrtcGetErrorString(rc).decode()}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(cubin.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
