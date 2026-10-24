"""Compile every kernel file to PTX with NVRTC and assemble it with ptxas - a real device-code
compile on a machine without nvcc or a GPU (CI installs both from PyPI: nvidia-cuda-nvrtc-cu12 and
nvidia-cuda-nvcc-cu12, which ships ptxas).

    python tools/nvrtc_compile.py --arch sm_80 [--out build/ptx] [--verbose]

Prints per-kernel registers, static/dynamic shared memory and spills from `ptxas -v`, and fails on
any compile error.  The .cu files keep their host-side launchers behind `#ifndef __CUDACC_RTC__`,
which NVRTC defines, so the same sources serve nvcc, g++ (host stub) and NVRTC.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_nvidia_dir() -> str:
    try:
        import nvidia  # type: ignore

        return list(nvidia.__path__)[0]
    except Exception:
        return ""


def load_nvrtc() -> ctypes.CDLL:
    cands = []
    nd = find_nvidia_dir()
    if nd:
        cands += glob.glob(os.path.join(nd, "cuda_nvrtc", "lib", "libnvrtc.so*"))
    cands += ["libnvrtc.so", "libnvrtc.so.12"]
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise SystemExit("libnvrtc not found: pip install nvidia-cuda-nvrtc-cu12")


def find_ptxas() -> str:
    nd = find_nvidia_dir()
    for c in [os.path.join(nd, "cuda_nvcc", "bin", "ptxas") if nd else "", "/usr/local/cuda/bin/ptxas", "ptxas"]:
        if c and (os.path.exists(c) or c == "ptxas"):
            return c
    return ""


def nvrtc_compile(lib: ctypes.CDLL, src: str, name: str, opts: list[str]) -> tuple[bool, str, str]:
    prog = ctypes.c_void_p()
    lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(), 0, None, None)
    c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    res = lib.nvrtcCompileProgram(prog, len(opts), c_opts)
    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    lib.nvrtcGetProgramLog(prog, log)
    ptx = ""
    if res == 0:
        ptx_size = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        buf = ctypes.create_string_buffer(ptx_size.value)
        lib.nvrtcGetPTX(prog, buf)
        ptx = buf.value.decode()
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return res == 0, log.value.decode(errors="replace"), ptx


def summarize_ptxas(stderr: str) -> list[str]:
    """ptxas -v lines -> 'kernel: N regs, S B smem, spills'"""
    out, name = [], None
    for line in stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            name = m.group(1)
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and name:
            if m.group(2) != "0" or m.group(3) != "0":
                out.append(f"{name[:60]:60s} SPILLS: {m.group(2)} B stores / {m.group(3)} B loads, stack {m.group(1)} B")
        m = re.search(r"Used (\d+) registers(?:, used \d+ barriers)?(?:, (\d+) bytes smem)?", line)
        if m and name:
            out.append(f"{name[:60]:60s} {m.group(1):>4s} regs  {m.group(2) or '0':>6s} B static smem")
            name = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "ptx"))
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    lib = load_nvrtc()
    ptxas = find_ptxas()
    os.makedirs(args.out, exist_ok=True)
    files = args.files or sorted(glob.glob(os.path.join(ROOT, "src", "kernels", "*.cu")))
    opts = [f"--gpu-architecture=compute_{args.arch.split('_')[1]}", "-std=c++17",
            f"-I{os.path.join(ROOT, 'include')}", f"-I{os.path.join(ROOT, 'src', 'kernels')}", "-lineinfo"]
    # CUDA headers NVRTC needs (cuda_fp16.h, mma.h, cuda/std/*): from the pip wheels or a toolkit install
    nd = find_nvidia_dir()
    for inc in [os.path.join(nd, "cuda_runtime", "include") if nd else "", os.path.join(nd, "cuda_cccl", "include") if nd else "",
                os.path.join(nd, "cuda_nvcc", "include") if nd else "",
                os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include")]:
        if inc and os.path.isdir(inc):
            opts.append(f"-I{inc}")
    failed = 0
    for f in files:
        rel = os.path.relpath(f, ROOT)
        with open(f) as fh:
            src = fh.read()
        ok, log, ptx = nvrtc_compile(lib, src, os.path.basename(f), opts)
        if not ok:
            failed += 1
            print(f"FAIL {rel}\n{log}")
            continue
        if args.verbose and log.strip():
            print(log)
        ptx_path = os.path.join(args.out, os.path.basename(f) + ".ptx")
        with open(ptx_path, "w") as fh:
            fh.write(ptx)
        n_kernels = ptx.count(".entry ")
        line = f"ok   {rel:40s} {n_kernels} kernel(s)"
        if ptxas:
            r = subprocess.run([ptxas, f"-arch={args.arch}", "-v", "-O3", ptx_path, "-o", ptx_path + ".cubin"], capture_output=True, text=True)
            if r.returncode != 0:
                failed += 1
                print(f"FAIL ptxas {rel}\n{r.stderr}")
                continue
            print(line)
            for s in summarize_ptxas(r.stderr):
                print("       " + s)
            if "spill stores" in r.stderr and not re.search(r"\b0 bytes spill stores", r.stderr):
                print("       note: register spills present (see ptxas -v output with --verbose)")
        else:
            print(line + "  (ptxas not found; PTX only)")
    print(f"\n{len(files) - failed}/{len(files)} files compiled for {args.arch}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
