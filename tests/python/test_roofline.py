import csv

import pytest
from roofline import Peaks, analyse, attainable, gemm_intensity, lookup_device


def test_intensity_square_fp32():
    # 2n^3 / (3 n^2 * 4) = n / 6
    assert gemm_intensity(1024, 1024, 1024, 4) == pytest.approx(1024 / 6)
    # fp16 inputs, fp32 output: 2n^3 / (2n^2*2 + n^2*4) = n/4
    assert gemm_intensity(1024, 1024, 1024, 2) == pytest.approx(256.0)


def test_binding_roof():
    p = Peaks(fp32_tflops=20, fp16_tc_tflops=320, hbm_gbs=2000)
    assert p.ridge_fp32 == pytest.approx(10.0)
    assert attainable(5.0, p, fp16=False) == (10.0, "memory")
    assert attainable(50.0, p, fp16=False) == (20.0, "fp32")
    assert attainable(50.0, p, fp16=True) == (100.0, "memory")
    assert attainable(500.0, p, fp16=True) == (320.0, "fp16 tensor core")


def test_device_lookup_and_analyse(tmp_path):
    p = lookup_device("NVIDIA A100-SXM4-80GB")
    assert p is not None and p.fp16_tc_tflops == 312
    assert lookup_device("some future gpu") is None
    path = tmp_path / "b.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("kernel,dtype,M,N,K,batch,split_k,ms,tflops,gbs,pct_cublas,pct_peak,arith_intensity,grid,block,smem_bytes".split(","))
        w.writerow(["double_buffered", "f32", 4096, 4096, 4096, 1, 1, 10.0, 13.7, 20.0, 80.0, 70.0, 682.7, "32x32x1", 256, 16384])
        w.writerow(["mma_ptx", "f16", 4096, 4096, 4096, 1, 1, 0.6, 229.0, 224.0, 90.0, 73.0, 1024.0, "32x32x1", 256, 56832])
    from roofline import load_rows
    pts = analyse(load_rows(str(path)), p)
    assert pts[0]["roof"] == "fp32" and pts[0]["pct_roofline"] == pytest.approx(100 * 13.7 / 19.5)
    assert pts[1]["roof"] == "fp16 tensor core" and pts[1]["pct_roofline"] == pytest.approx(100 * 229 / 312)
