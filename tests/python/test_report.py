from report import table


def test_markdown_table():
    rows = [
        {"kernel": "naive", "dtype": "f32", "M": "512", "N": "512", "K": "512", "pct_cublas": "3.2"},
        {"kernel": "naive", "dtype": "f32", "M": "1024", "N": "1024", "K": "1024", "pct_cublas": "2.9"},
        {"kernel": "cublas", "dtype": "f32", "M": "512", "N": "512", "K": "512", "pct_cublas": "100"},
        {"kernel": "mma_ptx", "dtype": "f16", "M": "512", "N": "512", "K": "512", "pct_cublas": "88"},
    ]
    md = table(rows, "pct_cublas", "f32")
    lines = md.splitlines()
    assert lines[0] == "| kernel | 512 | 1024 |"
    assert "| naive | 3.2 | 2.9 |" in lines
    assert "| cublas | 100.0 | - |" in lines
    assert "mma_ptx" not in md
