# Contributing to hip-ml-toolkit

## Development Setup

```bash
git clone https://github.com/Dallas-AI564/hip-ml-toolkit.git
cd hip-ml-toolkit
pip install -e ".[dev]"
```

## Code Style

- Python: Ruff formatter + linter
- C++/HIP: 4-space indent, K&R braces
- Max line length: 100

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Adding HIP Kernels

See `src/hipml/kernels/` for examples. Each kernel needs:
1. `.cpp` file with HIP kernel + `extern "C"` launcher
2. Python wrapper in `src/hipml/kernels/__init__.py`
3. Test in `tests/test_kernels.py`

## Pull Request Process

1. Fork → feature branch → PR
2. CI must pass (lint + test)
3. At least 1 review required

## License

MIT — contributions are MIT-licensed.
