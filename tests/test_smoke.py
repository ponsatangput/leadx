import leadx


def test_import():
    # __version__ derive จาก package metadata (importlib) — ตรง pyproject เสมอ
    assert leadx.__version__ and leadx.__version__[0].isdigit()
