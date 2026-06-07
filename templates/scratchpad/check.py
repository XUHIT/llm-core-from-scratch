"""本地草稿区快速检查脚本。

用法：
    python scratchpad/check.py

它会运行 current.py 和 reference.py 里的 main()。
如果两个 main() 都返回了结果，会优先用 torch.allclose 做数值对齐检查，再退回 NumPy。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_module(path: Path) -> ModuleType:
    """从文件路径加载一个 Python 模块。"""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(file_name: str) -> Any:
    """运行某个文件里的 main()，并返回 main() 的结果。"""
    path = ROOT / file_name
    if not path.exists():
        print(f"跳过 {file_name}：文件不存在")
        return None

    module = load_module(path)
    main = getattr(module, "main", None)
    if main is None:
        print(f"跳过 {file_name}：没有定义 main()")
        return None

    print(f"\n=== 运行 {file_name} ===")
    return main()


def try_allclose(actual: Any, expected: Any) -> bool | None:
    """尽量比较两个返回值是否数值接近。

    返回：
        True: 可以比较，且结果接近
        False: 可以比较，但结果不接近
        None: 当前返回值类型不适合自动比较
    """
    try:
        import torch

        if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            return bool(torch.allclose(actual, expected, atol=1e-8, rtol=1e-6))
    except ModuleNotFoundError:
        pass

    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    try:
        return bool(np.allclose(actual, expected, atol=1e-8, rtol=1e-6))
    except (TypeError, ValueError):
        return None


def main() -> None:
    """运行 current/reference，并做最小对照。"""
    current = run_main("current.py")
    reference = run_main("reference.py")

    print("\n=== 返回值概览 ===")
    print("current:", repr(current))
    print("reference:", repr(reference))

    if current is None or reference is None:
        print("\n没有同时拿到 current/reference 返回值，本次只做运行检查。")
        return

    is_close = try_allclose(current, reference)
    if is_close is True:
        print("\nallclose 通过：current 和 reference 数值一致。")
    elif is_close is False:
        raise AssertionError("allclose 失败：current 和 reference 数值不一致。")
    else:
        print("\n返回值类型不适合自动 allclose，请在文件里手写 assert。")


if __name__ == "__main__":
    main()
