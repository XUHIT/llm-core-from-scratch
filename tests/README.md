# Tests

测试优先覆盖 NumPy 与 PyTorch 两套实现的数值一致性。

建议每个核心模块至少包含：

- NumPy 版本的形状和数值测试。
- PyTorch 版本的形状、梯度和 GPU 可用性测试。
- NumPy 与 PyTorch 在固定输入下的输出对齐测试。
