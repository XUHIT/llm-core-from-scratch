# Algorithms

这里是项目主线：每道面试手撕题一个独立文件夹。

默认结构：

```text
xxx_algorithm_name/
├── README.md
├── numpy_impl.py
└── torch_impl.py
```

- `README.md`: 公式、shape 变化、手推重点、常见坑。
- `numpy_impl.py`: NumPy 手写实现。P0 基础题以它为主。
- `torch_impl.py`: PyTorch 对照实现，用于数值验证、autograd 对照和后续组合。

第一阶段只使用小数字样例和 synthetic tensors，不使用真实数据集。
