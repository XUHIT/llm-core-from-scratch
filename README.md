# llm-core-from-scratch

PyTorch 实现的大语言模型核心模块（NumPy 版本后补）。

目标：逐模块手写 LLM 关键组件，每个模块配知识点讲解和代码解析。

## 实现原则

**先用 PyTorch 跑通、验证、理解，再考虑用 NumPy 复刻底层细节。**

原因：NumPy 偏底层，需要手动写反向传播；PyTorch 能快速验证正确性、观察梯度、上 GPU 实验。等 PyTorch 版本稳定后，再补 NumPy 版本加深理解。

## 目录

```
src/llm_core_from_scratch/
├── common/                     # 共享工具
├── p0_fundamentals/            # 基础组件
│   ├── 001_linear_layer/
│   │   ├── torch_impl.py       # PyTorch 实现
│   │   └── README.md           # 知识点 + 代码解析 + 延伸
│   └── ...
├── p1_llm_core/                # LLM 核心层
│   └── ...
└── p2_inference/               # 推理与模型闭环
    └── ...

docs/
└── interview_scratch_foundations.md   # 完整题库与路线图

experiments/    # 实验代码和配置
notes/          # 实验笔记
results/        # 图表、指标、运行结果摘要
tests/          # 测试（形状 + 数值 + 双实现对齐，后补）
```

## 安装

```bash
pip install -e ".[torch,dev]"
```
