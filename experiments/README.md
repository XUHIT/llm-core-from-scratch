# Experiments

把可复现的实验代码、配置和轻量脚本放在这里。核心模块实现放在 `src/llm_core_from_scratch/`，这里主要放验证、对比和训练入口。

建议每个实验单独建一个子目录，例如：

```text
experiments/
  001_attention_from_scratch/
    README.md
    train.py
    config.yaml
```

大文件数据集和模型权重不要放在这里，放到服务器项目目录下的 `data/` 或 `checkpoints/`。
