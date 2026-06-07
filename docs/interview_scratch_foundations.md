# 第一阶段：面试高频基础手撕题库

这一阶段只解决最基础、最高频、最容易在面试中要求现场推导或手写的 LLM 核心计算题。

不使用真实数据集，不追求训练效果。所有正确性检验都使用小规模模拟数值和随机张量，重点观察 shape 变化、数值计算路径和梯度一致性。

## 交付标准

每个题目默认包含：

- PyTorch primary implementation
- 小数字可读样例
- 随机 synthetic tensor 数值对齐
- shape 检查
- 必要时 backward / autograd / gradcheck 对照
- NumPy reference implementation（辅助公式展开和数值对齐，不作为第一优先级）
- 常见错误说明

## P0：必须先做的基础

这些是后续所有 LLM 模块的地基。

| 编号 | 题目 | 重点 |
|---|---|---|
| 001 | Tensor shape / reshape / transpose / broadcasting | 理解 `[B, T, D]`、`[B, H, T, Dh]` 等维度变化 |
| 002 | Matrix multiplication | `matmul`、batch matmul、线性层的 shape |
| 003 | Linear regression | 最小二乘、MSE、闭式解与梯度下降 |
| 004 | K-means clustering | 距离矩阵、cluster assignment、centroid update |
| 005 | Linear layer forward/backward | `y = xW + b`，手写 `dx/dW/db` |
| 006 | Embedding lookup forward/backward | one-hot 等价、重复 token 的梯度累加 |
| 007 | Sigmoid / Tanh / ReLU / GELU / SiLU | 激活函数数值与导数 |
| 008 | Softmax forward/backward | 稳定 softmax、按哪个维度归一化 |
| 009 | Cross entropy forward/backward | logits 到 loss，`softmax - one_hot` |
| 010 | KL divergence | `KL(p || q)`、logits/log-probs、reduction |
| 011 | MSE loss forward/backward | 基础损失函数和 reduction |
| 012 | BatchNorm | 训练/推理模式差异、running mean/var 更新、手写 forward |
| 013 | Dropout | 训练时 mask 缩放 `x / (1-p)`、推理时恒等映射、反向传播时 mask 不变 |

## P1：LLM 核心层

这些是现代 decoder-only LLM 的常见手撕题。

| 编号 | 题目 | 重点 |
|---|---|---|
| 014 | LayerNorm forward/backward | 按最后一维归一化，均值方差和梯度 |
| 015 | RMSNorm forward/backward | 不减均值，Llama/Qwen 常用 |
| 016 | Causal mask | 上三角/下三角、mask 方向、广播 shape |
| 017 | Scaled dot-product attention | `QK^T / sqrt(d)`、mask、softmax、`PV` |
| 018 | Multi-head attention | QKV projection、split heads、merge heads |
| 019 | Grouped-query attention | MHA/MQA/GQA 的 head 映射和 KV 复用 |
| 020 | RoPE | pairwise rotate、position 维度、Q/K 旋转 |
| 021 | FFN / MLP | up projection、activation、down projection |
| 022 | SwiGLU FFN | gate/up/down projection，现代 LLM FFN |
| 023 | LM Head / Weight Tying | 输入 Embedding 和输出投影共享权重、`W @ W^T` 的梯度传播 |
| 024 | ALiBi 位置编码 | 不需要可学习参数的位置偏置、attention score 加线性递减 bias |

## P2：推理与最小模型闭环

这些用于把基础模块拼成可运行的最小 LLM，但仍然用 synthetic tensor 验证为主。

| 编号 | 题目 | 重点 |
|---|---|---|
| 025 | KV cache prefill/decode | full forward 与逐 token decode 数值一致 |
| 026 | Decoder block | RMSNorm + attention + residual + FFN/SwiGLU |
| 027 | Tiny GPT forward | token embedding、block stack、lm head |
| 028 | Greedy / temperature sampling | logits 缩放与采样分布变化 |
| 029 | Top-k sampling | 只保留概率最高的 k 个 token 后重归一化 |
| 030 | Top-p / nucleus sampling | 按累计概率阈值保留动态候选集合 |
| 031 | AdamW | Adam 与 decoupled weight decay 的区别 |
| 032 | Gradient clipping | global norm clipping |
| 033 | SFT loss | next-token cross entropy、label shift、assistant/padding mask |
| 034 | Beam search | beam size、log-prob 累加、终止条件、与 greedy/sampling 的对比 |
| 035 | Gradient accumulation | `loss / accum_steps`、optimizer step 时机、等效 batch size 计算 |

## 暂不放入第一阶段

这些重要，但不是第一批最基础面试手撕题。

- BPE tokenizer
- LoRA
- QLoRA
- DPO loss
- GRPO loss
- GSPO loss
- DAPO loss
- MTP loss
- Cross-attention
- MoE architecture
- MoE router
- MoE load balancing loss
- Speculative decoding
- PPO / GAE / GRPO / RLOO
- MLA / YaRN / NTK RoPE scaling

## Backlog：第二阶段候选题

### K-means clustering

K-means 不是 LLM 专属模块，但属于经典高频手撕基础算法，适合用小型二维点集验证 assignment 和 centroid update。

建议拆成：

1. pairwise squared distance
2. nearest centroid assignment
3. centroid update
4. empty cluster 处理
5. objective / inertia 计算
6. 多轮迭代收敛

最小 synthetic shape：

```text
N = 6 points
D = 2
K = 2 clusters
```

第一版优先做 PyTorch 向量化实现；必要时补 NumPy reference 做公式展开和数值对齐，不依赖真实数据。

### Top-k / top-p sampling

采样属于推理常见手撕题，放在 P2。实现时要拆开 top-k 和 top-p，不要只写一个笼统 sampling 函数。

建议拆成：

1. logits temperature scaling
2. softmax probabilities
3. top-k filtering and renormalization
4. top-p cumulative probability filtering
5. min tokens to keep
6. deterministic seed sampling

最小 synthetic shape：

```text
B = 2
V = 8 vocab size
```

### LoRA

面试中常见，适合在 `Linear layer`、`Embedding` 和基础训练循环之后实现。建议拆成：

1. frozen base weight 与 trainable low-rank adapter
2. `W_eff = W + scale * B @ A`
3. LoRA linear forward
4. LoRA 参数量计算
5. merge / unmerge 权重
6. 只训练 LoRA 参数的梯度检查

最小 synthetic shape：

```text
B = 2
T = 3
in_features = 4
out_features = 6
rank = 2
```

### QLoRA

QLoRA 是 LoRA 的量化微调版本，适合在 LoRA 和基础量化理解之后做。它不是第一阶段基础题，但对单卡 3090 很实用。

建议拆成：

1. frozen 4-bit quantized base model
2. trainable LoRA adapters
3. NF4 quantization intuition
4. double quantization intuition
5. dequantize-for-compute 的 forward 路径
6. 只更新 adapter 参数
7. 与普通 LoRA 的显存差异

最小 synthetic shape：

```text
in_features = 4
out_features = 6
rank = 2
quant_bits = 4
```

第一版只做 toy quantized linear + LoRA adapter 的数值路径，不做 bitsandbytes 工程复刻。

### Cross-attention

交叉注意力是 encoder-decoder、RAG reranker、multimodal projector 中常见题。第一阶段优先 self-attention，cross-attention 放在 attention 基础稳定后实现。

建议拆成：

1. query 来自 decoder hidden states
2. key/value 来自 encoder 或外部 memory
3. `Tq != Tk` 时的 score shape
4. padding mask 与 causal mask 的区别
5. multi-head cross-attention

最小 synthetic shape：

```text
B = 2
Tq = 3
Tk = 5
D = 8
H = 2
```

### MTP loss

这里先按 multi-token prediction loss 记录。它适合放在 cross entropy 和 Tiny GPT forward 之后，实现为训练目标专题。

建议拆成：

1. next-token prediction 与 multi-token prediction 的区别
2. 多个未来 offset 的 target 构造
3. logits shape: `[B, T, K, V]` 或多 head logits
4. 每个 offset 的 cross entropy
5. loss 加权与 reduction
6. ignore index / sequence 边界处理

最小 synthetic shape：

```text
B = 2
T = 5
K = 3 future tokens
V = 7 vocab size
```

### DPO loss

DPO 是偏好对齐中最适合先手撕的 loss：不需要 value model，也不需要在线采样。建议第二阶段在 cross entropy、KL divergence 和 Tiny GPT forward 之后实现。

建议拆成：

1. chosen / rejected response 的 log probability
2. policy logprob 与 reference logprob
3. implicit reward: `log pi(y|x) - log pi_ref(y|x)`
4. preference margin
5. `-log sigmoid(beta * margin)`
6. beta 对 loss 曲线的影响
7. sequence-level logprob 的 mask 与 reduction

最小 synthetic shape：

```text
B = 2 pairs
Tc = 4 chosen tokens
Tr = 4 rejected tokens
V = 8 vocab size
```

第一版只做 synthetic logits/logprobs 的 DPO loss 数值验证，不做真实偏好数据训练。

### SFT loss

SFT loss 适合放进第一阶段 P2：它不是新损失，本质是 decoder-only LM 的 next-token cross entropy，加上 label shift、padding mask 和 assistant-only mask。

建议拆成：

1. logits shape: `[B, T, V]`
2. labels shape: `[B, T]`
3. next-token shift: `logits[:, :-1]` 对齐 `labels[:, 1:]`
4. padding mask / attention mask
5. assistant-only loss mask
6. `ignore_index`
7. token-level loss 到 batch loss 的 reduction

最小 synthetic shape：

```text
B = 2
T = 5
V = 7 vocab size
```

第一版只用 synthetic logits、labels 和 loss mask 验证 shift、gather、mask、reduction 是否正确。

### PPO loss

PPO loss 是 RLHF 里常见但实现更重的题，建议第三阶段做 toy implementation。第一版先记录损失函数本身。

建议拆成：

1. old policy logprob 与 new policy logprob
2. ratio: `exp(logp_new - logp_old)`
3. advantage
4. clipped surrogate objective
5. value loss
6. entropy bonus
7. KL penalty / early stopping intuition
8. token-level mask 与 sequence-level reward

最小 synthetic shape：

```text
B = 2
T = 5
V = 7
```

第一版 PPO 不做 rollout，不做 reward model，只验证 clipped policy loss、value loss、entropy bonus 的张量计算。

### GAE

GAE 是 PPO 里最常见的 advantage estimation 方法，适合和 PPO loss 同阶段实现。

建议拆成：

1. rewards
2. values
3. bootstrap next value
4. TD residual: `delta_t = r_t + gamma * V_{t+1} - V_t`
5. reverse-time advantage recursion
6. `gamma` 和 `lambda` 的作用
7. terminal mask / padding mask
8. returns: `advantage + value`

最小 synthetic shape：

```text
B = 2
T = 5
```

第一版只用 synthetic rewards、values、dones 和 masks 验证 GAE 递推，不做环境 rollout。

### GRPO loss

GRPO loss 是 GRPO algorithm 中最核心的手撕部分。建议第三阶段实现，但先单独记录损失函数，避免和采样、reward 计算、训练循环混在一起。

建议拆成：

1. grouped responses 的 logprob
2. old policy logprob 与 current policy logprob
3. reference policy logprob
4. group-normalized advantage
5. ratio: `exp(logp_current - logp_old)`
6. clipped surrogate objective
7. KL regularization
8. token mask 与 response length normalization

最小 synthetic shape：

```text
B = 2 prompts
G = 4 responses per prompt
T = 6
V = 9
```

第一版只用 synthetic logprobs、advantages 和 masks 验证 GRPO loss，不做真实 rollout。

### GSPO loss

GSPO loss 是 sequence-level policy optimization 方向的题，和 GRPO 的关键差异是 importance ratio、reward assignment 和 clipping 更偏 sequence level，而不是逐 token level。

建议拆成：

1. grouped responses
2. sequence logprob
3. length-normalized sequence ratio
4. group baseline / advantage
5. sequence-level clipping
6. optional KL regularization
7. 与 GRPO token-level ratio 的对比

最小 synthetic shape：

```text
B = 2 prompts
G = 4 responses per prompt
T = 6
```

第一版只用 synthetic sequence logprobs、advantages 和 masks 验证 loss，不做真实 rollout。

### DAPO loss

DAPO loss 可以视为 GRPO 系列的改进型专题，重点在 token-level policy gradient、decoupled clipping、dynamic sampling 和 overlong reward shaping。

建议拆成：

1. token-level policy gradient loss
2. Clip-Higher / asymmetric clipping
3. dynamic sampling 的有效 group 条件
4. overlong reward shaping
5. token mask 与 response length normalization
6. 与 GRPO loss 的差异

最小 synthetic shape：

```text
B = 2 prompts
G = 4 responses per prompt
T = 6
V = 9
```

第一版只验证 synthetic logprobs、advantages、length masks 和 clipping 的张量计算。

### GRPO algorithm

GRPO 是 reasoning/RL 方向热点，但不适合第一阶段。建议第三阶段在 DPO 和 PPO loss 理解后做 toy group-based implementation。

建议拆成：

1. prompt 分组采样：每个 prompt 生成 `G` 个 responses
2. group reward mean/std
3. relative advantage normalization
4. policy ratio
5. clipped objective
6. KL regularization
7. token-level mask
8. 不使用 value model 的区别

最小 synthetic shape：

```text
B = 2 prompts
G = 4 responses per prompt
T = 6
V = 9
```

第一版 GRPO 不做真实推理任务，只用 synthetic rewards、logprobs 和 masks 验证 group relative advantage 与 loss 计算。

### MoE architecture / load balancing loss

面试中可能出现，但第一阶段暂不展开实现。建议第二阶段拆成：

1. dense FFN 与 MoE FFN 的结构差异
2. router logits 与 router probabilities
3. top-1 / top-2 / top-k expert selection
4. token dispatch 与 expert combine
5. capacity factor 与 dropped tokens
6. expert load 统计
7. load balancing loss
8. z-loss / router stability intuition

最小 synthetic shape：

```text
B = 2
T = 4
D = 8
E = 4 experts
K = 1 or 2 selected experts
```

第一版 MoE 不做 expert parallel，不做真实大模型训练，只验证 routing、dispatch、combine 和负载均衡损失的数值计算。

### BatchNorm

BatchNorm 虽然现代 LLM 不用，但属于面试极高概率手撕题。建议在基础激活函数和线性层之后实现。

建议拆成：

1. 训练 forward: `(x - mean) / sqrt(var + eps) * gamma + beta`
2. 推理 forward: 使用 running_mean / running_var
3. backward: `dx`、`dgamma`、`dbeta` 的完整推导
4. running mean/var 的 momentum 更新
5. 与 LayerNorm 的维度差异对比

最小 synthetic shape：

```text
B = 2
D = 4
```

### Dropout

经典正则化技巧，面试常考 train/eval 模式差异和 mask 缩放逻辑。

建议拆成：

1. 训练 forward: `x * mask / (1 - p)`
2. 推理 forward: 恒等映射（不 dropout）
3. backward: mask 不变，梯度同样被 mask 并缩放
4. inverted dropout 公式推导
5. 与 `nn.Dropout` 行为对齐

最小 synthetic shape：

```text
B = 2
D = 4
p = 0.5
```

### LM Head / Weight Tying

几乎所有现代 LLM 都共享输入 Embedding 和输出投影权重，面试常问原因和梯度细节。

建议拆成：

1. Embedding 权重 shape: `[V, D]`
2. LM head 权重 shape: `[D, V]`（即 Embedding 的转置）
3. 共享权重时 forward: `logits = x @ W_embed.T`
4. 共享权重时 backward: 两路梯度如何累加
5. 不共享 vs 共享的参数量对比
6. 与 `nn.Linear` 参数化方式的转换

最小 synthetic shape：

```text
B = 2
T = 3
V = 8
D = 4
```

### ALiBi 位置编码

比 RoPE 更简单的位置编码方案，不需要可学习参数，面试容易作为"不用 RoPE 怎么做位置编码"的备选答案。

建议拆成：

1. 构造 attention bias: `slope * distance`
2. 不同 head 使用不同 slope
3. bias 加到 attention scores 的方式
4. 与 RoPE 的对比（复杂度、长度外推、性能）
5. 与 causal mask 的叠加

最小 synthetic shape：

```text
B = 1
H = 2
T = 4
D = 8
```

### Beam search

greedy/top-k/top-p 都有了，缺 beam search 不完整。面试常问 beam size 的影响。

建议拆成：

1. 初始化: 从 `<bos>` 开始，beam 中维护 k 个候选序列
2. 每步扩展: 每个候选取 top-k 个后继，共 k*V 个，取 top-k
3. log-prob 累加与归一化
4. 终止条件: EOS token 或 max length
5. beam size 对生成质量和速度的影响
6. 与 greedy (beam=1) 的对比

最小 synthetic shape：

```text
B = 1
V = 8
beam_size = 2
max_len = 4
```

### Gradient accumulation

模拟大 batch 训练的标准技巧，理解 loss 除法和 optimizer step 时机是关键。

建议拆成：

1. `loss = batch_loss / accum_steps`
2. `loss.backward()` 梯度累加
3. 每 `accum_steps` 步执行 `optimizer.step()` + `optimizer.zero_grad()`
4. 等效 batch size = `per_device_batch * accum_steps`
5. 与普通训练的梯度等价性验证

最小 synthetic shape：

```text
total_samples = 8
per_step_batch = 2
accum_steps = 4
```


## 测试样例原则

优先使用极小 shape：

```text
B = 1 or 2
T = 2 or 4
D = 4 or 8
H = 2
Dh = D / H
```

每个模块至少保留一个“人眼能看懂”的小数字样例，例如：

```python
x = np.array([
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 1.0, 0.0, 3.0]]
])
```

随机测试只用于补充覆盖，必须固定 seed。

## 推荐开发顺序

1. `LinearRegression`
2. `KMeans`
3. `Linear`
4. `Embedding`
5. `BatchNorm`
6. `Dropout`
7. `Softmax`
8. `CrossEntropy`
9. `KLDivergence`
10. `MSE`
11. `LayerNorm`
12. `RMSNorm`
13. `CausalMask`
14. `ScaledDotProductAttention`
15. `MultiHeadAttention`
16. `RoPE`
17. `GroupedQueryAttention`
18. `ALiBi`
19. `KVCache`
20. `FFN`
21. `SwiGLU`
22. `LMHead` / Weight Tying
23. `DecoderBlock`
24. `TinyGPTForward`
25. `GreedySampling`
26. `TopKSampling`
27. `TopPSampling`
28. `BeamSearch`
29. `AdamW`
30. `GradientClipping`
31. `GradientAccumulation`
32. `SFTLoss`
