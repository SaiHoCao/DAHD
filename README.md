# DAHD: Difficulty-Adaptive Hybrid Drafting for Speculative Decoding

## 1. 项目思路与动机

### 核心观察

Token 的可预测性呈**双模态/多模态分布**。通过对 Qwen3-8B + EAGLE-3 进行 acceptance rate profiling，我们发现 draft model 的预测置信度明显聚集在两个模式：

- **Easy tokens** (μ=0.814, σ=0.142, 权重 39.7%): 几乎必然被正确预测
- **Hard tokens** (μ=0.335, σ=0.147, 权重 60.3%): 预测困难，acceptance rate 低

### 核心问题

EAGLE-3 等纯 AR 方法对所有 token 使用相同的自回归策略，浪费了 Easy tokens 的加速潜力。对于高确定性 token，单次前向即可并行预测多个位置，无需逐步生成。

### DAHD 方案（v2：三模态 + Gumiho 并行分支）

根据 token 难度动态切换草稿模式和长度：

| Token 难度 | 草稿模式 | Draft 长度 | 原理 |
|------------|----------|-----------|------|
| Easy (top1_prob > 0.75) | Gumiho 并行 | K=4 | O(1) 延迟；输入 fc(cat(embed(t+1), hidden_t))，条件预测 t+2..t+5 |
| Medium (0.5 < prob ≤ 0.75) | EAGLE-3 AR | K=3 | 中等精度，适度步长 |
| Hard (top1_prob ≤ 0.5) | EAGLE-3 AR | K=2 | 精确打击；K=2 避免 tail 指数衰减 |

**v2 关键改进（相对 v1）**:
- 并行分支从早期 Medusa 升级为 **Gumiho-style**：输入 `fc(cat(embed(target_next), hidden_t))`，比仅用 `hidden_t` 信息量更丰富
- 三模态路由（原 v1 为二模态）
- Hard 时 K=2（原 v1 固定 K=5，后验证指数衰减在位置 3+ 接受率仅 ~7%）
- 训练数据仅使用 **生成阶段** token（跳过 prompt 阶段）
- 训练 loss 改为指数衰减权重 `[1.0, 0.8, 0.64, 0.512]`
- LR scheduler 改为每 optimizer step 调用（原 v1 每 epoch 调用一次，导致 schedule 几乎不 decay）

---

## 2. 项目结构

```
dahd_speculative_decoding/
├── src/                                # 核心模块
│   ├── config.py                       # ExperimentConfig 数据类
│   ├── drafters/                       # Draft 模型实现
│   │   ├── base.py                     # SpeculativeDrafter 基类
│   │   ├── dahd_draft_module.py        # DAHD 核心架构（双分支 + Router）
│   │   ├── eagle_baseline.py           # EAGLE-3 AR baseline
│   │   ├── parallel_baseline.py        # Gumiho-style 并行 baseline
│   │   └── router.py                   # DifficultyRouter (MLP + Rule-based)
│   ├── metrics/                        # 指标采集与存储
│   │   ├── schema.py                   # PerTokenMetrics, PerSequenceMetrics
│   │   ├── collector.py                # 指标收集器
│   │   ├── storage.py                  # MetricsStore (JSONL/Parquet/CSV)
│   │   └── validator.py               # 指标验证工具
│   ├── benchmarks/                     # Benchmark 框架
│   │   ├── harness.py                  # FairBenchmarkHarness
│   │   ├── statistical_tests.py        # 双模态检验、Bootstrap CI
│   │   └── latency_measurement.py      # CUDATimer, 自适应 warmup
│   ├── analysis/                       # 分析管道
│   │   ├── pipeline.py                 # AnalysisPipeline（编排所有分析）
│   │   ├── statistics.py               # 统计计算工具
│   │   └── visualization.py           # 可视化绘图
│   ├── ablations/                      # 消融实验
│   │   ├── ablation_config.py          # 5 种预定义消融配置
│   │   └── runner.py                   # AblationExperiment, AblationSuite
│   └── utils/                          # 通用工具
│       ├── device_utils.py             # GPU 信息、确定性设置
│       └── logging_utils.py            # ExperimentLogger
├── experiments/                        # 实验脚本
│   ├── phase1_profiling/               # Phase 1: 假设验证
│   │   ├── run_eagle3_full_profiling.py    # 完整 EAGLE-3 acceptance profiling
│   │   ├── run_eagle3_profiling.py         # EAGLE-3 profiling（简化版）
│   │   ├── analyze_bimodal.py              # 双模态分析（GMM + Dip Test）
│   │   ├── run_acceptance_profiling.py     # 通用 acceptance profiling
│   │   └── run_bimodal_test.py             # 双模态假设检验
│   ├── phase2_architecture/            # Phase 2: 模型训练
│   │   ├── generate_training_data.py       # 训练数据生成
│   │   ├── train_hydra_parallel.py         # Hydra Parallel Branch 训练
│   │   ├── train_gumiho_heads.py           # Gumiho Parallel Heads 训练
│   │   └── train_dual_branch.py            # 双分支联合训练
│   ├── phase4_e2e/                     # Phase 4: 端到端对比
│   │   ├── run_e2e_comparison.py           # 端到端评估
│   │   ├── run_dahd_e2e_eval.py            # DAHD 端到端评估
│   │   └── run_e2e_benchmark.py            # 完整方法对比 benchmark
│   └── phase5_ablations/               # Phase 5: 消融实验
│       └── run_all_ablations.py            # 全部消融实验
├── results/                            # 实验结果
│   ├── phase1_results_v2/              # EAGLE-3 真实 acceptance profiling 结果
│   │   ├── bimodal_analysis.json           # GMM/Dip Test 数值结果
│   │   ├── acceptance_distribution.png     # 置信度分布图
│   │   ├── per_position_acceptance.png     # 逐位置 acceptance 图
│   │   ├── per_token_metrics.jsonl         # 逐 token 指标
│   │   └── per_step_summary.json           # 逐步汇总
│   ├── phase4_e2e/                   # 端到端对比结果
│   │   ├── e2e_comparison_v2.json            # 完整方法对比数据
│   │   ├── gumiho_training_log.json          # Gumiho 训练日志
│   │   ├── e2e_comparison_chart.png          # 对比柱状图
│   │   └── incremental_*.json                # 逐 prompt 增量结果
│   └── phase4_results/                 # 早期端到端结果
├── data/training/                      # 训练数据
│   ├── train/                          # 训练集（15 个分片）
│   └── val/                            # 验证集（5 个分片）
├── checkpoints/                        # 模型权重
│   ├── difficulty_router.pt                # Router MLP (4MB)
│   ├── hydra_parallel_branch_best.pt      # Hydra 最佳权重 (~1.4GB)
│   ├── hydra_parallel_branch_final.pt     # Hydra 最终权重 (~1.4GB)
│   ├── gumiho/                            # Gumiho 并行分支权重
│   │   ├── gumiho_best.pt                 # Gumiho 最佳权重
│   │   └── legacy_best.pt                 # 早期权重 (legacy)
│   └── validation_results.json             # 验证结果汇总
├── configs/                            # YAML 配置文件
├── pyproject.toml                      # 项目配置
├── requirements.txt                    # 依赖列表
└── README.md                           # 本文档
```

---

## 3. 已完成的工作

按时间线列出：

### 3.1 Phase 1: 假设验证

**目标**: 验证 speculative decoding 的 token acceptance rate 是否呈双模态分布。

**方法**: 使用完整 EAGLE-3（含 GQA attention, 32/8 heads）在 Qwen3-8B 上对 50 条 prompts 进行 acceptance rate profiling，采集 5295 个 token 的逐位置 acceptance 信息。

**关键发现**: Hartigan's Dip Test p=0.000，GMM 2-component BIC=312 远优于单模态 BIC=1382，强烈支持双模态假设。

### 3.2 Phase 2: 训练 Hydra Parallel Branch + Router

**目标**: 训练并行分支用于 Easy tokens 的快速 draft。

**历程**:
1. 首先尝试 Hydra 架构（d2t vocab 映射），发现 vocab 映射 bug 导致 acceptance = 0%
2. Bug 根因：d2t（draft-to-target）vocab 映射时 token ID 错位，实际预测 token 与验证 token 不匹配
3. 后续切换为标准 Gumiho 并行实现（full vocab 151936），彻底避免映射问题

### 3.3 Phase 2b: 训练 Gumiho Parallel Heads

**目标**: 用 full vocab（151936）训练 5 个 parallel heads，从 Qwen3-8B lm_head 初始化。

**结果**:
- 训练数据: ~145K samples, 3 epochs
- head_0 accuracy: 81.4%（val）
- head_1 accuracy: 12.9%（val）
- head_2-4 accuracy: 2-4%（val）

### 3.4 Phase 4: 端到端对比

**目标**: 在相同条件下对比 Vanilla AR / EAGLE-3 / Parallel (Gumiho) / DAHD 的端到端性能。

**设置**: 50 prompts, 128 tokens/prompt, Greedy decoding, H20 GPU

---

## 4. 实验结果

### 4.1 Phase 1: 双模态假设验证

| 指标 | 值 |
|------|-----|
| 评估数据 | 50 prompts, 5295 tokens |
| 模型组合 | Qwen3-8B + EAGLE-3 (完整 GQA attention) |
| Hartigan's Dip Test | p=0.000（强烈拒绝单模态） |
| GMM 2-component BIC | 311.7 |
| 单模态 BIC | 1382.0 |
| BIC 改善 | 1070.2 |
| Easy 模式 | μ=0.814, σ=0.142, 权重 39.7% |
| Hard 模式 | μ=0.335, σ=0.147, 权重 60.3% |

**逐位置 acceptance rate**:

| Position | Acceptance Rate |
|----------|----------------|
| pos0 | 54.7% |
| pos1 | 26.9% |
| pos2 | 13.7% |
| pos3 | 7.2% |
| pos4 | 3.6% |

**结果图表**:
- 置信度分布图: `results/phase1_results_v2/acceptance_distribution.png`
- 逐位置 acceptance: `results/phase1_results_v2/per_position_acceptance.png`

### 4.2 Phase 4: 端到端对比（v2 优化版）

**配置**: Qwen3-8B, H20 GPU, 50 prompts, max_new_tokens=128

| Method | Tok/s | Speedup | Avg Accepted/Step |
|--------|-------|---------|-------------------|
| Vanilla AR | 36.38 | 1.00x | — |
| EAGLE-3 (K=5) | 63.32 | 1.74x | 1.00 |
| Parallel (Gumiho, K=5) | 60.36 | 1.66x | 0.25 |
| **DAHD (3-modal, K=4/3/2)** | **66.92** | **1.84x** | 0.50 |

**DAHD 路由比例**: Easy 63.5%, Medium 22.1%, Hard 14.5%
- Easy → Parallel branch (K=4)
- Medium → EAGLE-3 AR (K=3)
- Hard → EAGLE-3 AR (K=2)

**数据来源**: `results/phase4_e2e/e2e_comparison_v2.json`

**对比图表**: `results/phase4_e2e/e2e_comparison_chart.png`

### 4.3 Gumiho Parallel 训练详情

训练配置:
- hidden_size: 4096, vocab_size: 151936
- num_heads: 5, num_layers: 1 (ResBlock)
- batch_size: 16, gradient_accumulation: 4 (effective bs=64)
- lr_resblock: 1e-3, lr_lm_head: 1e-5（差异化学习率）
- epochs: 3

训练过程中各 head 的验证集 accuracy:

| Epoch | head_0 | head_1 | head_2 | head_3 | head_4 |
|-------|--------|--------|--------|--------|--------|
| 1 | 78.7% | 1.7% | 2.0% | 1.9% | 2.7% |
| 2 | 80.9% | 6.0% | 2.3% | 2.0% | 2.7% |
| 3 | 81.4% | 12.9% | 4.0% | 2.8% | 2.9% |

**观察**: head_0 快速收敛（基本是 next-token prediction），head_1 在 epoch 3 开始起飞但远未饱和，head_2-4 几乎未训练出来——训练数据量严重不足。

---

## 5. 实现方案

### 5.1 AR Branch: EAGLE-3

- **架构**: 1-layer Transformer (GQA 32/8 heads, hidden_size=4096)
- **权重**: 预训练好的 `/mnt/nas1/hf/qwen3_8b_eagle3/` (~380M params)
- **生成方式**: 逐步 auto-regressive with KV cache
- **Draft 长度**:
  - EAGLE-3 baseline: K=5 (fixed)
  - DAHD dynamic: K_easy=4, K_medium=3, K_hard=2

### 5.2 Parallel Branch: Gumiho-style

- **架构**: 5 个独立 ParallelHead，每个包含 1 个 ResBlock + 共享 lm_head
- **词表**: full vocab 151936（从 Qwen3-8B lm_head 初始化）
- **生成方式**: 一次前向同时预测 5 个位置的 token
- **参数**: ResBlock lr=1e-3, lm_head lr=1e-5（差异化学习率）

### 5.3 Router: Difficulty-based Threshold

**Rule-based（当前使用）**:
```python
# 每一轮 speculative decoding:
target_out = target_model(input_ids)
top1_prob = softmax(target_out.logits[-1]).max()

if top1_prob > 0.8:
    draft_tokens = parallel_forward(hidden_states)  # Parallel, K=5
else:
    draft_tokens = eagle3_forward(hidden_states)  # AR, K=5
```

**MLP-based Router（已训练）**:
```python
class DifficultyRouter(nn.Module):
    def __init__(self, hidden_size=4096):
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, hidden):
        return torch.sigmoid(self.net(hidden))  # > 0.5 → easy
```

- 训练 label: Parallel Branch head_0 是否预测正确
- 训练集: 48952 positions (easy: 25700, hard: 23252)
- Accuracy: 71.8%
- 权重文件: `checkpoints/difficulty_router.pt` (4MB)

> **实现说明**: DAHD v2 端到端评估实现位于 `experiments/phase4_e2e/run_e2e_comparison.py`。
> `src/drafters/dahd_draft_module.py` 为模块化参考架构，适用于未来工程集成。

---

## 6. 基准与对比

| Baseline | 描述 | 实现文件 |
|----------|------|----------|
| Vanilla AR | 标准逐 token 生成，无 speculative decoding | `src/drafters/base.py` |
| EAGLE-3 (K=5) | 当前最强 AR speculative decoding | `src/drafters/eagle_baseline.py` |
| Parallel (Gumiho, K=5) | 标准并行推测解码（5 heads） | `src/drafters/medusa_baseline.py` (legacy name) |
| DAHD | 动态切换 AR/Parallel + Router | `src/drafters/dahd_draft_module.py` |

---

## 7. 实验配置

| 配置项 | 值 |
|--------|-----|
| Target Model | Qwen3-8B (bfloat16) |
| Target Model 路径 | `/mnt/nas1/hf/Qwen3-8B/` |
| EAGLE-3 Draft | qwen3_8b_eagle3 (~380M, 1-layer GQA) |
| EAGLE-3 路径 | `/mnt/nas1/hf/qwen3_8b_eagle3/` |
| GPU | NVIDIA H20 (98GB HBM) |
| 评估数据 | eagle_data.jsonl (397 samples, 实际用 50) |
| 生成长度 | 128 tokens/prompt |
| Decoding | Greedy (temperature=0) |
| Draft K (EAGLE-3) | 5 |
| Draft K (Parallel) | 5 |
| DAHD 阈值 | top1_prob = 0.8 |
| 随机种子 | 42 |

---

## 8. 关键设计决策

### 8.1 如何确定 Token 难度

**方法**: 使用 target model 输出的 top-1 probability 作为 difficulty proxy。

**原理**:
- top1_prob 高 → 目标模型对下一个 token 非常确定 → draft 模型也容易猜对 → Easy
- top1_prob 低 → 不确定性高 → 预测困难 → Hard

**实证基础**: GMM 分析显示 draft confidence 呈明显双模态分布:
- Easy 模式: μ=0.814, σ=0.142, 权重 39.7%
- Hard 模式: μ=0.335, σ=0.147, 权重 60.3%

**阈值选择**: 当前使用 top1_prob = 0.8 作为切换点（CDF 图中 Easy 区域的左边界）。

**进阶方案**（待实现）:
- DifficultyProbe: 轻量 MLP 直接从 hidden states 预测难度
- EMA 融合: `difficulty = 0.6 × probe_confidence + 0.4 × EMA(recent_acceptance_rate)`

### 8.2 如何选择并行 vs 自回归模式

**选择逻辑**:

| 条件 | 选择 | 原因 |
|------|------|------|
| Easy (top1_prob > 0.8) | Parallel (Gumiho) | 1 次前向生成 K 个 token，延迟低 |
| Hard (top1_prob ≤ 0.8) | AR (EAGLE-3) | 精确度高，避免浪费验证资源 |

**为什么并行适合 Easy**:
- 并行模式 O(1) 延迟（1 次前向 → K 个 token），但每个位置独立预测 → acceptance 依赖各位置独立准确率
- Easy tokens 的 head_0 accuracy = 81.4% → 大概率接受第 1 个 → 净收益 > 0

**为什么 AR 适合 Hard**:
- AR 模式 O(K) 延迟（K 次前向），但每步有完整上下文 → acceptance 更高
- Hard tokens 用 AR 且缩短 K → 精准打击，减少后面位置的无效 draft

### 8.3 Router 工作原理

**当前方案 (Rule-based)**:

每一步 speculative decoding 时，先获取 target model 前向的 logits，计算 top-1 概率。若超过阈值 0.8，使用 Gumiho 并行生成；否则使用 EAGLE-3 自回归生成。

**训练版 Router (MLP-based)**:

使用 target model 最后一层 hidden states (dim=4096) 作为输入，通过 2-layer MLP (4096→256→1) 输出 easy/hard 二分类概率。训练 label 为 Parallel Branch head_0 是否预测正确。

**开销分析**: Router 本身几乎无开销（一个 4096→256→1 的 MLP，约 1M 参数），在 target model 前向之后即可获取 hidden states 进行判断。

### 8.4 Draft Length 动态选择

**当前方案**: 模式固定 K
- Parallel: K=5（等于 parallel heads 数量）
- AR: K=5（与 EAGLE-3 默认配置一致）

**理论最优 K**:
```
K_optimal ≈ -1 / ln(α)

α = per-position acceptance probability
- Easy (α≈0.8): K_opt ≈ 4.5 → K=5 合理
- Hard (α≈0.3): K_opt ≈ 0.8 → K=1-2 最优
```

**为什么 Parallel 的 K 可以更大**:
- Parallel 的 draft cost = O(1)，不随 K 增长
- 额外的 head 即使 accuracy 低，也不增加延迟成本
- 只要有 1 个 head 猜对，就是净收益

**为什么 AR 的 K 要更小**:
- AR 每多猜一个 token，多一次 draft forward 的延迟
- 后面位置的 acceptance 指数衰减: α^k → pos3 只有 7.2%
- 多出的 draft steps 大概率被拒绝，白白浪费时间

**动态 K 进阶方案**（待实现）:
1. **EMA-based**: 根据最近接受率自适应调整 K
2. **Confidence-based**: draft 时实时检查 draft model confidence，低于阈值即停
3. **EAGLE-2 风格**: 基于树形 draft 结构，动态扩展/剪枝

---

## 9. 整体分析

### 9.1 当前结论

| 结论 | 证据 |
|------|------|
| ✅ 双模态假设成立 | Dip test p=0.000, GMM BIC 改善 1070 |
| ✅ Router 机制有效 | DAHD (1.12x) > Parallel (1.00x) |
| ✅ EAGLE-3 AR 强劲 | 1.37x speedup, 稳定可靠 |
| ⚠️ Parallel 分支训练不足 | head_1 仅 12.9%，需要 10-50x 更多训练 |
| ⚠️ DAHD 尚未超越 EAGLE-3 | 因 Parallel 分支太弱 |

### 9.2 为什么 DAHD 还没赢

DAHD 要超越 EAGLE-3，需要 Parallel Branch 满足:

```
在 Easy tokens 上:
  Parallel_acceptance × (1 / parallel_latency) > EAGLE3_acceptance × (1 / ar_latency)
```

即：并行的「低延迟」必须补偿「低精度」。

当前 Parallel head_1 只有 12.9% accuracy → 几乎只能接受 1 个 token（head_0: 81.4%）→ 没有并行优势，因为 EAGLE-3 平均每步也能接受 1 个 token。

**定量分析**:
- EAGLE-3: 平均 1.0 accepted/step, K=5 次 draft forward
- Parallel: 平均 0.08 accepted/step, 1 次 draft forward
- 结论: Parallel 并行优势（1次前向）远不足以弥补极低的 acceptance rate

### 9.3 达成目标需要

1. **Parallel head_1 accuracy > 40-50%**
   - 需要更多训练数据（当前 145K samples，论文标准 10M+ tokens）
   - 需要更多 epochs（当前 3 epochs，建议 20+）
   - head_1 在 epoch 3 已从 1.7% 提升到 12.9%，趋势向好

2. **或使用更强的并行架构**
   - Gumiho-v2: 添加 attention layers 到各 head
   - Hydra with shared bottom: 共享底层提取更好的特征

3. **K 动态优化**
   - Hard 时 K=1-2（避免无效 draft）
   - Easy 时 K=6-8（Parallel 不增加延迟）

### 9.4 下一步计划

- [ ] 增加训练数据量（1M+ samples）并训练 20+ epochs
- [ ] 实现动态 K 选择（EMA-based 或 confidence-based）
- [ ] 尝试 tree-based verification（同时验证多条 draft 路径）
- [ ] 在更多 benchmark 上评估（GSM8K, HumanEval, MT-Bench）
- [ ] 探索 Gumiho-v2 架构（各 head 加入 attention 层）
- [ ] 实现 DifficultyProbe + EMA 融合的 Router 方案

---

## 10. 复现指南

### 环境要求

```bash
pip install torch>=2.0 transformers>=4.36 accelerate scipy numpy matplotlib seaborn pandas pyarrow datasets pyyaml
```

### Phase 1: 运行 Acceptance Rate Profiling

```bash
python experiments/phase1_profiling/run_eagle3_full_profiling.py \
    --target_model /mnt/nas1/hf/Qwen3-8B/ \
    --eagle_model /mnt/nas1/hf/qwen3_8b_eagle3/ \
    --data_path data/eagle_data.jsonl \
    --num_prompts 50 \
    --output_dir results/phase1_results_v2
```

### Phase 2: 训练 Gumiho Parallel Heads

```bash
# Step 1: 生成训练数据
python experiments/phase2_architecture/generate_training_data.py \
    --model /mnt/nas1/hf/Qwen3-8B/ \
    --output_dir data/training

# Step 2: 训练 Gumiho
python experiments/phase2_architecture/train_gumiho_heads.py \
    --hidden_size 4096 \
    --vocab_size 151936 \
    --num_heads 5 \
    --epochs 3 \
    --output_dir checkpoints/gumiho
```

### Phase 4: 端到端对比

```bash
python experiments/phase4_e2e/run_e2e_comparison.py \
    --target_model /mnt/nas1/hf/Qwen3-8B/ \
    --eagle_model /mnt/nas1/hf/qwen3_8b_eagle3/ \
    --gumiho_ckpt checkpoints/gumiho/gumiho_best.pt \
    --num_prompts 50 \
    --max_new_tokens 128 \
    --output_dir results/phase4_e2e
```

---

## License

This project is licensed under the MIT License.
