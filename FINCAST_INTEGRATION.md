# FinCast 时序大模型集成指南

## 📋 模型信息

**FinCast** 是一个专门用于金融时序预测的基础大模型，特点：

- **架构**: Decoder-only Transformer
- **训练数据**: 超过200亿金融时序数据点
- **核心技术**: 
  - PQ-Loss: 联合点预测和概率预测
  - Mixture-of-Experts (MoE): 跨领域专业化
- **预训练权重**: `v1.pth` (已下载)
- **官方仓库**: https://github.com/vincent05r/FinCast-fts
- **Hugging Face**: https://huggingface.co/Vincent05R/FinCast

## 🔧 集成步骤

### 步骤1: 克隆官方代码

```bash
cd /Users/kobal/Library/CloudStorage/OneDrive-s3wh/毕业设计/建模/src/models/FinCast

# 克隆官方仓库
git clone https://github.com/vincent05r/FinCast-fts.git

# 或者如果已经下载，将代码复制到FinCast目录
```

### 步骤2: 安装依赖

根据README，需要运行：
```bash
cd FinCast-fts
bash env_setup.sh
bash dep_install.sh
```

或者手动安装（根据项目需求）：
```bash
pip install transformers>=4.30.0
pip install einops
pip install accelerate
```

### 步骤3: 使用我们的集成代码

我已经为你创建了集成代码，包括：

1. **`src/models/foundation_models.py`**: FinCast模型封装类
2. **`config.py`**: FinCastConfig配置类
3. **`main.py`**: 支持 `--model_type FinCast` 参数

## 🚀 使用方法

### 快速开始

```bash
# 基础训练（微调FinCast）
python main.py \
    --model_type FinCast \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --experiment_name fincast_finetune

# 使用LoRA微调（推荐，节省显存）
python main.py \
    --model_type FinCast \
    --use_lora \
    --lora_rank 8 \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --experiment_name fincast_lora

# 冻结backbone，只训练预测头
python main.py \
    --model_type FinCast \
    --freeze_backbone \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --experiment_name fincast_head_only
```

### 命令行参数说明

#### FinCast专用参数

```bash
--model_type FinCast                # 使用FinCast模型
--fincast_model_path <path>         # FinCast权重路径（默认: src/models/FinCast/v1.pth）
--fincast_config_path <path>        # FinCast配置文件路径
--freeze_backbone                   # 冻结预训练的backbone
--use_lora                          # 使用LoRA微调
--lora_rank 8                       # LoRA秩（默认8）
--lora_alpha 16                     # LoRA alpha（默认16）
--lora_dropout 0.1                  # LoRA dropout（默认0.1）
```

#### 标准训练参数

```bash
--lookback_window 60                # 输入窗口长度
--forecast_horizon 5                # 预测步数
--num_epochs 20                     # 训练轮数
--batch_size 16                     # 批大小
--learning_rate 1e-4                # 学习率（FinCast建议1e-4到1e-5）
--weight_decay 1e-5                 # 权重衰减
--eval_interval 50                  # 测试集评估间隔
```

## 📊 实验配置建议

### 配置1: 全量微调（显存充足）

```bash
python main.py \
    --model_type FinCast \
    --lookback_window 60 \
    --forecast_horizon 5 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 1e-6 \
    --eval_interval 100 \
    --experiment_name fincast_full_finetune
```

**特点**:
- 更新所有参数
- 需要较大显存（建议16GB+）
- 训练时间较长
- 可能获得最佳性能

### 配置2: LoRA微调（推荐）

```bash
python main.py \
    --model_type FinCast \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lookback_window 60 \
    --forecast_horizon 5 \
    --num_epochs 30 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --eval_interval 100 \
    --experiment_name fincast_lora_r8
```

**特点**:
- 只训练少量参数（~1%）
- 显存需求小（8GB可运行）
- 训练速度快
- 性能接近全量微调

### 配置3: 仅微调预测头（快速验证）

```bash
python main.py \
    --model_type FinCast \
    --freeze_backbone \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --eval_interval 50 \
    --experiment_name fincast_head_only
```

**特点**:
- 仅训练最后的预测层
- 显存需求最小
- 训练最快
- 适合快速验证效果

## 🔍 模型架构说明

FinCast模型包含以下组件：

```python
FinCastModel
├── Embedding Layer          # 输入嵌入
├── Transformer Blocks       # 多层Transformer（带MoE）
│   ├── Self-Attention
│   ├── MoE Feed-Forward     # 混合专家
│   └── Layer Norm
├── Output Projection        # 输出投影
└── Prediction Head          # 预测头（可自定义）
```

### 微调策略对比

| 策略 | 训练参数 | 显存需求 | 训练速度 | 性能 |
|------|----------|----------|----------|------|
| 全量微调 | 100% | 高（16GB+） | 慢 | 最佳 |
| LoRA微调 | ~1% | 中（8GB） | 中 | 接近全量 |
| 仅预测头 | < 0.1% | 低（4GB） | 快 | 基线 |

## 📈 预期效果

基于FinCast论文结果，在金融时序数据上：

- **Zero-shot**: MAE ~0.05-0.08（无微调）
- **Few-shot** (10样本): MAE ~0.03-0.05
- **Full Fine-tune**: MAE ~0.02-0.03

你的数据（296只股票，431天）应该能达到或超过Few-shot性能。

## 🐛 常见问题

### Q1: 显存不足怎么办？

**A**: 尝试以下方法：
1. 使用LoRA微调: `--use_lora`
2. 减小batch size: `--batch_size 8`
3. 减小输入窗口: `--lookback_window 30`
4. 使用梯度累积: 在config中设置 `gradient_accumulation_steps=4`

### Q2: 如何选择学习率？

**A**: 根据微调策略：
- 全量微调: `1e-5` 到 `1e-4`
- LoRA微调: `1e-3` 到 `5e-4`
- 仅预测头: `1e-3` 到 `1e-2`

### Q3: FinCast模型找不到权重文件？

**A**: 确保权重文件在正确位置：
```bash
ls src/models/FinCast/v1.pth

# 如果不存在，手动下载
# 或使用 --fincast_model_path 指定路径
python main.py --model_type FinCast --fincast_model_path /path/to/v1.pth
```

### Q4: 如何对比FinCast和传统模型？

**A**: 运行批量实验：
```bash
# 传统LSTM
python main.py --model_type LSTM --experiment_name exp_lstm

# FinCast微调
python main.py --model_type FinCast --use_lora --experiment_name exp_fincast_lora

# 对比结果
python -c "
import pandas as pd
lstm_metrics = pd.read_csv('results/exp_lstm/metrics_comparison.csv', index_col=0)
fincast_metrics = pd.read_csv('results/exp_fincast_lora/metrics_comparison.csv', index_col=0)

print('LSTM结果:')
print(lstm_metrics)
print('\nFinCast结果:')
print(fincast_metrics)
"
```

## 📝 下一步计划

1. **运行快速测试**:
   ```bash
   python main.py --model_type FinCast --num_epochs 2 --batch_size 16
   ```

2. **对比基线模型**:
   - LSTM: `python main.py --model_type LSTM --num_epochs 20`
   - FinCast: `python main.py --model_type FinCast --use_lora --num_epochs 20`

3. **调优LoRA参数**:
   - 尝试不同rank: 4, 8, 16, 32
   - 尝试不同alpha: 8, 16, 32

4. **撰写论文**:
   - 对比Zero-shot、Few-shot、Full Fine-tune
   - 分析FinCast在股票预测上的优势

## 🎓 论文实验建议

### 实验1: Zero-shot vs Fine-tuned

```bash
# Zero-shot（加载预训练权重，不训练）
python main.py --model_type FinCast --num_epochs 0 --experiment_name fincast_zeroshot

# Fine-tuned
python main.py --model_type FinCast --use_lora --num_epochs 20 --experiment_name fincast_finetuned
```

### 实验2: LoRA Rank消融实验

```bash
for rank in 4 8 16 32; do
    python main.py \
        --model_type FinCast \
        --use_lora \
        --lora_rank $rank \
        --num_epochs 20 \
        --experiment_name fincast_lora_r${rank}
done
```

### 实验3: 与传统模型对比

```bash
# 运行所有基线
for model in LSTM GRU Transformer; do
    python main.py --model_type $model --num_epochs 20 --experiment_name exp_${model}
done

# FinCast
python main.py --model_type FinCast --use_lora --num_epochs 20 --experiment_name exp_fincast
```

## 📚 相关文档

- **FinCast论文**: CIKM 2025（待发布链接）
- **官方GitHub**: https://github.com/vincent05r/FinCast-fts
- **Hugging Face**: https://huggingface.co/Vincent05R/FinCast
- **本项目文档**: 
  - `NEW_FEATURES_GUIDE.md`: 新功能使用指南
  - `CHECKPOINT_AND_MONITORING.md`: Checkpoint和监控
  - `QUICK_TEST.md`: 快速测试

---

**准备好开始了！** 🚀

现在需要的步骤：
1. 等我完成代码实现
2. 安装FinCast依赖
3. 运行快速测试
4. 查看results目录下的结果
