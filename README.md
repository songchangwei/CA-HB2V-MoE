# CA-HB2V-MoE

这是 `combined_all_reviewers_clean.pdf` 中 Condition-Aware Hierarchical
Band-to-View Mixture of Experts 的 PyTorch 实现，针对当前 IEMOCAP LOSO
数据与 WavLM-large 微调结果增加了按 fold 配对的训练入口。

`model.py` 的前向路径为：

`waveform -> WavLM-large + 三路 MFCC -> 共享因果膨胀卷积 -> Band-MoE -> View-MoE -> classifier`。

默认参数与文档一致：16 kHz、400 点窗、160 点 hop、三段频带
`[50,1000]`、`(1000,3000]`、`(3000,7600]` Hz，每带 20 个 Mel filter 和
13 个正交 DCT-II 系数。每带独立 CMVN，条件向量为五个预 DCT Mel 统计量。

## WavLM 微调策略

```python
from model import CAHB2VMoE

model = CAHB2VMoE(
    num_classes=4,
    wavlm_name="/path/to/current-finetuned-wavlm-large",
    # 若当前模型是 train_stage1.py 生成的完整 checkpoint，也可以传入：
    # wavlm_checkpoint="/path/to/stage1_checkpoint.pt",
)
```

模型调用现有仓库的 `WavLMBackbone`，并默认冻结：

- `wavlm.feature_extractor`：CNN 特征提取层；
- `wavlm.encoder`：24 层 Transformer 编码器。

`feature_projection`、WavLM 投影、MFCC 编码器、两个 dense softmax MoE 和分类器
保持可训练。可以用下面的代码核对冻结策略：

```python
for name, parameter in model.named_parameters():
    if name.startswith("backbone.wavlm"):
        print(name, parameter.requires_grad)
```

`forward` 返回 `CAHB2VMoEOutput`，包括 `logits`、`probabilities`、两级路由权重
`band_gates/view_gates`、条件统计和 `load_balance_loss`。训练损失可直接写为：

```python
output = model(waveform, waveform_lengths=lengths)
loss = cahb2v_loss(output, labels, model.load_balance_weight)
```

其中 `waveform` 是 `[batch, samples]`，`lengths` 是每条未填充波形的有效样本数。
`attention_mask` 也可以替代 `waveform_lengths`。

## 使用当前 LOSO 清单与微调权重

`train_loso.py` 会自动使用同一个外层折和验证 session 下的三个 manifest
及其 Stage-1 checkpoint。它优先选择 `best_stage1.pt`，缺失时按 Stage-1
验证排名选择仍存在的 top-k checkpoint；若该 case 没有任何 checkpoint，则报错，
不会静默回退到未微调权重。

当前权重目录检查到 18/20 个 case 有可用 checkpoint；
`loso_Ses03/val_session_Ses04` 与 `loso_Ses05/val_session_Ses03` 暂缺权重，
需先补跑对应 Stage-1 case 再运行 CA 模型。

先检查解析结果，再开始该 case 的训练：

```bash
python CA-HB2V-MoE/train_loso.py \
  --fold loso_Ses02 --val-session Ses05 --dry-run

python CA-HB2V-MoE/train_loso.py \
  --fold loso_Ses02 --val-session Ses05
```

如果希望直接指定某个已经微调好的 Stage-1 权重，可以使用
`--stage1-checkpoint`（`--wavlm-checkpoint` 为同义别名），例如：

```bash
python CA-HB2V-MoE/train_loso.py \
  --fold loso_Ses02 \
  --val-session Ses05 \
  --stage1-checkpoint \
  /home/user512-003/songcw/codex_vscode/scw_er/lcmr_tff_two_stage_project/outputs/1024/stage1_no_aug/loso/loso_Ses02/cases/val_session_Ses05/best_stage1.pt
```

指定单个 checkpoint 后，会跳过 Stage-1 根目录下的自动查找。

默认使用 `manifests/loso`、`outputs/1024/stage1_no_aug` 和项目旁边的
`pretrained_model/wavlm-large`。程序会自动进入其中的 `loso/` 子目录，并为每个
case 选择对应的微调 checkpoint。每个 case 的已解析配置、验证最佳模型、训练历史
和最终测试集 UA/WA 会保存到 `outputs/CA-HB2V-MoE/loso/<fold>/val_session_*/`。
测试清单不会参与训练或 checkpoint 选择。训练目录中还会生成 `train.log`，记录
参数量、每轮训练/验证指标、early stopping 和测试结果；训练、验证、测试阶段默认
显示 tqdm 进度条。将配置中的 `trainer.progress_bar` 设为 `false` 可关闭进度条，或
通过 `train.py --log-file /path/to/run.log` 指定日志文件位置。

每个 epoch 都会记录训练集和验证集的 UA、WA 与 loss。训练结束后按验证集 UA
降序保留 top-5 checkpoint（相同 UA 时用 WA 做确定性排序），并对每个保留模型
重新评估确定性的训练集、验证集和测试集。结果保存在：

```text
top5_checkpoints.json
top5_checkpoints.csv
top5_results.json
top5_train_metrics.json/csv
top5_val_metrics.json/csv
top5_test_metrics.json/csv
top5_rank_1/{train,val,test}/
```

每个 split 目录包含 `metrics.json`、`predictions.csv`、`records.jsonl` 和
`routing.json`。`records.jsonl` 保存逐样本标签、预测、类别概率以及 Band-MoE 和
View-MoE gate；`routing.json` 保存各专家 gate 的均值、标准差、熵和选择频率。

## 消融模型与可训练参数量

`ablations.py` 提供四个可直接用于 `train.py` 的模型变体，统一使用当前
微调的 WavLM，并冻结 CNN 特征提取器和 Transformer encoder：

| `model.variant` | 说明 |
| --- | --- |
| `wavlm_only` | WavLM utterance 表征 + 分类器 |
| `wavlm_fullband_mfcc` | WavLM + 全带 39 维 MFCC（静态、delta、delta-delta） |
| `standard_hierarchical_moe` | 标准 Band-MoE + 标准 View-MoE，不输入条件统计 |
| `ca_hb2v_moe` | 完整的条件感知 CA-HB2V-MoE |

在 `num_classes=4`、`d_model=256` 和 WavLM-large（hidden size 1024）下，
总参数量/可训练参数量分别为：WavLM-only `315,717,060 / 791,300`，
WavLM + 全带 MFCC `316,184,260 / 1,258,500`，标准双 MoE
`317,632,587 / 2,706,827`，完整 CA-HB2V-MoE `317,641,291 / 2,715,531`。
其中冻结的 WavLM CNN 与 Transformer 共 `314,925,760` 个参数。类别数变化
只会改变分类器参数量。

在训练配置中切换变体即可运行对应实验：

```yaml
model:
  variant: wavlm_only  # or wavlm_fullband_mfcc, standard_hierarchical_moe, ca_hb2v_moe
  num_classes: 4
  wavlm_name: /path/to/wavlm-large
  wavlm_checkpoint: /path/to/best_stage1.pt
```

`train.py` 启动时会打印参数量，并将相同内容写入输出目录的
`parameter_report.json`。也可以直接计算一个模型：

```python
from ablations import build_ablation_model, parameter_report
model = build_ablation_model("ca_hb2v_moe", num_classes=4, ...)
print(parameter_report(model))
```

当前 LOSO 训练设置使用 inverse-square-root 类别权重、验证 UA 选模、梯度裁剪、
early stopping，并对已微调的 WavLM feature projection 使用较低学习率。CNN 与
Transformer encoder 按要求冻结；共享 MFCC 编码器、两级 MoE、融合投影与分类头
保持训练。大 Stage-1 checkpoint 使用 mmap 加载，以避免把其优化器状态完整复制进内存。

## SAVEE 与 RAVDESS 训练入口

仓库提供两个数据集专用 launcher，自动读取已有的 speaker-independent/基于
actor 的 manifest，并为每个 case 写入完整的 `resolved_config.yaml` 后调用同一套
`train.py`。训练输出仍包含日志、tqdm 进度条、UA/WA、验证 UA 排序的 top-5
checkpoint，以及每个 top-5 模型的 train/val/test 预测和 Band/View-MoE 路由结果。

请在 `audio_torch` 环境中运行。SAVEE 的 Stage-1 权重默认从
`outputs/savee_stage1_no_aug/speaker_independent` 按 fold/case 自动匹配；也可以用
`--stage1-checkpoint` 直接指定某个权重：

```bash
# 先检查清单、权重和解析后的配置
/home/user512-003/miniconda3/envs/audio_torch/bin/python \
  CA-HB2V-MoE/train_savee.py \
  --fold fold_04_test_KL --case val_DC --dry-run

# 训练一个 SAVEE case（显式指定 Stage-1 权重）
/home/user512-003/miniconda3/envs/audio_torch/bin/python \
  CA-HB2V-MoE/train_savee.py \
  --fold fold_04_test_KL --case val_DC \
  --stage1-checkpoint outputs/savee_stage1_no_aug/speaker_independent/\
    fold_04_test_KL/cases/val_DC/best_stage1.pt \
  --output-root outputs/CA-HB2V-MoE/savee \
  --epochs 50 --batch-size 16 --num-workers 4

# 依次训练全部 12 个 SAVEE fold/case
/home/user512-003/miniconda3/envs/audio_torch/bin/python \
  CA-HB2V-MoE/train_savee.py --all \
  --stage1-root outputs/savee_stage1_no_aug/speaker_independent
```

RAVDESS 的 manifest 位于 `manifests/ravdess/ltspo`。如果没有对应的 RAVDESS
Stage-1 权重，必须显式传入 `--allow-base-wavlm` 才会允许使用基础 WavLM；有权重时
launcher 默认会从 `outputs/1024/ravdess_stage1_no_aug/ltspo` 按 fold/case
匹配；也可以改用 `--stage1-root` 或 `--stage1-checkpoint`：

```bash
# 使用默认 Stage-1 权重的 RAVDESS 单 case 训练
/home/user512-003/miniconda3/envs/audio_torch/bin/python \
  CA-HB2V-MoE/train_ravdess.py \
  --fold fold_01_test_01-04 --case val_05-08 \
  --epochs 50 --batch-size 16

# 使用已有 Stage-1 权重训练全部 30 个 RAVDESS fold/case
/home/user512-003/miniconda3/envs/audio_torch/bin/python \
  CA-HB2V-MoE/train_ravdess.py --all \
  --stage1-root /path/to/ravdess_stage1_root
```

两个入口都支持 `--variant wavlm_only`、`wavlm_fullband_mfcc`、
`standard_hierarchical_moe` 和 `ca_hb2v_moe`，以及 `--output-root`、`--dry-run`、
`--no-progress` 等通用参数。RAVDESS 的类别数自动设为 8，SAVEE 自动设为 7。
