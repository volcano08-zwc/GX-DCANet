# V121 运行 26BCI 数据集

这套代码与原 BCI-2a 实现并列，未覆盖 `lynet_2a.py`、`LYNet.py` 或 `train_lynet_2a.py` 的行为。

## 文件对应关系

```text
V121_FastFP32/
├── config/
│   ├── lynet_2a.yaml              # 原 2a 配置
│   └── lynet_26bci.yaml           # 26BCI：16×500、二分类、batch=64、500 epoch
├── data/
│   ├── lynet_2a.py                # 原 2a 数据处理
│   └── lynet_26bci.py             # 26BCI 读取、session-RUN-EA、功率和划分工具
├── model/
│   ├── LYNet.py                   # 原 22 通道/4 类模型
│   └── LYNet26BCI.py              # 16 通道/2 类/500 Hz 模型
├── protocol/
│   ├── lynet_protocol.py          # 原 2a 协议
│   └── lynet_26bci_protocol.py    # 26BCI 分组验证、早停、外层测试协议
└── train_lynet_26bci.py           # 26BCI 训练入口
```

## 使用哪些数据

每位患者使用 `work/csanet_treatment_experiments_20260910/data/<G|X>/dataset.npz` 中的全部已导出窗：

- `x`: `[N,16,500]`，1 秒 EEG 窗，500 Hz；
- `y`: `0=静息/idle`，`1=主动/active_intent`；
- `group_index`: 标准数据对应完整 trial，治疗数据对应完整标注 segment；
- `domain`: `0=标准采集`，`1=治疗采集`；
- `fold`: 标准数据为 `-1`，治疗数据为原始固定折 `0/1/2`；
- `source_start_samples`: 窗在原始 session 中的起点。

标准数据中一个 trial 提供 26 个窗：连续 7 秒主动阶段得到 13 个窗，后续 7 秒 rest 得到 13 个窗，窗长 1 秒、步长 0.5 秒。治疗数据的一个 group 是一个完整人工审核 segment，长度可以不同，因此每个 group 的窗数不固定。

## session_id 如何作为 run

一个 `session_id` 对应一次连续采集、同一套电极状态和同一份原始 EEG，因此代码把一个 session 定义成一个 run。每个 run 独立估计一个白化矩阵，绝不把其他 session 的任务标签用于该矩阵。

EA 参考段来自原始连续 EEG，但只提取非任务区间：

- 标准 session：从唯一的 5 秒 `initial_prepare` 去掉首尾各 1 秒，留下中间 3 秒，切成 3 个不重叠的 1 秒参考窗；不使用 cue、主动、recovery 或 rest；
- 治疗 session：治疗范式没有标准化的 `initial_prepare`。代码排除 manifest 中的全部标注 segment，并在边界两侧各保留 1 秒 guard；同时避开采集中断点，然后从每个剩余安全间隙中央取 1 秒，最多 50 个；
- 真实数据检查结果：G 的 6 个 session 和 X 的 11 个 session 都取得了自身参考窗，没有使用患者级兜底参考。

参考窗执行与导出数据一致的通道重排、CAR、4–40 Hz 四阶 Butterworth 双向滤波，然后计算：

```text
R = 所有参考窗协方差的平均
R_reg = 0.9 R + 0.1 trace(R)/16 · I
W = R_reg^(-1/2)
X_EA = W X
```

`0.1` 收缩用于处理 CAR 引起的秩亏。每个样本同时保留原始预处理视图和 EA 视图。EA 视图再计算 6 个频带 × 16 通道的 96 维 log-power。

## 模型适配

`LYNet26BCI` 保留 V121 的双视图门控融合、可学习 Sinc 前端、CX 主干、受保护的 power residual 和成对增强，仅修改数据契约：

- 通道：22 → 16；
- 时间点：1000 → 500；
- 采样率：250 Hz → 500 Hz；
- 输出：4 类 → 2 类；
- power：528 维 → 96 维；
- Sinc 卷积核按采样率缩放为 `129/65/41/41` 点，保持与原模型接近的实际时间长度；
- 时间增强的重点窗从 2 秒改为 0.5 秒。

## 数据集划分与防泄漏

每位患者单独训练和报告结果。

1. 标准数据预训练
   - 直接读取该患者原模型目录中的 `split.json`；
   - G：104 个 train trial、4 个 validation trial；
   - X：36 个 train trial、6 个 validation trial；
   - 特征归一化统计量只由标准 base-train 窗拟合，之后冻结。

2. 治疗数据外层 3 折
   - 每次将原始 treatment fold `f` 完整保留为 test；
   - 另外两折只按完整 segment group 划分 inner-train 和 inner-validation；
   - 标准数据加入适配训练，不加入治疗测试；
   - inner-validation 只选择最佳 epoch，不接触 outer-test；
   - 选出 epoch 后，从同一个标准预训练 checkpoint 重新开始，用全部非测试数据训练固定 epoch 数，再评估 test fold；
   - 三个 test fold 拼成该患者唯一一次 OOF 治疗预测。

3. 防止滑窗泄漏
   - 同一 trial/segment 的所有重叠窗始终位于同一个集合；
   - 测试 fold 不参与归一化、epoch 选择或权重学习；
   - 导出 EEG 的双向滤波是在每个 1 秒窗内部独立完成，不会跨 train/test 边界滤波；
   - 新提取的 EA 参考也逐窗独立双向滤波；
   - EA 可使用测试 session 自身的无标签非任务参考，属于无监督的 session 校准，不读取测试标签。

训练损失按“域 → 类别 → group → 窗”分层加权，避免长治疗 segment 或多数类仅凭窗数支配训练。默认 batch size 为 64，标准预训练和治疗适配的最大 epoch 都为 500，patience 为 50。

## 运行方法

在仓库根目录运行：

```powershell
# 只检查数据、生成/验证 session-EA 缓存
python train_lynet_26bci.py --preprocess-only

# GPU 完整运行 G、X
python train_lynet_26bci.py

# 只运行一位患者
python train_lynet_26bci.py --cases G
```

服务器上的数据路径可用环境变量覆盖，无需改代码：

```bash
export BCI26_DATA_ROOT=/path/to/26BCI
python train_lynet_26bci.py --cases G,X
```

EA 缓存写入 `cache/26bci_v121/`，模型、逐折预测、split audit 和汇总指标写入 `output/V121_26BCI/`。这两个目录均已被 `.gitignore` 排除，不会把患者数据、模型权重或训练输出提交到 Git。
