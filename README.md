# 基于有界局部更新的模态水平波数因果序贯跟踪

**版本 v1.1.4（2026-10-27）** — 在 v1.1.3 基础上补入补充材料 S26 的 OCMS 对比实现
（`experiments/s26_ocms_comparison/`）、`src/mode_fri/direct_benchmark.py` 与
`configs/canonical_dynamic_benchmark_v1.json`，并在 `requirements.txt` 声明 `cvxpy`/`clarabel`；
这些文件在 v1.1.3 中缺失。归档可支持从派生数据复算论文报告的主要图表与汇总统计；部分实验级重跑需要外部输入与原始计算环境，三级复现范围见 `docs/REPRODUCE.md`。

一次全局 Diff-FRI（可微有限新息率）初始化，随后是由声速剖面变化驱动的有界局部更新（CSMT）。
在 40 m 合成波导中，全帧波数误差相对逐帧独立估计降低 **3.09 倍**；在 SWellEx-96 实验 S5
数据上，单次局部更新消耗 30 s 观测窗口的 **0.36%**，而全局重估需要 **342%**。设计的真正边界是
**模态归属（association）**，而不是拟合残差。

> English version: [README_EN.md](README_EN.md)

## 快速开始

```bash
pip install -r requirements.txt
python experiments/cadence_audit.py      # 重写 data/swellex/cadence_audit.csv（受哈希保护）
bash figures/make_all.sh                 # 输出到 figures/out_repro/，不覆盖归档图
```

只想看某一张图、不落盘：

```bash
python figures/fig2_accuracy.py --stdout > /tmp/fig2.pdf
```

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/mode_fri/` | 算法实现：试模态字典、Diff-FRI 初始化、CSMT 更新、参数配置 |
| `experiments/` | 四个实验的驱动脚本，加上重锚节奏审计 |
| `data/synthetic/` | 正文章节 3.1（图 2）的逐帧与逐实现导出 |
| `data/robustness/` | 章节 3.2 的单因子扫描 |
| `data/swellex/` | 章节 3.3–3.4（图 3）的逐帧导出与重锚审计 |
| `data/baselines/` | 表 I 的估计器类别对比 |
| `data/arrays/` | SWellEx 压力场数组（供实验级重跑，需原工程布局） |
| `data/savex15/` | Sec. S10：SAVEX15 驱动的前向场、SSP 与逐实现结果 |
| `data/vla_geometry/` | Sec. S20：受控 VLA 几何扫描的逐帧指标与协议锁 |
| `data/ablation_initialization/` | Sec. S6：三阶段初始化消融的 100-trial 记录 |
| `data/predictor_holdout/` | Sec. S18：固定系数预测器的独立 holdout |
| `protocols/` | 实验参数（JSON） |
| `figures/` | 绘图脚本与输出的 PDF |
| `docs/` | 数据字典与复现说明 |

## v1.1.0 新增数据集

### 1. `data/savex15/`（对应 Supplementary Sec. S10）

| 子路径 | 内容 | 规模 |
|---|---|---|
| `arrays/KRAKEN_INDEPENDENT_FORWARD_design_40.npz` | 40 m 前向场：`roots (144, 8)`、`receiver_mode_shapes (144, 40, 8)`、`c (144, 801)`、`receivers (40)` | 831 KB |
| `arrays/KRAKEN_INDEPENDENT_FORWARD_sav15_101.npz` | 101 m 前向场：`roots (144, 19)`、`receiver_mode_shapes (144, 21, 19)`、`c (144, 405)`、`receivers (21)` | 521 KB |
| `arrays/SSP_INPUT_design_40.npz` | 144 帧实测 SSP（40 m，0.5 m 网格） | 632 KB |
| `arrays/SSP_INPUT_sav15_101.npz` | 144 帧实测 SSP（101 m，0.25 m 网格） | 251 KB |
| `results/` | 20 个配对实现 × 2 构型的逐帧 CSV/JSON 与汇总（83 个文件） | 8.7 MB |
| `scripts/` | StageA（KRAKEN 前向）与 StageB（配对跟踪）脚本、预声明协议、输入溯源 | 68 KB |

`roots` 为模态水平波数参考，`receiver_mode_shapes` 为接收深度采样的模态深度函数，`c` 为逐帧声速剖面。
数组由 `scripts/NN_FRI_SAVEX15_Physics_Repair_v1.py` 生成，协议见 `scripts/StageB_PREDECLARED_PROTOCOL.md`。

**注意**：SAVEX15 的原始温盐观测由 SEANOE 分发（doi:10.17882/72285），本仓库不重复分发；
这里提供的是由该观测构造的 SSP 与前向场，足以复算 Sec. S10 的全部数字。

### 2. `data/vla_geometry/`（对应 Supplementary Sec. S20）

| 子路径 | 内容 |
|---|---|
| `csv/frame_metrics.csv` | 逐帧指标，**1680 行** = 20 配对 trial × 21 帧 × 4 个子阵 |
| `csv/trial_summary.csv`、`geometry_summary.csv`、`paired_differences_vs_full.csv`、`geometry_protocol.csv` | 逐 trial、逐几何与配对差的汇总 |
| `scripts/` | 几何扫描与动态局部基耦合脚本 |
| `PROTOCOL_LOCK.json`、`RESULT.json`、`RESULT_HASHES.csv`、`RESULT_SUMMARY.md` | 预声明协议锁、结果与哈希 |

四个子阵为 `full_40`(40 元/39 m)、`sparse_fullspan_20`(20 元/39 m)、`short_mid_20`(20 元/19 m)、
`short_mid_14`(14 元/13 m)。其中只有 `sparse_fullspan_20` 与 `short_mid_20` 保持元件数固定。

### 3. `data/ablation_initialization/`（对应 Supplementary Sec. S6）

| 子路径 | 内容 |
|---|---|
| `results/frame_metrics.csv` | 三阶段初始化消融的逐帧误差（387 KB） |
| `results/*_trial_summary.csv`、`*_aggregate.csv`、`paired_component_statistics.csv` | 逐 trial、聚合与配对统计 |
| `protocol/ABLATION_PROTOCOL_LOCK.json`、`ABLATION_RESULT.json` | 预声明协议锁与结果 |
| `REPORT.md` | 结果说明 |

对应论文中的 1.094 / 0.727 / 0.708 ×10⁻³ rad/m 与 88%/92%/92% 成功率，以及
“接受 22 次、其中 15 次改善、7 次恶化”与三阶段耗时 0.0203 / 0.1141 / 2.4417 s。

### 4. `data/predictor_holdout/`（对应 Supplementary Sec. S18）

| 子路径 | 内容 |
|---|---|
| `results/frame_metrics.csv` | 独立 holdout 的逐帧指标（171 KB） |
| `results/trial_summary.csv`、`downstream_*`、`paired_statistics.csv` | 逐 trial 与配对统计 |
| `protocol/CSMT_V2_PARAMETER_LOCK.json`、`HOLDOUT_RESULT.json` | 参数锁与结果 |
| `REPORT.md` | 结果说明 |

该 holdout 与主基准使用不同的随机种子命名空间，用于 β ∈ {0, 0.25, 0.5, 0.75, 1.0} 的
开环与闭环敏感性对比（论文“2.3% 变化”一句的来源）。

## 复现各图与各表

| 对象 | 命令 | 读取的数据 |
|---|---|---|
| 图 1（技术路线） | `python figures/fig1_flow.py` | 无 |
| 图 2（精度） | `python figures/fig2_accuracy.py` | `data/synthetic/*.csv` |
| 图 3（实测） | `python figures/fig3_swellex.py` | `data/swellex/csmt_fixed_ssp_i9606.csv`、`…/Table_final_experimental_summary.csv` |
| 表 I | — | `data/baselines/estimator_comparison.csv` |
| Sec. S10（SAVEX15） | 数据已包含；脚本需原工程布局（见 `docs/REPRODUCE.md` 的 Tier C） | `data/savex15/arrays/*.npz` |
| Sec. S20（VLA 几何） | 数据已包含；脚本需原工程布局（同上） | `data/vla_geometry/csv/*.csv` |

论文中出现的**绝大多数数字**，要么直接来自 `data/` 下的某个 CSV，要么由 `figures/` 或
`experiments/` 中的脚本从 CSV 算出；唯一例外是 Sec. S2 的 K=3/5/8 扫描，其逐 trial 记录未单独导出
（S2 已注明）。没有任何计时值是硬编码的：图 3(b) 用到的两个中位数是从实验汇总表里读的。

## 未包含的内容

| 项 | 原因 | 获取方式 |
|---|---|---|
| SWellEx-96 原始波形 | 567 MB，已公开存档 | <https://swellex96.ucsd.edu/s5.htm> |
| SAVEX15 原始温盐观测 | 由 SEANOE 分发 | doi:10.17882/72285 |
| SAVEX15 前向场的 KRAKEN 原始输出（`.prt`/`.mod`/`.env`） | 约 42 MB 中间文件，可由 StageA 脚本重建 | 本仓库提供其 `.npz` 结果 |
| KRAKEN 可执行文件 | 第三方软件，不再分发 | 向该模型作者获取 |

压力场数组已随仓库提供（`data/arrays/Y_frames.npz`，456 KB）。但
`experiments/rerun_cadence_from_arrays.py` 仍依赖原工程目录 `SWellEx96_S5_CSMT/`（KRAKEN 参考与 `.env` 输入）
以及 `dynamic_local_basis_coupling_suite` 模块，**无法在解压后的仓库中独立运行**，因此计入 Tier C。
`data/arrays/Y_frames.npz` 本身足以让读者自行编写重跑入口。

## v1.0.1 → v1.1.0 变更摘要

- 新增 `data/savex15/`（11 MB）：4 个前向场/SSP 数组、83 个逐实现结果、StageA/StageB 脚本与协议
- 新增 `data/vla_geometry/`（1.7 MB）：1680 行逐帧指标、4 个汇总表、3 个脚本、协议锁与哈希
- 新增 `data/ablation_initialization/`（0.46 MB）：Sec. S6 的逐帧与逐 trial 记录
- 新增 `data/predictor_holdout/`（0.23 MB）：Sec. S18 的独立 holdout 与 β 重算记录
- 更新 `MANIFEST.sha256`、`README`、`README_EN`、`CITATION.cff`
- 至此论文原本标注“realization-level records available on request”的两处数据已全部包含在本仓库内

## v1.1.3 → v1.1.4 变更摘要

- 补入补充材料 S26 的 OCMS 对比实现 `experiments/s26_ocms_comparison/`（v1.1.3 中完全缺失）。
- 补入 `src/mode_fri/direct_benchmark.py` 与 `configs/canonical_dynamic_benchmark_v1.json`；此前 `unified_benchmark.py` 与 `canonical_dynamic_benchmark_v1.py` 引用了归档中不存在的文件。
- `requirements.txt` 声明 S26 对比所需的 `cvxpy` 与 `clarabel`。
- `MANIFEST.sha256` 重建。

## v1.1.2 → v1.1.3 变更摘要

- `figures/fig1_flow.py`：流程框图中的虚线在标签处断开。
- `figures/fig3_swellex.py`：图 (a) 增加 7 个保留帧的灰色阴影带；图 (b) 纵轴标注对数刻度；图注同步更新。
- `CITATION.cff` 与两份 `README` 的成本表述由绝对耗时改为观测窗口比例，与论文正文口径一致。

## v1.1.1 → v1.1.2 变更摘要

- 通讯作者邮箱（`wangpengyu@ouc.edu.cn`）保留，与 Zenodo 记录页和论文首页保持一致。
- `README` 中英文的复现声明收窄为两级表述（派生数据复算 / 实验级重跑），与 `docs/REPRODUCE.md` 的 Tier B/C 一致。
- `CITATION.cff` 的引用提示改为“引用本归档；论文发表后可另行引用”，不再把软件 DOI 表述为期刊文章 DOI。

## v1.1.0 → v1.1.1 变更摘要

- 全部文本文件统一为 **LF** 换行，新增 `.gitattributes`（`* text=auto eol=lf`）；
  `MANIFEST.sha256` 按 LF 重算，GitHub、Zenodo 与本地三处现在一致。
- 修正 v1.1.0 的清单不匹配：该版清单基于 Windows 工作区的 CRLF 计算，
  而 release 归档为 LF，导致 136 条误报。
- `figures/make_all.sh` 改为输出到 `figures/out_repro/`，不再覆盖受清单保护的归档图。
- `docs/REPRODUCE.md` 的 Tier A 范围收窄，并注明 K=3/5/8 扫描无逐 trial 导出。

## 数据许可与第三方条款

代码采用 MIT 许可，派生的表格与图采用 CC BY 4.0。SWellEx-96 与 SAVEX15 原始数据仍受
各自条款约束；`data/` 中不包含任何原始波形或原始温盐观测。许可范围与第三方归属见
`DATA_LICENSE.md`。

## 完整性校验

`MANIFEST.sha256` 列出了每个受版本控制文件的 SHA-256。校验或重新生成：

```bash
find . -type f -not -path './.git/*' -not -name 'MANIFEST.sha256' \
  -exec sha256sum {} + | sort -k2 > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

## 引用

见 `CITATION.cff`。请同时引用论文与存档版本。

## 联系

王鹏宇 — wangpengyu@ouc.edu.cn
