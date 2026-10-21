# S20 受控 VLA 几何扫描 — 数据说明

## 来源

    E:\Users\lenovo\NNFRI\04_往期论文稿\往期论文稿\批注10.4\controlled_vla_geometry_sweep_v1 (2).zip

解压后复制到本目录（`cache/` 下 100 个 `.npz` 未复制，原始 zip 保留完整副本）。

## 这是什么

一个**预注册的、协议锁定的 receiver-mask-only 审计**：

> Single-factor receiver-mask audit of NN-FRI initialization and complete CSMT sequence.

| 项目 | 值 |
|---|---|
| 实验名 | `controlled_vla_geometry_sweep_v1` |
| 协议 hash | `f11062362dacc92595fefba82c25cec009aa0c08bb56c3db4d29dac42c71ece9` |
| 脚本 SHA-256 | `7e8b5064e16a7f1945a8a3aa6afbfb93f310e0f51c3a2fbae11200935f325158` |
| canonical 协议 hash | `954a6be701d44324a4bde82339f0c945c5d09ef9833691d6c4f221de0b384a38` |
| canonical 环境 SHA-256 | `c42efbc93ba335fceadbef8e70b19aacd33b5cc47dbb2f50ee216bd59fa2befb` |
| dynamic helper SHA-256 | `9640c7a536afd0ffb27c604dd1c40c355181ae2f3d84d4feb6cd47338617b9a6` |
| seed namespace | `controlled_vla_geometry_sweep_v1_20261005`（**独立于 Stage-B**） |
| trials | 20 ｜ K = 3 ｜ success 阈值 1.5e-3 rad/m |
| 关键开关 | `same_full_array_Y_then_mask_rows = true`、`regenerate_noise_after_masking = false`、`retune_any_algorithm_parameter_by_geometry = false` |
| 预注册声明 | "Frozen before the first receiver-mask trial is evaluated." |

**设计要点**：全阵列观测在每个 trial/frame 只生成一次，然后**按行掩码**取子阵。掩码后
**不重新生成噪声**，**不按几何重新调参**。因此该扫描把「接收器几何」从其他所有因素中
隔离出来。KRAKEN 分支仅作为**评分用的 model-reference**，**不是 field truth**。

## 四种几何

| mask | N | 深度范围 | 孔径 | 用途 |
|---|---|---|---|---|
| `full_40` | 40 | 1-40 m | 39 m | 基准 |
| `sparse_fullspan_20` | 20 | 1-40 m（稀疏） | 39 m | **固定孔径、只减接收器数** |
| `short_mid_20` | 20 | 11-30 m | 19 m | 接收器数与孔径同时减半 |
| `short_mid_14` | 14 | 14-27 m | 13 m | 1/3 孔径 |

`sparse_fullspan_20` 是**分离两个变量**的关键。

## 关键结果

| mask | 初始化 RMSE | 初始化 success | 跟踪 RMSE | 跟踪 success |
|---|---|---|---|---|
| `full_40` | 7.71e-4 | 0.700 | 2.27e-4 | 0.9950 |
| `sparse_fullspan_20` | 7.19e-4 | 0.750 | **2.95e-4** | **0.9925** |
| `short_mid_20` | 1.55e-2 | 0.000 | 1.27e-2 | 0.0500 |
| `short_mid_14` | 2.81e-2 | 0.000 | 2.43e-2 | 0.0000 |

配对差异（vs `full_40`，95% bootstrap 置信区间）：

| 对比 | 指标 | Δ | CI |
|---|---|---|---|
| `sparse_fullspan_20 - full_40` | 跟踪 RMSE | **+6.80e-5** | [5.69e-5, 7.92e-5] |
| `sparse_fullspan_20 - full_40` | 跟踪 success | **-0.0025** | [-0.0075, 0] |
| `short_mid_20 - full_40` | 跟踪 RMSE | +1.25e-2 | [8.31e-3, 1.66e-2] |
| `short_mid_20 - full_40` | 跟踪 success | -0.945 | [-0.975, -0.915] |
| `short_mid_14 - full_40` | 跟踪 RMSE | +2.41e-2 | [2.38e-2, 2.44e-2] |
| `short_mid_14 - full_40` | 跟踪 success | -0.995 | [-1, -0.9875] |

**结论：孔径效应约为接收器数效应的 180 倍。** 接收器数减半但保持全孔径时，跟踪性能
几乎不变（success 0.9950 → 0.9925）；孔径减半则崩溃（→ 0.0500）。

## 与论文的对应

| 位置 | 内容 |
|---|---|
| 补充材料 **S20** | "Receiver-mask sensitivity of the initializer and the tracking sequence" |
| **Table S13** | 四种几何的汇总表 |
| **Fig. S5** | `figures/03_csmt_tracking_rmse.png` → `Supp_FigS5_geometry_tracking_rmse.png` |

论文正文把该节明确标注为 receiver-mask 审计，并写出协议 hash 与 seed namespace。

## SHA-256 记录（本目录）

       9549d316e7f5e4b22ef9927df71e0a0b301fcc1135016441a25763725ae263c5  COMPLETE.marker  120 B
   9d4308482a699a306ed015f5e60875aa6e0570d5a2787e0ecbc96ccfff3e0b1c  csv\frame_metrics.csv  656347 B
   5e5b5f73236ecd2099dc72080a2cf8b1235eb85407efabf416fb9ba50a5c2f76  csv\geometry_protocol.csv  1465 B
   fffa39f795080f3443d09f9bc13bde121e54a9392fd6c37c991fad9ee70894ec  csv\geometry_summary.csv  2011 B
   23a02aa24c0482f9d788c8465f4a5e88fe8cd879ea1117c02bcb2e4b235ff41f  csv\paired_differences_vs_full.csv  1457 B
   6abf3e27f6c02dfb2c404bade05fef91097d5571f27442473d5f4625e5d12ea0  csv\trial_summary.csv  18853 B
   97b7308d7073bd6bd4bd7ff054a310efa8a69187eb3e001a7a9fa463a549c64c  EXPERIMENT_STARTED.marker  119 B
   cd023429b7847d55b9eadcef876bf93211b7f08e44c14b336ad03bdbd70808d4  figures\01_geometry_masks.png  108103 B
   140276515f4c4c08fef5a69f8777bf17920f26125010484589a0f5ce54603141  figures\02_static_initializer_rmse.png  436268 B
   771aab43835fb32f326daa8c74cf7b4d24a7ff83595848983d61d2aeb24d3163  figures\03_csmt_tracking_rmse.png  311472 B
   18c250f8e33eb39234b4f8f6f6ceaa5d9a7578eef5125f883b5877abee1fee9e  figures\04_success_proportions.png  73702 B
   cf50d08005407d4763a33b74b8faf3058ac8d8124b2f74e356ed538784d13ae3  PROTOCOL_LOCK.json  8461 B
   8eb575305d54f577e62d8ae16e000f8aa418086abd9cbb9dbd78f26bcb3cecfb  RESULT.json  8613 B
   abfe4beeb2d97e7acbc2c58854a4892b4121252939b5104f4b457ee28b0def10  RESULT_HASHES.csv  1210 B
   76d648e5affe799b548fb6791ddb682157dda4e32c397dacfb17765cbef69b4f  RESULT_SUMMARY.md  3105 B

原始包的 `RESULT_HASHES.csv` 另登记了 12 个文件的 SHA-256。

## 与已撤下实验的区别

本目录取代的旧材料在 `02_论文数据/S19_阵型敏感性/`（那里也有 README 记录撤下原因）。
旧材料缺少协议锁、`lock_sha256`、hash 清单，且种子命名空间与任何受审计协议都不一致；
本目录具备全部要件，因此可以进入论文。

## scripts/ 哈希记录

| 文件 | SHA-256 | 与协议锁 |
|---|---|---|
| `scripts/controlled_vla_geometry_sweep_v1.py` | `7e8b5064e16a7f1945a8a3aa6afbfb93f310e0f51c3a2fbae11200935f325158` | ✅ 等于 `locked_payload.script_sha256` |
| `scripts/dynamic_local_basis_coupling_suite.py` | `9640c7a536afd0ffb27c604dd1c40c355181ae2f3d84d4feb6cd47338617b9a6` | ✅ 等于 `locked_payload.dynamic_helper_sha256` |
| `scripts/find_sweep_script.py` | `2cd5422986e1dd97f2bceff973a633384afcaa6b3ea90d9806df50765b27aaa3` | 工具脚本，非实验件 |

**证据链状态**：协议锁 → 源码 → 原始输出（1680 行 `frame_metrics.csv`）→ 汇总 CSV → 图表 → 哈希清单，**全部闭环**。

