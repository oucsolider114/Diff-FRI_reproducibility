# 基于有界局部更新的模态水平波数因果序贯跟踪

一次全局 Diff-FRI（可微有限新息率）初始化，随后是由声速剖面变化驱动的有界局部更新（CSMT）。
在 40 m 合成波导中，全帧波数误差相对逐帧独立估计降低 **3.09 倍**；在 SWellEx-96 实验 S5
数据上，单次局部更新耗时 **0.108 s**，而全局重估需要 **102.58 s**。设计的真正边界是
**模态归属（association）**，而不是拟合残差。

> English version: [README_EN.md](README_EN.md)

## 快速开始

```bash
pip install -r requirements.txt
python experiments/cadence_audit.py      # 生成 data/swellex/cadence_audit.csv
bash figures/make_all.sh                 # 生成 figures/out/*.pdf
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
| `protocols/` | 实验参数（JSON） |
| `figures/` | 绘图脚本与输出的 PDF |
| `docs/` | 数据字典与复现说明 |

## 复现各图

| 图 | 命令 | 读取的数据 |
|---|---|---|
| 图 1（技术路线） | `python figures/fig1_flow.py` | 无 |
| 图 2（精度） | `python figures/fig2_accuracy.py` | `data/synthetic/*.csv` |
| 图 3（实测） | `python figures/fig3_swellex.py` | `data/swellex/csmt_fixed_ssp_i9606.csv`、`…/Table_final_experimental_summary.csv` |

论文中出现的**每一个数字**，要么直接来自 `data/` 下的某个 CSV，要么由 `figures/` 或
`experiments/` 中的脚本从 CSV 算出。没有任何计时值是硬编码的：图 3(b) 用到的两个中位数
是从实验汇总表里读的。

## 未包含的内容

| 项 | 原因 | 获取方式 |
|---|---|---|
| SWellEx-96 原始波形 | 567 MB，已公开存档 | <https://swellex96.ucsd.edu/s5.htm> |
| SAVEX15 环境数据 | 由 SEANOE 分发 | doi:10.17882/72285 |
| KRAKEN 可执行文件 | 第三方软件，不再分发 | 向该模型作者获取 |

压力场数组已随仓库提供（`data/arrays/Y_frames.npz`，456 KB），因此
`experiments/rerun_cadence_from_arrays.py` 可以在本仓库内直接重跑优化器，而不只是复算。

## 数据许可与第三方条款

代码采用 MIT 许可，派生的表格与图采用 CC BY 4.0。SWellEx-96 与 SAVEX15 原始数据仍受
各自条款约束；`data/` 中不包含任何原始波形。

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
