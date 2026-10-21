NN_FRI · SAVEX15 144步物理环境修复 (Stage A / v1)
==================================================

目的：修复旧 Plan-A 的物理声速输入与环境数据时间泄漏；在完全新目录重算
144个10分钟时间步、两套几何的 KRAKEN 环境、模式根与真实 KRAKEN 模态形状。
此次先不运行 CSMT，旧 RMSE 只能作为历史探索结果，不能与新实验混写。

【已经完成的离线源文件审计】
- 144个明确UTC抽样时刻对应的源观测数据已核验，两个时刻各1个传感器标记。
- 选择原则预先锁定：最晚的、距北系泊5km内、测深≥95m、严格早于
  2015-05-20的合格CTD；现有归档自动选中CTD12 (05-18 23:45 UTC)。
- CTD12距离北系泊约0.344km；实测末端约97.29m；超过其末端作常值延拓。
- 当前压缩包已经包含必要的三份经QC后的历史输入数据，不需要你重新找文件。
- 开发环境未安装真实GSW或KRAKEN二进制，已做输入完整性检查、
  用明确标记为非物理的测试替身跑通288例数据管道/ZIP导出，
  还用历史49Hz真实.MOD验证了解析格式；300Hz实求解需要你本地运行。

【重要边界】
SAVEX15 在这个实验里提供真实温度/盐度环境观测，不提供同步真实声学数据。
模型生成的KRAKEN波数是“给定环境假设的数值参考”，不是实际海洋
瞬时模态波数。假定砂质海底 (1650m/s、1.9g/cm3、0.5dB/波长)
沿用旧Plan-A，为前后可比而固定；不是用海试测得的海底参数。
“40m设计几何”是反事实设计条件，不是SAVEX站点水深。
如果模态阶数变化，程序保留每帧全部根并使用显式长度掩码；
后续需单独进行模态形状重叠/分支连续性检查，不能直接把阶数等于身份。

【1. 解压】
将整个ZIP解压到新的目录，例如：
    D:\NN_FRI_SAVEX15_Rebuild\
不必覆盖原项目，且请勿删除原实验文件。
解压后该目录应有Python脚本、INPUT_PROVENANCE.json、outputs\savex15_qc_v1\的三份数据。

【2. 确认环境】
在Windows的CMD中输入：
    cd /d D:\NN_FRI_SAVEX15_Rebuild
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe -c "import numpy,pandas,gsw; print('GSW OK',gsw.__version__)"
如缺少gsw，再在同一个已有虚拟环境中执行：
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe -m pip install gsw

【3. 只检查QC输入，尚不进行物理计算】
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe NN_FRI_SAVEX15_Physics_Repair_v1.py --inspect-inputs
预期：selected CTD12；144 picks；当前归档有2个QC标记抽样时刻。
若不符请停止，截图给我。

【4. TEOS-10修复并生成288个环境文件】
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe NN_FRI_SAVEX15_Physics_Repair_v1.py --prepare-only
运行结束后，会生成全新 outputs\savex15_stageA_physics_repaired_v1\ 目录。
该步骤实际采用GSW公式：
    SP + p + lon/lat -> SA,  SA + t + p -> CT,  SA + CT + p -> sound_speed
它会保存静态预日CTD盐度、当时系泊温度、逐帧SSP和每例.env指纹。

【5. 仅用第一例测试KRAKEN】
以下是先前已可用的KRAKEN路径；如实际安装地址不同，替换exe路径即可。
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe NN_FRI_SAVEX15_Physics_Repair_v1.py --run-kraken --limit 1 --kraken-exe "D:\at\Acoustics-Toolbox-main\Kraken\kraken.exe"
初次测试应看到 1/288 stamp，且不应报 MOD 二进制格式错误。
若MOD布局与以前不同、或KRAKEN输出错误，立即停止，把第一例目录里的
.env/.mod/.prt/solver_stdout.txt/solver_stderr.txt 上传，不要手工改参数。

【6. 第一例成功后，执行全288例】
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe NN_FRI_SAVEX15_Physics_Repair_v1.py --run-kraken --kraken-exe "D:\at\Acoustics-Toolbox-main\Kraken\kraken.exe"
已成功且校验一致的案例会自动复用；无需重复运行第一例。
注意：全部288例求解可能耗时；运行结束看到两个144帧汇总时才算完成。
原旧Plan-A的结果不会被覆盖。

【7. 导出并上传最小审计ZIP】
    D:\NN_FRI_Mode_Extension\.venv\Scripts\python.exe NN_FRI_SAVEX15_Physics_Repair_v1.py --export-min
把下面这个文件上传给我：
    D:\NN_FRI_SAVEX15_Rebuild\outputs\savex15_stageA_physics_repaired_v1\NN_FRI_SAVEX15_StageA_AUDIT_EXPORT.zip
它包含所有环境文件、模式文件、运行SHA、关键PRT和144帧独立前向模态数组；
不用上传原始NetCDF，也不用上传整个D盘工程。

【下一阶段预注册计划】
收到真正的288例KRAKEN重算后，我先核验物理模式根是否随网格/截止变化，
再完成Stage B：用独立KRAKEN模态形状生成声场，检查真实分支连续性，
消除整个144帧真值用于初始化/调超参的信息泄漏，使用冻结代码并多种子对照。
这里不保证动态SSP一定好于保持初值；101m几何此前曾出现反例。
SACLANT历史实测可辨识性分析作为单独证据线保留，不会被这次SAVEX仿真代替。
