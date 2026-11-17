# IGRS：代码与结论数据

本仓库配套论文 **Incumbent-guided screening for sequential distribution system restoration under cold load pickup**，用于理解方法、查阅工况结果和复算论文中的数值比较。

## 快速开始

使用 Python 3.12，在仓库根目录执行：

```text
python tools/verify_package.py
python tools/reproduce_results.py
```

两条命令仅依赖 Python 标准库，分别检查文件完整性、复算已有结果，不调用潮流求解器。本目录是精简补充资料，不包含论文图片和完整逐样点原始仿真归档；上述核验检查所提供结果的内部一致性，不代表重新运行物理仿真。

## 文件导航

- `code/snapshots/`：各研究对应的冻结代码。
- `code/configs/`：实验配置、工况及依赖版本。
- `data/results/`：全部工况的结论数据、438 套计划的结果和辅助分析。
- `data/manuscript_tables.json`：正文表格的数值内容。
- `inputs/`：小型模型输入、CLPU 参数及来源。
- `docs/result_map.csv`、`docs/code_map.csv`：论文结果和代码入口的对应关系。
- `tools/`、`manifests/`：结果复算和文件完整性核验。

24 个主试验工况、8 个 Direct Beam 工况、8 个有限参照工况、6 个 13 节点工况及 123 节点结果均保留，失败方案和零增益工况也保留。代码阅读方式见 `docs/reconstruction.md`；数据说明及引用来源见 [英文 README](README.md)。
