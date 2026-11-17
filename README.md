# IGRS: code and supporting results

This repository accompanies **Incumbent-guided screening for sequential distribution system restoration under cold load pickup**. It contains study-specific source code, model inputs and processed results for understanding the method and checking the numerical comparisons reported in the paper.

## Start here

Use Python 3.12 and run these commands from the repository root:

```text
python tools/verify_package.py
python tools/reproduce_results.py
```

Both commands use the Python standard library. They check file integrity and recompute summaries from the included processed results without running a power-flow solver. This compact companion contains no figure files or full raw simulation archives; these checks establish numerical consistency of the supplied results, not a new physical simulation.

## Contents

| Location | Contents |
|---|---|
| `code/snapshots/` | Frozen implementations corresponding to the reported studies |
| `code/configs/` | Study settings, cases and recorded dependency versions |
| `data/results/` | Case-level comparisons, all 438 finite-reference plans and supporting analyses |
| `data/manuscript_tables.json` | Numerical contents of the manuscript tables |
| `inputs/` | Small model inputs, CLPU parameters and source metadata |
| `docs/result_map.csv` | Paper-to-data navigation |
| `docs/code_map.csv` | Study-to-code navigation |
| `tools/`, `manifests/` | Processed-result checks and file integrity information |

All 24 main cases, eight Direct Beam cases, eight finite-reference cases, six 13-node cases and the 123-node delivery result are retained. Failed plans and zero-gain cases remain in their result tables. Start with `docs/result_map.csv` for results and `docs/reconstruction.md` for code.



## Sources and reuse

The feeder implementation derives from [ADAMSLab](https://github.com/adamslab-ub/Real-Time-Outage-Management-Active-DNR-GRL), commit `b715982a5abad85685e8ec629b2aeeb07a0aa376`. Its MIT license is retained in `licenses/ADAMSLab_LICENSE.txt`. CLPU inputs derive from [Hachmann et al.](https://doi.org/10.3390/en15207675); acquisition metadata are in `inputs/source_inventory/`. See `THIRD_PARTY_NOTICES.md` for the applicable license boundaries.

中文说明见 [README_zh.md](README_zh.md)。
