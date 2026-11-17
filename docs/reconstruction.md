# Reading the implementation

Use `code_map.csv` to find the frozen source snapshot for each study and its matching `code/configs/` directory. Within a snapshot, `src/dnr/` contains the method implementation and `scripts/` contains study entry points. Retained tests and protocols describe that implementation. Study versions are kept separate so that different implementations are not silently mixed.

For the main comparison, begin with `code/snapshots/20261031T145450_585540Z_clpu_public_confirmation/scripts/run_clpu_public_confirmation.py`. Its `src/dnr/nature34_clpu_public_ablation_control.py` and related CLPU search modules implement the compared validation policies. For the 123-node candidate handoff, begin with `code/snapshots/20261107T153635_900471Z_nc123_igrs_handoff_acceptance/scripts/run_nature123_igrs_handoff.py` and `src/dnr/nature123_igrs_handoff.py` in the same snapshot.

`inputs/reconstruction_overlay/` contains small feeder inputs, imported CLPU parameters and fault partitions. `code/snapshots/nature123_model_v2/` contains the 123-node feeder source files. The recorded model retains the author's line lengths and the documented configuration-4 mutual-capacitance sign adjustment. Source acquisition metadata and the upstream license are included.

The main comparison uses four prespecified fault pairs and six CLPU events, seed 0, a 30-minute horizon, 60-second action slots and 30-second sampling. The summary script retains 10,000 crossed resamples with seed 20261024.

The two commands in the root README verify the supplied processed results only. Full raw simulator outputs and historical workspace trees are outside this compact companion. Frozen source files retain their original paths and source/run bindings; fresh physical execution requires a matching solver environment, path adaptation and acceptance checks. Do not disable numerical, topology, voltage or thermal checks when adapting the code.
