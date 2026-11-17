# Study implementations

`main34/` contains the primary 24-case implementation. `variants/` contains the additional or different source files needed by the other studies. Identical file contents are stored once. An individual variant directory is not a standalone source tree; `index.json` identifies every original path and the exact file version required by each study.

From the repository root, list the studies or restore a complete source tree:

```text
python tools/materialize_code.py --list
python tools/materialize_code.py --study main34
python tools/materialize_code.py --study feeder123_handoff
```

Restored files appear under `work/<study>/`, with their original relative names and bytes. Existing differing files are never overwritten. The implementation is not executed by this command. Matching study configurations are listed in `docs/code_map.csv`.
