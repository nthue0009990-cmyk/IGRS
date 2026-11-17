Joint-action adaptation diagnostics

call_outcomes.csv preserves all eight recorded solver calls for c02 and c07: two cold rolling-method development tasks, four fixed-wait diagnostics and two free-window diagnostics. The fields report the scenario, event, presolve setting, solver status, returned-vector status, elapsed time and AC-evaluation count.

These calls did not deliver a complete three-phase-validated restoration sequence. The fixed-wait feasible vectors were diagnostic results, not successful complete-plan baselines. The full solver matrices and detailed logs are outside this compact companion. The implementation modules chen_joint_domain.py and chen_joint_rolling.py are retained in the later frozen code snapshots.
