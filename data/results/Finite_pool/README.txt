Post hoc bounds on the attained normal service energy within the stored candidate set

Population and target
The 24 cases are the original four double-line faults crossed with six CLPU
events in the 34-node primary study. In each case the population consists
only of the complete schedules in the common P/E candidate pool plus that
case's verified starting Beam schedule. Normal service over 30 one-minute
slots is the sole objective examined here. The returned schedule gives a
lower bound because it passed the prescribed checks; energy from topology
and load-connection history supplies a potential value for each alternative.

First audit. The two workflows produced the same pool and selected the
same verified schedule in every pair. Using exact nominal-load energy for
the comparison, 160 (case, candidate) pairs had strictly higher energy
than the returned schedule before checking feasibility. The same plan was
already documented to fail for 127 of these pairs. The count concerns
case-specific schedules; it does not count independent operating events.

Second audit. A target candidate can also be excluded when it reproduces
the complete action prefix of a failed trajectory through the sampled
violation. The comparison uses the same fault, CLPU event, initial state,
action timing, state history, and operating model. The source record must
identify the relevant logical sample and a reliable physical constraint
violation. Sharing a final topology alone does not qualify. For 138 of
the 160 higher-energy pairs, a full-sequence failure or a matching failed
prefix was found, including 11 additional exclusions through prefix replay.
The upper bound in each case is the larger of the attained lower bound
and the greatest normal energy of any still-unresolved higher candidate.

The two bounds coincide in 19 of 24 cases. In the other five, 22 candidates
remain unresolved: case IDs 4, 5, 7, 10 and 11 have 1, 6, 6, 3 and 6,
respectively. The largest remaining gap is 508.3282 kWh in case 10. These
are upper bounds on possible improvements; the unresolved alternatives lack
positive feasibility evidence. Closure holds only within each stated finite
set at the prescribed samples. Other switch, waiting and pickup schedules
outside these sets remain unassessed. This audit used no new power-flow solves.
It is retrospective and does not change the main paired comparisons.

Independent use of this archive
Run "python check_pool.py" in the extracted Finite_pool directory (standard
Python 3). It matches the 160 exact action sequences across both evidence
lists, checks the included prefix records against candidate actions, and
recomputes per-case exclusions, remaining upper bounds and closure counts.
The higher_energy_index.csv supplies recorded graph energy values for just
these 160 candidates. Sequences are 30 commands encoded in the study's
original action space. No published solver, per-sample certification or
electrical simulation is called by this script.

The full original project is required to recompute candidate energies
from network connectivity, or to trace each prefix match back to the
physical certificate and original sampled trajectory. Those large phase
and logical-point logs are not contained here. The files in source/ are
exact copies of the two project audit scripts and the frozen graph module.
The project paths, sealed run identities and input digests can be found in
source_manifest.json, config.json, and evidence.json. To inspect or replay
the original audits, use those paths in a separate copy of the project;
the original scripts enforce archive hashes and exclusive output creation.
For self-contained numerical replay of the published bounds use check_pool.py.
