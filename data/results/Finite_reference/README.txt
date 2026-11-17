Finite-domain reference for transfer and selective whole-load pickup

Study identity and endpoint

Study: 20261106T152015_140215Z_ijepes_bounded_ref.
The completed independent audit reports PASS_COMPLETE for all eight cases.
The SHA-256 of completion_manifest.json is
e4d4479cd55e304a0cf07ebca064ed63489738150667ec87631f22ef15b7efc1.
The controller's earlier COMPLETE_PENDING_INDEPENDENT_AUDIT status is its
original pre-audit checkpoint. independent_audit/summary.json supplies the
subsequent final audit result. Both original records are preserved.

Design and operating domain

All eight previously used development cases c00-c07 are included. Each case
retains its original double fault, CLPU event, 68 loads, load multiplier 0.876,
alpha 0.5, equipment ratings and electrical limits. The objective integrates
normal supplied power over 30 one-minute slots. During the first five slots,
one command per slot may wait, close an eligible tie, or change the readiness
of the selected whole load. At most two distinct tie closures are allowed,
with no reopening or other line-switch operations. The remaining 25 slots
are WAIT. Zero-based slot 0 is t = 0 s; the fifth slot starts at 240 s.

The selected load is the initially unserved eligible controllable load with
the greatest original reference active power; full load name breaks ties.
Eligibility is determined by graph reachability through at most two legal
tie closures, without observing a new electrical result. Other loads retain
their initial readiness. The selected load can remain ready, be withdrawn
once while its bus is unenergized, and be restored once after its bus has
been energized. It can also remain withdrawn until horizon end. c01 has
no eligible load and retains its naturally switch-only domain.

Radiality, fault isolation, continuity of already served loads and monotonic
energization of controlled lines are enforced. Every passing schedule is
checked at the initial state and at slot offsets 0, 30 and 60 s for the full
30-minute horizon. Numerical convergence, isolation/topology, phase-voltage,
line-current and known transformer limits follow the frozen model. A reliable
first violation is enough to exclude a schedule; unexamined later samples
are not represented as successful. Feasibility refers to these prescribed
quasi-static samples and modeled limits, not continuous-time transients.

The action catalog was developed before this reference was executed, with
both legal-tree enumeration and full command-string enumeration agreeing.
The archived proposal in source/ records that development history. Its
pre-execution wording is historical; final results are given here.

All-case result table

Normal service energy in kWh. S denotes the switch-and-WAIT subset; J adds
the selected load's readiness actions. L and U coincide for both S and J in
every case. The final column restricts to actual withdrawal and reconnection.

Case Faults     Event Selected  PASS FAIL  S optimum  J optimum  J-S gain  Reconnect maximum
c00  l21/l26       30 s844        56    0   773.9460   773.9460     0.0000    760.1855
c01  l4/l30        20 none         6    0   771.3180   771.3180     0.0000    no candidates
c02  l10/l24        6 s890        16   15    31.0980   495.1444   464.0464    all 10 violate
c03  l21/l30        3 s844       126    0   772.8364   772.8364     0.0000    755.1047
c04  l11/l16       30 s844        31    0   745.2570   745.2570     0.0000    725.2331
c05  l17/l20       20 s860        31    0   762.5580   762.5580     0.0000    757.5648
c06  l17/l29        6 s844       126    0   770.2960   770.2960     0.0000    746.1184
c07  l12/l24        3 s890         6   25    31.0980    31.0980     0.0000    all 10 violate

No case or schedule is omitted. The 438 schedules comprise 398 PASS,
40 reliable FAIL and zero unresolved. By family: switch-only 93/10/0 of
103; withdrawn without reconnection 205/10/0 of 215; withdrawn and
reconnected 100/20/0 of 120 (PASS/FAIL/unresolved). "No candidates" and
"all violate" are distinct conditions, and neither is encoded as zero
service energy. tables/case_bounds.csv includes all optimum ties and exact
bounds; tables/all_438_plans.csv retains each individual disposition.

c02 mechanism and matched schedules

The unique joint-domain optimum is [3,19,0,0,0], followed by 25 WAITs.
Here 3 withdraws readiness of Load.s890 and 19 closes the 814-828 tie.
At t = 0 s the initially unserved load is held off; at t = 60 s other
loads are transferred. The original supplied loads remain served. Relative
to all-WAIT, 960.096 kW of normal demand is supplied for the remaining
29 minutes, adding 464.0464 kWh. s890 is not reconnected during the horizon.

With exactly the same tie timing, [0,19,0,0,0] violates at t = 60 s:
minimum energized voltage is 0.798389 p.u. at node 890.2 (limit 0.90),
and transformer xfm1 winding 1 reaches 578.69 kVA (limit 500 kVA).
For [3,19,0,0,3], reconnection at t = 240 s produces 0.816244 p.u.
at node 890.1 and 604.66 kVA at the same winding. All ten permitted
reconnection schedules in c02 have voltage and transformer violations.
tables/matched_best.json contains the matched actions and failure quantities.
Operations after t = 240 s are outside the free action window; these results
do not determine whether a later reconnection would be feasible or useful.

Bounds and interpretation

Within each declared finite set, L is the greatest energy of a fully checked
passing schedule. U is the maximum of L and the energy of every unresolved
schedule. Reliable failures are excluded. Complete classification and L = U
establish the finite-domain optimum under the specified model and samples.
An empty family or a family with no feasible schedule has no feasible optimum;
the corresponding CSV cells remain empty. High-budget heuristics elsewhere
in the paper are not assigned these finite-domain optimum guarantees.

The reference measures the value of action permissions. IGRS, Forward and
Beam were not run on this domain. Their older full-domain energies cannot be
subtracted from these restricted optima to form an optimality gap. All eight
cases are development cases. This enumeration does not estimate population
frequencies or provide an unseen generalization test.

Budget and observed cost

Each case ran in one cold process with an empty cache; no numerical evidence
was shared across cases. The per-case hard limit was 1,200 s and 4 GiB.
Identical solver inputs could be reused within a case while each logical
sample retained its own time and pickup history. All schedules were visited
in deterministic command-string order. No parameter change or retry occurred.

The archive records 36,183 requests and 10,402 unique logical samples;
25,781 semantic hits and 6,686 exact-input reuses leave 3,716 paid physical
evaluations. Of those evaluations, 3,710 passed and 6 were reliable failures.
Repeated plans can hit the same failed point, accounting for 40 failed plans
with 6 distinct failed physical inputs. Including initialization, numerical
engine entries totaled 14,888. The frozen bounds were 4,221 evaluations and
16,908 entries. The eight-case cold controller time was 670.6 s; the largest
individual cold task was 207.22 s. Saved-evidence audit time was 148.4 s.

Replaying the compact evidence

Run "python check_reference.py" from this directory with standard Python 3.
The script checks packaged input hashes, reconstructs every family count,
computes bounds and all optimum ties from the individual table, checks
catalog membership and exact recorded energy, and compares cost totals with
the completed audit. It performs no power flow or optimization. It verifies
the compact archive's internal numerical agreement, not the omitted original
per-phase physical arrays. Those were checked by the original independent
audit, whose eight reports and summary are supplied unchanged.

The full project is needed to replay the original physical audit or experiment.
The copied source scripts retain their original project dependencies and are
included for traceability; check_reference.py is the standalone replay entry.
The source manifest supplies original project-relative paths and SHA-256
values. The full run manifest binds additional original files that are too
large to include here. This compact supplement does not claim to contain the
complete simulator, feeder inputs or raw per-sample electrical logs.
