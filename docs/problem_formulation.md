# Offloading objective (formal)

**System.** Edge nodes `E = {0..3}` and one cloud region `c`; node `n` has `C_n` cores at `f_n` GHz,
busy-core power `P_n`, and (cloud only) price `π_c` USD per core-hour. Devices attach to a home edge
`s(i)`. The device–edge access link, edge–edge metro links and the edge–cloud WAN have bandwidths
`B_acc`, `B_met`, `B_wan(t)` and RTTs `r_acc`, `r_met`, `r_wan`; `B_wan(t)` varies over time
(diurnal pattern, AR(1) noise, injected outages).

**Task.** Task `i` arrives at `a_i` with work `w_i` (giga-cycles), input `d_i` MB, output `o_i` MB
and relative deadline `D_i`.

**Decision.** `x_in ∈ {0,1}`, `Σ_n x_in = 1`: place task `i` on node `n` (and, at runtime, the
orchestrator may re-assign tasks that are still queued).

**Latency.** `T_i = (start_i − a_i) + w_i / f_n + τ(o_i, s(i), n)`, where
`τ(d, s, n) = r_acc + 8d/B_acc` plus `r_met + 8d/B_met` if `n` is another edge, or
`r_wan + 8d/B_wan(t)` if `n = c`; `start_i ≥ a_i + δ_i + τ(d_i, s(i), n)` waits for a free core
(FCFS), with `δ_i` the wait until the next decision epoch.

**Energy.** `E_i = P_n · w_i / f_n + Σ_links P_tx · 8(d_i + o_i)/B_link`.

**Monetary cost.** cloud: `F_i = π_c · (w_i/f_c)/3600 + egress · o_i`; edge: electricity price × `E_i`.

**Objective** (per task, normalised; minimised on average over the workload):

```
J_i = w1 · T_i/T_ref + w2 · E_i/E_ref + w3 · F_i/F_ref + ρ · 1[T_i > D_i]
min (1/|I|) Σ_i J_i   s.t.  Σ_n x_in = 1,  core capacity and FCFS order on every node
```

`(w1, w2, w3) = (0.5, 0.25, 0.25)`, `ρ = 1`. `T_ref`, `E_ref` are the median standalone latency and
energy of a task on its home edge, and `F_ref` the median cloud price of a task, so a task that runs
alone at home scores roughly `w1 + w2 + w3`. The form follows the base paper's weighted cost
(TF-DDRL, IEEE TSC 2025), adding queueing, a time-varying WAN and an explicit deadline penalty that
the base paper does not model.
