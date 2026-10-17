"""Network and cost model shared by the simulator and the estimate tool."""
from __future__ import annotations

import math

import numpy as np

from ..contracts import NodeSpec, TaskSpec


class Network:
    """Device->edge access links, edge<->edge metro links, time-varying WAN.

    WAN bandwidth follows a diurnal curve plus AR(1) noise, sampled once per
    second of simulated time. ``drops`` injects outages for stress tests:
    a list of (start_s, end_s, factor) that multiplies the bandwidth.
    """

    def __init__(self, cfg: dict, horizon_s: float, seed: int = 0,
                 drops: list[tuple[float, float, float]] | None = None,
                 day_length_s: float = 3600.0):
        n = cfg["network"]
        self.cfg = n
        self.access_mbps = n["access_mbps"]
        self.metro_mbps = n["metro_mbps"]
        self.access_rtt = n["access_rtt_ms"] / 1000
        self.metro_rtt = n["metro_rtt_ms"] / 1000
        self.wan_rtt = n["wan_rtt_ms"] / 1000
        rng = np.random.default_rng(seed)
        steps = int(math.ceil(horizon_s)) + 2
        tt = np.arange(steps)
        # A compressed "day": the trace window is replayed faster than real time,
        # so the diurnal cycle is compressed with it.
        diurnal = 1 - n["wan_diurnal_amp"] * 0.5 * (1 + np.sin(2 * np.pi * tt / day_length_s))
        noise = np.zeros(steps)
        for i in range(1, steps):
            noise[i] = 0.95 * noise[i - 1] + rng.normal(0, n["wan_noise"] * math.sqrt(1 - 0.95 ** 2))
        bw = n["wan_mbps"] * diurnal * (1 + noise)
        for start, end, factor in drops or []:
            bw[int(start):int(end) + 1] *= factor
        self.wan_series = np.clip(bw, 1.0, None)

    def wan_mbps(self, t: float) -> float:
        return float(self.wan_series[min(int(t), len(self.wan_series) - 1)])

    def transfer_s(self, mb: float, src_edge: int, node: NodeSpec, t: float) -> float:
        """Time to move ``mb`` from the device behind ``src_edge`` to ``node``."""
        s = self.access_rtt + mb * 8 / self.access_mbps
        if node.tier == "cloud":
            s += self.wan_rtt + mb * 8 / self.wan_mbps(t)
        elif node.node_id != src_edge:
            s += self.metro_rtt + mb * 8 / self.metro_mbps
        return s

    def hop_s(self, mb: float, a: NodeSpec, b: NodeSpec, t: float) -> float:
        """Time to move ``mb`` between two nodes (used for migration)."""
        if a.node_id == b.node_id:
            return 0.0
        if "cloud" in (a.tier, b.tier):
            return self.wan_rtt + mb * 8 / self.wan_mbps(t)
        return self.metro_rtt + mb * 8 / self.metro_mbps

    def tx_energy_j(self, mb: float, src_edge: int, node: NodeSpec, t: float) -> float:
        e = self.cfg["p_tx_edge_w"] * mb * 8 / self.access_mbps
        if node.tier == "cloud":
            e += self.cfg["p_tx_wan_w"] * mb * 8 / self.wan_mbps(t)
        elif node.node_id != src_edge:
            e += self.cfg["p_tx_edge_w"] * mb * 8 / self.metro_mbps
        return e


def exec_s(task: TaskSpec, node: NodeSpec) -> float:
    return task.work_gcycles / node.freq_ghz


def exec_energy_j(task: TaskSpec, node: NodeSpec) -> float:
    return node.p_core_w * exec_s(task, node)


def money_usd(task: TaskSpec, node: NodeSpec, energy_j: float, cfg: dict) -> float:
    """Cloud: core-seconds x price + egress. Edge: electricity for the energy used."""
    if node.tier == "cloud":
        return (exec_s(task, node) / 3600 * node.price_core_hour
                + task.output_mb / 1024 * cfg["network"]["egress_usd_per_gb"])
    return energy_j / 3.6e6 * cfg["pricing"]["electricity_usd_per_kwh"]


class Objective:
    """Weighted cost J of one finished task (lower is better).

    J = w1 * T/T_ref + w2 * E/E_ref + w3 * F/F_ref + penalty * missed

    The reference scales are medians of the standalone (empty-system,
    source-edge) values over the workload, so J is unit-free and a task that
    runs alone on its own edge scores about w1 + w2 + w3.
    """

    def __init__(self, cfg: dict, t_ref: float, e_ref: float, f_ref: float):
        o = cfg["objective"]
        self.w = (o["w_latency"], o["w_energy"], o["w_cost"])
        self.penalty = o["miss_penalty"]
        self.ref = (max(t_ref, 1e-9), max(e_ref, 1e-9), max(f_ref, 1e-12))

    @classmethod
    def calibrate(cls, cfg: dict, tasks: list[TaskSpec], nodes: list[NodeSpec],
                  net: Network) -> "Objective":
        ts, es, fs = [], [], []
        for task in tasks:
            node = nodes[task.source_edge]
            t = net.transfer_s(task.input_mb + task.output_mb, task.source_edge, node, 0) + exec_s(task, node)
            e = exec_energy_j(task, node) + net.tx_energy_j(task.input_mb, task.source_edge, node, 0)
            ts.append(t)
            es.append(e)
            # Edge money is tiny, so the cost reference is the cloud price of the same task.
            fs.append(money_usd(task, nodes[-1], e, cfg))
        return cls(cfg, float(np.median(ts)), float(np.median(es)), float(np.median(fs)))

    def __call__(self, latency_s: float, energy_j: float, cost_usd: float, missed: bool) -> float:
        w1, w2, w3 = self.w
        tr, er, fr = self.ref
        return (w1 * latency_s / tr + w2 * energy_j / er + w3 * cost_usd / fr
                + (self.penalty if missed else 0.0))
