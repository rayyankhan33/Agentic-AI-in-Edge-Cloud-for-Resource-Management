import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecagents.agents.allocation_agent import AllocationAgent, LLMBatch, LLMChoice  # noqa: E402
from ecagents.agents.forecast_agent import Holt, Naive, rolling_eval  # noqa: E402
from ecagents.agents.nlp_agent import RuleBasedParser, intent_to_tasks  # noqa: E402
from ecagents.agents.orchestrator import OrchestrationAgent  # noqa: E402
from ecagents.config import build_nodes, load_config  # noqa: E402
from ecagents.contracts import TaskIntent, TaskSpec, TaskType  # noqa: E402
from ecagents.experiment import make_scenario  # noqa: E402
from ecagents.schedulers.heuristics import AllCloud, GreedyWeighted, LocalEdge, RandomPlacement  # noqa: E402
from ecagents.schedulers.milp import BatchMILP  # noqa: E402
from ecagents.sim.network import Network, Objective, exec_s  # noqa: E402
from ecagents.sim.simulator import Simulator  # noqa: E402


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(400)


def one_task(**kw):
    base = dict(task_id=0, arrival_s=0.1, work_gcycles=2.0, input_mb=1.0, output_mb=0.1,
                deadline_s=5.0, source_edge=0)
    base.update(kw)
    return TaskSpec(**base)


def test_intent_contract_rejects_bad_values():
    with pytest.raises(ValidationError):
        TaskIntent(task_type="video_analytics", input_size_mb=-1, deadline_ms=100)
    with pytest.raises(ValidationError):
        TaskIntent(task_type="teleport", input_size_mb=1, deadline_ms=100)


def test_single_task_latency_matches_model():
    cfg = load_config()
    nodes = build_nodes(cfg)
    net = Network(cfg, 100, seed=0)
    task = one_task()
    obj = Objective.calibrate(cfg, [task], nodes, net)
    m = Simulator(cfg, nodes, net, obj, LocalEdge(), epoch_s=0.5).run([task])
    expected = (0.5 - 0.1) + net.transfer_s(1.0, 0, nodes[0], 0.5) + exec_s(task, nodes[0]) \
        + net.transfer_s(0.1, 0, nodes[0], 0.5)
    assert m.n_tasks == 1
    assert m.mean_latency_s == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("sched", [LocalEdge(), AllCloud(), RandomPlacement(1), GreedyWeighted(), BatchMILP()])
def test_every_task_finishes(scenario, sched):
    m, sim = scenario.run(sched)
    assert m.n_tasks == len(scenario.tasks)
    for rec in sim.records.values():
        assert rec.finish_s >= rec.task.arrival_s
        assert 0 <= rec.node_id < len(scenario.nodes)


def test_greedy_beats_random(scenario):
    g, _ = scenario.run(GreedyWeighted())
    r, _ = scenario.run(RandomPlacement(0))
    assert g.mean_weighted_cost < r.mean_weighted_cost


def test_orchestrator_only_moves_waiting_tasks(scenario):
    m, sim = scenario.run(LocalEdge(), OrchestrationAgent())
    assert m.migrations > 0
    assert m.n_tasks == len(scenario.tasks)
    assert all(r.migrated <= 3 for r in sim.records.values())


class FakeLLM:
    available = True

    def __init__(self, node_id):
        self.node_id = node_id

    def generate_json(self, prompt, schema, system=""):
        import json
        tasks = json.loads(prompt)["tasks"]
        return LLMBatch(decisions=[LLMChoice(task_id=t["task_id"], node_id=self.node_id) for t in tasks])


def test_allocation_agent_falls_back_on_invalid_node(scenario):
    agent = AllocationAgent(FakeLLM(node_id=99), margin=10.0)   # escalate everything, answer invalid
    m, _ = scenario.run(agent)
    assert agent.n_escalated > 0
    assert agent.n_llm_ok == 0
    assert m.llm_fallback_rate == 1.0


def test_allocation_agent_uses_valid_llm_choice(scenario):
    cloud = scenario.nodes[-1].node_id
    agent = AllocationAgent(FakeLLM(node_id=cloud), margin=10.0, top_k=len(scenario.nodes))
    scenario.run(agent)
    assert agent.n_llm_ok > 0
    assert all(r["node_id"] == cloud for r in agent.trace if r["source"] == "llm")


def test_rule_parser_units():
    p = RuleBasedParser()
    i = p.interpret("Generate the report from the 2 GB log dump within 30 minutes, no rush")
    assert i.task_type == TaskType.BATCH_ANALYTICS
    assert i.input_size_mb == pytest.approx(2048)
    assert i.deadline_ms == pytest.approx(1_800_000)
    tasks = intent_to_tasks(i, first_id=10, arrival_s=1.0, source_edge=2)
    assert tasks[0].task_id == 10 and tasks[0].deadline_s == pytest.approx(1800)


def test_holt_tracks_trend():
    y = np.arange(100, dtype=float)
    assert Holt(0.5, 0.3).fit_predict(y, 1) == pytest.approx(100, abs=1.0)
    r_holt = rolling_eval(y, Holt(0.5, 0.3), warmup=10)
    r_naive = rolling_eval(y, Naive(), warmup=10)
    assert r_holt["mae"] < r_naive["mae"]
