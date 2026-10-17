"""Structured contracts shared by the simulator, schedulers and agents.

Every message that crosses an agent boundary is one of these pydantic models,
so LLM outputs can be validated before they touch the simulator.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    VIDEO_ANALYTICS = "video_analytics"
    ML_INFERENCE = "ml_inference"
    BATCH_ANALYTICS = "batch_analytics"
    IOT_AGGREGATION = "iot_aggregation"
    WEB_REQUEST = "web_request"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TaskIntent(BaseModel):
    """Output of the NLP Interpretation Agent (what the user asked for)."""

    task_type: TaskType
    input_size_mb: float = Field(gt=0, le=10_000)
    deadline_ms: float = Field(gt=0, le=86_400_000)
    priority: Priority = Priority.NORMAL
    preferred_location: Literal["edge", "cloud", "any"] = "any"
    count: int = Field(default=1, ge=1, le=1000)
    max_cost_usd: Optional[float] = Field(default=None, ge=0)


class TaskSpec(BaseModel):
    """A schedulable unit of work inside the simulator."""

    task_id: int
    arrival_s: float = Field(ge=0)
    work_gcycles: float = Field(gt=0)        # CPU work in giga-cycles
    input_mb: float = Field(ge=0)            # data uploaded with the task
    output_mb: float = Field(ge=0)           # result returned to the device
    deadline_s: float = Field(gt=0)          # relative deadline from arrival
    source_edge: int = Field(ge=0)           # edge node the device is attached to
    task_type: TaskType = TaskType.BATCH_ANALYTICS
    priority: Priority = Priority.NORMAL
    preferred_location: Literal["edge", "cloud", "any"] = "any"


class NodeSpec(BaseModel):
    node_id: int
    name: str
    tier: Literal["edge", "cloud"]
    cores: int = Field(ge=1)
    freq_ghz: float = Field(gt=0)
    p_idle_w: float = Field(ge=0)           # idle power of the whole node
    p_core_w: float = Field(ge=0)           # extra power per busy core
    price_core_hour: float = Field(ge=0)    # USD per core-hour (cloud); 0 for edge


class NodeState(BaseModel):
    node_id: int
    tier: Literal["edge", "cloud"]
    busy_cores: int
    queue_len: int
    backlog_s: float              # estimated seconds of queued work per core
    utilisation: float            # busy_cores / cores


class SystemState(BaseModel):
    time_s: float
    nodes: list[NodeState]
    wan_bandwidth_mbps: float     # current edge<->cloud bandwidth
    recent_arrival_rate: float    # tasks/s over the last observation window


class Forecast(BaseModel):
    """Output of the Prediction & Forecasting Agent."""

    horizon_s: float
    arrival_rate: float           # predicted tasks/s
    wan_bandwidth_mbps: float     # predicted edge<->cloud bandwidth
    method: str


class CandidateEstimate(BaseModel):
    """What the estimate tool reports for placing one task on one node."""

    node_id: int
    tier: Literal["edge", "cloud"]
    est_latency_s: float
    est_energy_j: float
    est_cost_usd: float
    weighted_cost: float
    meets_deadline: bool


class PlacementDecision(BaseModel):
    """Output of the Resource Allocation (Offloading) Agent."""

    task_id: int
    node_id: int
    rationale: str = ""
    source: Literal["llm", "fallback", "heuristic", "milp", "drl"] = "heuristic"


class MigrationAction(BaseModel):
    """Output of the Observation & Orchestration Agent."""

    task_id: int
    from_node: int
    to_node: int
    reason: str


class EpisodeMetrics(BaseModel):
    scheduler: str
    n_tasks: int
    mean_latency_s: float
    p95_latency_s: float
    deadline_miss_rate: float
    total_energy_kj: float
    total_cost_usd: float
    mean_weighted_cost: float
    edge_share: float
    migrations: int = 0
    decision_ms_per_task: float = 0.0
    llm_calls: int = 0
    llm_fallback_rate: float = 0.0
